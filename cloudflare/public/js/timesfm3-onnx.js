/**
 * TimesFM-3 in the browser.
 *
 * Wraps the exported ONNX graph (scripts/export_onnx.py) with the same
 * pre/post-processing as timesfm3/forecaster.py: left-pad the context to a
 * patch multiple, crop history beyond the model's maximum context, roll the
 * decode forward for horizons beyond one pass (the model's own point
 * forecasts become context), denormalized outputs, and quantile-crossing
 * repair.  Nothing leaves the page: the model file is fetched once and every
 * forecast runs in WebAssembly on the visitor's machine.
 *
 *   const model = await TimesFM3Browser.load({ort, modelUrl, metaUrl}, onProgress);
 *   const r = await model.forecast({targets: [[...]], horizon: 48});
 *   r.point[0], r.quantiles[0][step][q]
 */

const ROLE_TARGET = 0n, ROLE_PAST = 1n, ROLE_FUTURE = 2n;

export class TimesFM3Browser {
  constructor(ort, session, meta) {
    this.ort = ort;
    this.session = session;
    this.meta = meta;
    this.patch = meta.patch_len || 32;
    this.maxContext = meta.max_context_len || 2048;
    this.maxHorizon = meta.max_horizon_len || 256;
    this.levels = meta.quantiles || [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
  }

  /** ort: the onnxruntime-web module; onProgress(loadedBytes, totalBytes). */
  static async load({ ort, modelUrl, metaUrl, wasmPaths }, onProgress) {
    if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
    ort.env.wasm.numThreads = Math.min(4, (navigator.hardwareConcurrency || 2));
    const meta = metaUrl ? await (await fetch(metaUrl)).json() : {};
    const resp = await fetch(modelUrl);
    if (!resp.ok) throw new Error(`model download failed: HTTP ${resp.status}`);
    const total = Number(resp.headers.get("content-length") || meta.onnx_bytes || 0);
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (onProgress) onProgress(loaded, total);
    }
    const buf = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    const session = await ort.InferenceSession.create(buf.buffer, {
      executionProviders: ["wasm"], graphOptimizationLevel: "all",
    });
    return new TimesFM3Browser(ort, session, meta);
  }

  async forecast({ targets, horizon, pastCovariates = [], futureCovariates = [], fixQuantileCrossing = true }) {
    if (!targets || !targets.length) throw new Error("At least one target series is required.");
    if (!(horizon > 0)) throw new Error("horizon must be positive.");
    const toArr = (s) => Float64Array.from(s, (v) => (v === null || v === undefined ? NaN : Number(v)));
    const evolving = [...targets, ...pastCovariates].map(toArr);
    const known = futureCovariates.map(toArr);
    const context = evolving[0].length;
    for (const s of evolving) if (s.length !== context) throw new Error("All targets and past covariates must share one context length.");
    for (const s of known) if (s.length !== context + horizon) throw new Error(`Future covariates must cover context + horizon (${context + horizon} steps).`);

    const nTargets = targets.length;
    const pointChunks = [], quantChunks = [];
    let produced = 0;
    let ev = evolving.map((s) => Array.from(s));
    while (produced < horizon) {
      const chunk = Math.min(horizon - produced, this.maxHorizon);
      const ctxLen = context + produced;
      const { point, quantiles } = await this._chunk(ev, nTargets, known.map((s) => Array.from(s.subarray(0, ctxLen + chunk))), chunk);
      pointChunks.push(point.slice(0, nTargets));
      quantChunks.push(quantiles.slice(0, nTargets));
      ev = ev.map((s, i) => s.concat(point[i]));
      produced += chunk;
    }
    const point = pointChunks[0].map((_, i) => pointChunks.flatMap((c) => c[i]));
    let quantiles = quantChunks[0].map((_, i) => quantChunks.flatMap((c) => c[i]));
    if (fixQuantileCrossing) quantiles = quantiles.map((rows) => rows.map((q) => [...q].sort((a, b) => a - b)));
    return { point, quantiles, quantile_levels: this.levels };
  }

  async _chunk(evolving, nTargets, known, chunk) {
    const P = this.patch, context = evolving[0].length;
    const paddedContext = Math.min(this.maxContext, Math.ceil(context / P) * P);
    const leftPad = paddedContext - Math.min(context, paddedContext);
    const crop = Math.max(0, context - paddedContext);
    const hp = Math.ceil(chunk / P);
    const total = paddedContext + hp * P;
    const n = evolving.length + known.length;
    const values = new Float32Array(n * total);
    const observed = new Uint8Array(n * total);
    const roles = new BigInt64Array(n);
    const put = (row, t, v) => { const i = row * total + t; if (Number.isFinite(v)) { values[i] = v; observed[i] = 1; } };
    evolving.forEach((s, row) => { roles[row] = row < nTargets ? ROLE_TARGET : ROLE_PAST; for (let k = crop; k < context; k++) put(row, leftPad + (k - crop), s[k]); });
    known.forEach((s, j) => { const row = evolving.length + j; roles[row] = ROLE_FUTURE; for (let k = crop; k < context; k++) put(row, leftPad + (k - crop), s[k]); for (let k = 0; k < chunk; k++) put(row, paddedContext + k, s[context + k]); });
    const T = this.ort.Tensor;
    const feeds = {
      values: new T("float32", values, [1, n, total]),
      observed: new T("bool", observed, [1, n, total]),
      roles: new T("int64", roles, [1, n]),
      horizon_patches: new T("int64", BigInt64Array.from([BigInt(hp)]), [1]),
    };
    const out = await this.session.run(feeds);
    const pt = out.point.data, qt = out.quantiles.data, Q = this.levels.length;
    const rows = evolving.length, start = paddedContext;
    const point = [], quantiles = [];
    for (let r = 0; r < rows; r++) {
      const p = new Array(chunk), q = new Array(chunk);
      for (let k = 0; k < chunk; k++) {
        p[k] = pt[r * total + start + k];
        const base = ((r * total) + start + k) * Q;
        q[k] = Array.from(qt.subarray(base, base + Q));
      }
      point.push(p); quantiles.push(q);
    }
    return { point, quantiles };
  }
}
