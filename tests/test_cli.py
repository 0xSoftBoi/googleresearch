import json

import numpy as np
import pytest
import torch

from timesfm3 import TimesFM3Config, TimesFM3Model
from timesfm3.cli import main


@pytest.fixture
def csv_path(tmp_path, seasonal):
    p = tmp_path / "data.csv"
    lines = ["date,a,b"]
    for i, v in enumerate(seasonal):
        stamp = str(np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h"))
        lines.append(f"{stamp},{v:.4f},{2 * v:.4f}")
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_models_and_version(capsys):
    assert main(["models", "--no-bundled"]) == 0
    out = capsys.readouterr().out
    assert "ewma" in out and "last-value" in out
    with pytest.raises(SystemExit):
        main(["--version"])


def test_forecast_csv_stdout(csv_path, capsys):
    assert main(["forecast", csv_path, "--horizon", "3", "--model", "ewma", "--no-bundled",
                 "--columns", "b"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("step,timestamp,series,point,q10")
    assert len(lines) == 4 and lines[1].split(",")[2] == "b"
    assert lines[1].split(",")[1] == "2024-01-17T16:00:00"


def test_forecast_json_file(csv_path, tmp_path):
    out = tmp_path / "fc.json"
    assert main(["forecast", csv_path, "-H", "5", "-m", "drift", "--no-bundled",
                 "--freq", "D", "-o", str(out)]) == 0
    j = json.loads(out.read_text())
    assert j["model"] == "drift" and len(j["forecasts"]) == 2
    assert len(j["forecasts"][0]["point"]) == 5 and "q50" in j["forecasts"][0]["quantiles"]
    assert j["timestamps"][0] == "2024-01-18T15:00:00"


def test_backtest_table_and_json(csv_path, capsys):
    assert main(["backtest", csv_path, "--context", "96", "-H", "24", "--windows", "4",
                 "--models", "ewma,drift", "--no-bundled"]) == 0
    out = capsys.readouterr().out
    assert "ratio vs last-value" in out and "reference" in out and "ewma" in out
    assert main(["backtest", csv_path, "--context", "96", "-H", "24", "--windows", "4",
                 "--models", "ewma", "--no-bundled", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert {s["model"] for s in j["scores"]} == {"ewma", "last-value"}


def test_pack_and_serve_checkpoint(tmp_path, csv_path, capsys):
    cfg = TimesFM3Config.tiny()
    torch.manual_seed(0)
    raw = tmp_path / "raw.pt"
    torch.save({"config": cfg, "model": TimesFM3Model(cfg).state_dict()}, raw)
    packed = tmp_path / "packed.pt"
    assert main(["pack", str(raw), str(packed), "--name", "mine", "--description", "d"]) == 0
    assert "packaged" in capsys.readouterr().out
    assert main(["models", "--no-bundled", "-c", f"{packed}", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "* mine" in out
    assert main(["forecast", csv_path, "-H", "2", "--no-bundled", "-c", f"alias={packed}",
                 "--device", "cpu", "--columns", "a"]) == 0
    assert "a" in capsys.readouterr().out


def test_error_paths(tmp_path, capsys):
    assert main(["forecast", str(tmp_path / "missing.csv"), "-H", "2", "--no-bundled"]) == 1
    assert "error:" in capsys.readouterr().err
    p = tmp_path / "short.csv"
    p.write_text("a\n1\n2\n3\n")
    assert main(["backtest", str(p), "--context", "8", "-H", "2", "--no-bundled"]) == 1
    assert main(["forecast", str(p), "-H", "2", "--model", "ghost", "--no-bundled"]) == 1
