.PHONY: install dev test serve docker docker-run starter-model data clean

install:            ## core + serving extras
	pip install -e ".[serve]"

dev:                ## everything needed to hack on it
	pip install -e ".[serve,test,examples]"

test:
	python -m pytest -q

serve:              ## dashboard on http://localhost:8000
	timesfm3 serve --port 8000

docker:
	docker build -t timesfm3 .

docker-run:
	docker run --rm -p 8000:8000 timesfm3

data:               ## public benchmark datasets used to train the starter model
	bash data/download.sh

starter-model: data ## retrain the bundled checkpoint (~35 min on 4 CPU cores)
	python scripts/train_starter.py --steps 6000 --out timesfm3/assets/starter-small.pt

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# ---- Cloudflare edge front end (cloudflare/) ----
.PHONY: edge-install edge-dev edge-deploy
edge-install:
	cd cloudflare && npm install

edge-dev:           ## landing + gateway on :8787, proxying a local `timesfm3 serve`
	cd cloudflare && npx wrangler dev --port 8787 --var API_ORIGIN:http://localhost:8000 --var ADMIN_TOKEN:dev

edge-deploy:        ## needs CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID
	cd cloudflare && npx wrangler deploy
