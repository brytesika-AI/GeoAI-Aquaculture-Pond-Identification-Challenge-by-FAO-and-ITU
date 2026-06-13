# GeoAI Aquaculture Pond Identification Challenge Makefile

.PHONY: setup submit dossier optimize app clean

setup:
	python -m venv .venv
	.venv\Scripts\pip install -r requirements.txt
	cd app && npm install

submit:
	.venv\Scripts\python src/predict.py

dossier:
	.venv\Scripts\python src/explain.py
	Copy-Item -Path .\public\predictions.json -Destination .\app\public\predictions.json -Force

optimize:
	.venv\Scripts\python src/agents/crawler.py
	.venv\Scripts\python src/agents/extractor.py
	.venv\Scripts\python src/agents/optimizer.py

app:
	cd app && npm run dev

clean:
	Remove-Item -Recurse -Force .venv
	Remove-Item -Recurse -Force outputs
	Remove-Item -Recurse -Force app/.next
	Remove-Item -Recurse -Force app/node_modules
