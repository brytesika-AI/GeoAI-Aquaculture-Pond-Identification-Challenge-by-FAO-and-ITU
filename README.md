# GeoAI Aquaculture Pond Identification Challenge Solution

A complete, reproducible repository containing a leaderboard-winning machine learning pipeline, an FAO/ITU-aligned trustworthiness dossier, an interactive Next.js companion map application using real Google Maps, and a multi-agent optimization system.

---

## 1. Project Directory Structure

```
├── .venv/                         # Local Python Virtual Environment
├── app/                           # Next.js Companion Map Application
│   ├── public/                    # Static assets (predictions.json)
│   ├── src/app/                   # React App Router pages & styles
│   └── package.json               # Frontend dependencies
├── data/                          # Gitignored raw datasets
│   ├── Train.csv                  # Tabular monthly features (training)
│   ├── Test.csv                   # Tabular monthly features (testing)
│   ├── SampleSubmission.csv       # Submission template
│   └── Trustworthiness_Evaluation.pdf # Dossier template
├── docs/                          # Explainability figures and dossier
│   ├── figures/                   # SHAP & PDP exported plots
│   └── TRUSTWORTHINESS.md         # FAO/ITU trustworthiness dossier
├── outputs/                       # Saved OOF predictions and reports
├── src/                           # Machine Learning source code
│   ├── agents/                    # Multi-Agent Optimization System
│   │   ├── crawler.py             # Retrieves Zindi tips
│   │   ├── extractor.py           # Parses crawled recommendations
│   │   └── optimizer.py           # Evaluates parameter configurations
│   ├── features.py                # Spectral indices & temporal aggregates
│   ├── train.py                   # Spatial validation & ensemble training
│   ├── predict.py                 # Pipeline execution & submission generator
│   └── generate_mock_data.py      # Script to create mock validation data
├── Makefile                       # Automation shortcut tasks
├── requirements.txt               # Locked Python dependencies
└── README.md                      # Setup and usage guide (this file)
```

---

## 2. Setup & Installation

### Prerequisite: Python & Node.js
- Ensure Python 3.10+ and Node.js 18+ are installed on your machine.

### Step 1: Initialize Virtual Environment & Dependencies
Run the setup commands to install Python packages and Next.js libraries:
```bash
# Initialize python venv & install dependencies
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Install Next.js frontend dependencies
cd app
npm install
cd ..
```

---

## 3. How to Reproduce the Submission

### Option A: End-to-End Run with Mock Data (Out-of-the-box Test)
If you do not have the raw Zindi datasets yet, the repository includes a mock data generator so you can verify the execution immediately:
```bash
# 1. Generate realistic mock satellite datasets
python src/generate_mock_data.py

# 2. Run the feature extraction and train LightGBM + XGBoost + CatBoost
.venv\Scripts\python src/predict.py

# 3. Generate SHAP plots, PDP graphs, and predictions.json
.venv\Scripts\python src/explain.py
```

### Option B: Run with Real Zindi Data
1. Download the datasets from the [Zindi Competition Page](https://zindi.africa/competitions/geoai-aquaculture-pond-identification-challenge/data).
2. Place `Train.csv`, `Test.csv`, `SampleSubmission.csv`, and `Trustworthiness_Evaluation.pdf` inside the `./data/` folder.
3. Run the submission target:
   ```bash
   make submit
   make dossier
   ```
This will output `submission.csv` at the root of the workspace, aligned exactly with the row order and format of `SampleSubmission.csv`.

---

## 4. Cross-Validation & Leaderboard Relationship

### Spatial Cross-Validation
Because the private test set evaluates models on geographical regions separate from the training set, raw coordinates (lat/lon) can lead to extreme spatial overfitting. 
- We implement **KMeans geographical clustering** to partition coordinates into 5 folds. We hold out entire spatial groups during validation (**GroupKFold**).
- Your **Spatial-CV OOF Score** (printed at the end of training) is the only metric you should trust. A high Stratified-CV score is misleading if the model overfits to coordinates.

### The Zindi Sorting Data Leakage Warning
> [!WARNING]
> Zindi forum threads indicate that the original `Train.csv` and `Test.csv` files were sorted by target label before being split. This means the row index itself leaks the target variable. 
- While this leak allows achieving a perfect 1.0 score on the public leaderboard, it **will not generalize** to the private test set or real-world deployments.
- To build a trustworthy model, **never** include row indices or sorted indices as features, and evaluate performance strictly using spatial group splits.

---

## 5. Running the Multi-Agent Optimization System

The multi-agent system looks up Zindi discussion trends, extracts parameter recommendations, and automatically evaluates them on the cross-validation pipeline:
```bash
# Run the Multi-Agent optimization loop
make optimize
```
- `crawler.py` scans Zindi boards and saves raw textual tips.
- `extractor.py` parses recommendations into structured JSON configurations.
- `optimizer.py` runs experiments (e.g., toggling coordinates, tuning XGBoost depth), evaluates OOF AUC scores, and outputs a final `optimization_report.json` detailing the best parameter settings.

---

## 6. Launching and Deploying the Map App

The companion map application reads directly from the static file `public/predictions.json` generated by the explainability script.

### Step 1: Running Locally (No API Key Required)
Since the app uses Leaflet with public tile servers, it runs out of the box without any API Key configuration:
1. Start the Next.js development server:
   ```bash
   make app
   # App runs at http://localhost:3000
   ```

### Step 2: Deploy to Vercel
Deploy your application directly to Vercel:
```bash
cd app
npx vercel --prod
```
The application will build, compile TypeScript, and be hosted live automatically on Vercel. 
No environment variables need to be set in Vercel settings for mapping to work.

---

## 7. LlamaIndex Agentic Extrapolator & Google Earth Engine

We have integrated a LlamaIndex-based agentic framework that allows you to fetch satellite data for any coordinate in Africa (e.g. Uganda, Kenya, Nigeria) and run the trained model ensemble to extrapolate predictions.

- **Single Coordinate Agent Run**:
  ```bash
  .venv\Scripts\python -m src.agents.agentic_extrapolator
  ```
- **Bulk Coordinates Extrapolation Run**:
  ```bash
  .venv\Scripts\python -m src.bulk_extrapolator
  ```
  This script reads coordinates from a CSV file (`./data/africa_coordinates.csv`), queries the Google Earth Engine API (or the high-fidelity simulator fallback), processes the raw monthly bands, runs predictions, and saves the final predictions to `./outputs/africa_predictions.csv`.
- **How it works**: It uses LlamaIndex `FunctionTool` definitions to fetch monthly Sentinel-1 and Sentinel-2 composites from Google Earth Engine, compute spectral indices (NDVI, NDWI, AWEI), perform ensemble model inference, and output a detailed responsible AI safety/trustworthiness audit.
- **Detailed Documentation**: Read the full architecture, GEE setup, and case studies (like Lake Victoria and Kajjansi Station) in [EXTRAPOLATION.md](file:///c:/Users/bright.sikazwe/Downloads/Zambia%20Elections%20Tracker/GeoAI-Aquaculture-Pond-Identification-Challenge-by-FAO-and-ITU/docs/EXTRAPOLATION.md).

---

## 8. Deployment Links & Deliverables

- **Live Companion Map Web App**: [https://app-six-nu-23.vercel.app](https://app-six-nu-23.vercel.app)
- **GitHub Repository**: [https://github.com/brytesika-AI/GeoAI-Aquaculture-Pond-Identification-Challenge-by-FAO-and-ITU](https://github.com/brytesika-AI/GeoAI-Aquaculture-Pond-Identification-Challenge-by-FAO-and-ITU)


