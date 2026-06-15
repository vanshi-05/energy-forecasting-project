# Full-Stack Deployment and Setup Guide

This document details the step-by-step instructions for running the energy forecasting project locally, building the Docker container, and deploying it to the cloud for showcase/portfolio hosting.

---

## 💻 Local Environment Setup

### 1. Requirements
Ensure you have the following installed:
- Git
- Python 3.12 (versions 3.8 to 3.12 are fully compatible)
- Docker (optional, for container tests)

### 2. Manual Commands
If you prefer not to use `run.py`, you can run the scripts individually:

```bash
# Install packages
pip install -r requirements.txt

# Step 1: Ingest/Generate Raw Data
python src/data/generate_synthetic_data.py

# Step 2: Data Preprocessing
python src/data/preprocess_residential.py
python src/data/preprocess_commercial.py
python src/data/preprocess_industrial.py

# Step 3: Add tabular & cyclical features
python src/data/feature_engineering.py

# Step 4: Model Training (Fits Seasonal baseline, trains LSTM, XGBoost, and PyTorch Hybrid)
python -m src.models.train_models

# Step 5: Model Evaluation (Outputs results/metrics and visual charts)
python -m src.models.evaluate_model

# Step 6: Start FastAPI Server
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🐳 Docker Setup

To verify the container locally:

1. **Build the container**:
   ```bash
   docker build -t energy-forecasting .
   ```
2. **Run it locally**:
   ```bash
   docker run -d -p 8000:8000 --name energy-app energy-forecasting
   ```
3. **Verify the server is running**:
   Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## ☁️ Cloud Deployment

Since this is a full-stack project with a FastAPI backend, you can easily host it online so hiring managers can view it live.

### Option A: Deploying to Render (Free Tier - Recommended)
Render allows you to deploy Docker containers directly from GitHub:

1. Push your local repository to your GitHub account (accomplished in git sync steps).
2. Go to [Render](https://render.com/) and sign in.
3. Click **New +** and select **Web Service**.
4. Connect your GitHub repository.
5. In the Service settings:
   - **Environment**: Select `Docker`.
   - **Branch**: Select `main`.
   - **Region**: Choose the closest region to your users.
   - **Plan**: Select the Free plan.
6. Click **Deploy Web Service**. Render will build the Dockerfile and launch the dashboard.

### Option B: Deploying to Heroku
If you have a Heroku account, you can deploy using the Heroku CLI:

```bash
# Login to Heroku Container Registry
heroku container:login

# Create a new Heroku app
heroku create energy-forecasting-dashboard

# Build and push the Docker image
heroku container:push web --app energy-forecasting-dashboard

# Release the image to start the service
heroku container:release web --app energy-forecasting-dashboard

# Open the live app
heroku open --app energy-forecasting-dashboard
```

### Option C: Deploying to AWS Elastic Beanstalk
1. Zip the files including `Dockerfile`, `requirements.txt`, and the directories (`src/`, `models/`, `data/`).
2. Go to the AWS Elastic Beanstalk console.
3. Create a new application and environment.
4. Select **Docker** as the Platform.
5. Upload the zip file and create the environment.
