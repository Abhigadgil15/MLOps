# Lab 6: CI/CD Pipeline for Machine Learning with GitHub Actions and GCP

Hey! In this lab, I'm walking you through how I set up a CI/CD pipeline for a machine learning project using GitHub Actions and Google Cloud Platform (GCP). The goal is to automate model training, versioning, and deployment in a reproducible way.

## ⚡ Overview

The workflow I created automates:

- Training a machine learning model.
- Saving the model to Google Cloud Storage (GCS).
- Building a Docker image with the trained model.
- Pushing the Docker image to Google Cloud Artifact Registry, tagged with both the current model version and `latest`.

## 🎯 Learning Objectives

By the end of this lab, I wanted to achieve:

- Automating the ML pipeline using GitHub Actions.
- Interacting with GCP services directly from CI/CD.
- Versioning models and keeping track of deployments.
- Creating a Docker image that packages the trained model for reproducibility.

## 🗂 Project Structure
```
Lab6/
├── src/
│   └── train_and_save_model.py  # Script for training & saving the model
├── trained_models/              # Local folder for saved models
├── .github/
│   └── workflows/
│       └── lab6_pipeline.yaml  # CI/CD workflow
├── Dockerfile                   # Dockerfile for building the model image
├── requirements.txt             # Python dependencies
└── .env                         # Environment variables (GCS_BUCKET_NAME, VERSION_FILE_NAME)
```

## 🔧 Prerequisites

Before running anything:

- GitHub account
- Google Cloud project with:
  - Cloud Storage
  - Artifact Registry
  - Cloud Build enabled
- Service account with these roles:
  - Storage Admin
  - Storage Object Admin
  - Artifact Registry Administrator
- Python 3.10 installed locally

**Disclaimer:** I will be switching my Google Cloud account to the free tier, so some features or quotas may differ if you follow along.

## 📝 Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Lab6
```

### 2. Configure GCP Credentials

- Download your service account JSON key.
- Add it as a GitHub Secret:
  - `GCP_SA_KEY` → JSON key content
  - `GCP_PROJECT_ID` → Your GCP project ID
  - `GCS_BUCKET_NAME` → Bucket to store models
  - `VERSION_FILE_NAME` → File to track model version (e.g., `model_version.txt`)

For local testing, set the environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### 3. Local Development

Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GCS_BUCKET_NAME=your-bucket-name
VERSION_FILE_NAME=model_version.txt
```

Run the training script locally:
```bash
python src/train_and_save_model.py
```

This will train the model, save it locally, upload it to GCS, and update the model version.

### 4. GitHub Actions Workflow

The workflow (`.github/workflows/lab6_pipeline.yaml`) handles:

- Checking out the code.
- Setting up Python and dependencies.
- Running tests (if any).
- Authenticating with GCP using the service account.
- Training and saving the model to GCS.
- Building a Docker image with the trained model.
- Pushing the Docker image to Artifact Registry with:
  - Version tag (v1, v2…)
  - `latest` tag

**Trigger Options:**

- Push to `main` branch
- Pull request to `main`
- Manual trigger via GitHub (`workflow_dispatch`)

### 5. Verify Deployment

- **GCS:** Check `trained_models` and `model_version.txt`.
- **Artifact Registry:** Confirm Docker image is available with both version and `latest` tags.

### 6. Git Commands for Lab6

To trigger the workflow:
```bash
git add .
git commit -m "Update Lab6 CI/CD pipeline"
git push origin main
```

Or create an empty commit if no code changes:
```bash
git commit --allow-empty -m "Trigger Lab6 workflow"
git push
```

## ⚠️ Notes

- Ensure `.gitignore` excludes unnecessary files like:
  - `.venv/`
  - `*.joblib`
  - `.env`
- The Dockerfile ensures the model can be deployed reproducibly.
- Model versions increment automatically on each run.

## 📚 References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Google Cloud Artifact Registry](https://cloud.google.com/artifact-registry/docs)
- [Python dotenv](https://pypi.org/project/python-dotenv/)
- [Docker Documentation](https://docs.docker.com/)