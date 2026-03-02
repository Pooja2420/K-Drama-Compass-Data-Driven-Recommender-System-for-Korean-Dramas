# K-Drama Compass: Data-Driven Recommender System

A production-grade recommender system for Korean dramas using sentiment analysis, collaborative filtering, and NLP.

## Project Structure

```
KDrama/
├── src/                    # Source code
│   ├── data/               # Data ingestion & ETL
│   ├── features/           # Feature engineering
│   ├── models/             # ML models
│   ├── api/                # FastAPI backend
│   └── utils/              # Shared utilities
├── data/
│   ├── raw/                # Raw datasets (not committed)
│   ├── processed/          # Cleaned datasets (not committed)
│   └── external/           # External data sources
├── notebooks/              # Jupyter notebooks (EDA, experiments)
├── models/
│   ├── artifacts/          # Trained model files
│   └── registry/           # Model versioning
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── frontend/               # UI (Streamlit / React)
├── docker/                 # Docker configs
├── logs/                   # Application logs
└── .github/workflows/      # CI/CD pipelines
```

## Setup

```bash
# Clone repo
git clone https://github.com/Pooja2420/K-Drama-Compass-Data-Driven-Recommender-System-for-Korean-Dramas.git
cd KDrama

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Copy env file
cp .env.example .env
```

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Setup & Repo Structure | ✅ Complete |
| 2 | Data Pipeline | 🔜 Pending |
| 3 | EDA & Feature Engineering | 🔜 Pending |
| 4 | Sentiment Analysis & NLP | 🔜 Pending |
| 5 | Recommendation Engine | 🔜 Pending |
| 6 | MLOps & Model Management | 🔜 Pending |
| 7 | Backend API | 🔜 Pending |
| 8 | Frontend / UI | 🔜 Pending |
| 9 | Containerization & CI/CD | 🔜 Pending |
| 10 | Cloud Deployment | 🔜 Pending |
| 11 | Monitoring & Observability | 🔜 Pending |
| 12 | Testing & Security | 🔜 Pending |
| 13 | Documentation | 🔜 Pending |
