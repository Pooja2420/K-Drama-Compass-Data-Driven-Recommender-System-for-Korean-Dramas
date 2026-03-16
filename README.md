# K-Drama Compass: Data-Driven Recommender System

A production-grade recommender system for Korean dramas that combines content-based filtering, collaborative filtering, and NLP-based sentiment analysis to deliver personalised drama recommendations through a secured REST API.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Methodology](#methodology)
   - [Content-Based Filtering](#content-based-filtering)
   - [Collaborative Filtering (SVD)](#collaborative-filtering-svd)
   - [Hybrid Recommender](#hybrid-recommender)
   - [Sentiment Analysis](#sentiment-analysis)
6. [System Architecture](#system-architecture)
7. [API Reference](#api-reference)
8. [Evaluation & Outcomes](#evaluation--outcomes)
9. [Quickstart](#quickstart)
10. [Docker](#docker)
11. [Running Tests](#running-tests)
12. [Project Structure](#project-structure)
13. [Tech Stack](#tech-stack)
14. [CI/CD](#cicd)

---

## Introduction

K-Drama Compass answers a deceptively simple question: *"I loved Crash Landing on You — what should I watch next?"*

Rather than relying on a single signal, the system fuses three complementary sources of information:

- **What a drama is about** — TF-IDF on synopses, genre/network encoding, and cast reputation (content-based)
- **What audiences actually watched together** — latent factor patterns extracted from hundreds of thousands of user reviews (collaborative filtering)
- **What viewers felt** — polarity and subjectivity scores extracted from free-text reviews using TextBlob and optionally a fine-tuned BERT model

The result is a blended, explainable recommendation score surfaced through a FastAPI backend with JWT authentication, rate limiting, and an MLflow experiment tracker.

---

## Dataset

Three raw CSV files (sourced from MyDramaList / Kaggle):

| File | Rows (approx.) | Key Columns |
|---|---|---|
| `korean_drama.csv` | ~2,000 dramas | `kdrama_id`, `drama_name`, `synopsis`, `genre`, `org_net`, `content_rt`, `rank`, `pop`, `tot_eps`, `duration`, `aired_on`, `start_dt`, `end_dt` |
| `reviews.csv` | ~100,000 reviews | `user_id`, `title`, `review_text`, `overall_score`, `story_score`, `acting_cast_score`, `music_score`, `rewatch_value_score`, `ep_watched` |
| `wiki_actors.csv` | ~10,000 cast entries | `drama_name`, `actor_name`, `role` (Main / Support) |

> Raw data is not committed to the repository. Place the files under `data/raw/` before running the pipeline.

---

## Data Preprocessing

The ETL pipeline lives in `src/data/etl.py` and is triggered by `scripts/run_pipeline.py`.

### Dramas

| Step | Detail |
|---|---|
| Missing text fields | `director`, `screenwriter`, `synopsis` filled with `"Unknown"` / `"No synopsis available"` |
| Categorical imputation | `aired_on`, `org_net` filled with column mode |
| Duration imputation | Median fill for missing episode length (seconds) |
| Date parsing | `start_dt` / `end_dt` parsed with `pd.to_datetime`; missing dates replaced with median |
| Temporal features | `start_month`, `start_day_of_week`, `duration_days` derived from parsed dates |
| Duration category | Binned into `short` (<30 min), `medium` (30–60 min), `long` (>60 min) |

### Reviews

| Step | Detail |
|---|---|
| Missing text | `review_text` filled with placeholder string |
| Episode parsing | `ep_watched` string (e.g. `"16 of 16"`) split into `episodes_watched` and `total_episodes` |
| Title normalisation | Lowercased and stripped for consistent joining |
| Score-based label | `overall_score ≥ 7` → Positive (2), `4–6` → Neutral (1), `< 4` → Negative (0) |

### Actors

| Step | Detail |
|---|---|
| Role casing | Standardised to title case (`Main Role`, `Support Role`) |
| Name normalisation | Drama names lowercased for merge key |

All cleaned DataFrames are written to `data/processed/` as CSV files, plus a merged `actors + reviews` table used for actor reputation features.

---

## Feature Engineering

`src/features/feature_engineering.py` assembles a unified feature matrix (the **feature store**) indexed by `kdrama_id`.

### Feature Blocks

| Block | Method | Dimensionality |
|---|---|---|
| **TF-IDF synopsis** | `TfidfVectorizer(max_features=300, ngram_range=(1,2), min_df=2)` on drama synopses | 300 |
| **Categorical encoding** | One-hot encoding of `org_net`, `content_rt`, `aired_on` | variable |
| **Numeric features** | `MinMaxScaler` applied to `year`, `tot_eps`, `duration`, `rank`, `pop`, `start_month`, `start_day_of_week`, `duration_days` | 8 |
| **Review aggregates** | Per-drama mean scores (`overall`, `story`, `acting`, `music`, `rewatch`), review count, sentiment distribution | 9 |
| **Actor reputation** | Per-drama cast avg score, cast size, main/support role counts | 4 |

All blocks are joined on `kdrama_id` and `NaN` values are zero-filled. The final feature store is saved to `data/processed/feature_store.csv`.

---

## Methodology

### Content-Based Filtering

`src/models/recommender/content_based.py`

Given the unified feature matrix **F** (shape `n_dramas × n_features`), the similarity between drama *i* and drama *j* is:

```
similarity(i, j) = cos(F_i, F_j)
                 = (F_i · F_j) / (||F_i|| × ||F_j||)
```

The full `n × n` cosine similarity matrix is pre-computed at fit time and stored in memory. At inference time, the row corresponding to the query drama is retrieved, sorted descending, and the top-N results are returned.

**Strengths:** Works for cold-start scenarios (no interaction history needed); captures synopsis semantics and structural drama attributes.

### Collaborative Filtering (SVD)

`src/models/recommender/collaborative.py`

A user–item rating matrix **R** (shape `n_users × n_dramas`) is built from `overall_score` values in the reviews dataset using a sparse CSR matrix. The matrix is decomposed with **Truncated SVD**:

```
R ≈ U × Σ × V^T
```

where:
- **U** (`n_users × k`) — user latent factors
- **V^T** (`k × n_dramas`) — item latent factors
- **k** = 50 components (default)

Item-item similarity for the recommendation step is computed as the dot product of item latent vectors:

```
cf_score(i, j) = V_i · V_j
```

A 90/10 train-test split is used, and RMSE on the held-out set is logged at fit time. Explained variance ratio is also logged.

**Strengths:** Captures collaborative patterns ("viewers who liked A also liked B") without requiring feature engineering; handles sparse data via matrix factorisation.

### Hybrid Recommender

`src/models/recommender/hybrid.py`

Combines both models with a configurable weight `α ∈ [0, 1]`:

```
final_score = α × norm(cb_score) + (1 − α) × norm(cf_score)
```

Both score vectors are independently min-max normalised to `[0, 1]` before blending:

```
norm(x) = (x − min(x)) / (max(x) − min(x))
```

The algorithm:
1. Fetch top-`candidate_pool` (default: 50) results from both CB and CF.
2. Normalise each model's scores to `[0, 1]`.
3. Outer-join on `drama_name` (dramas only in one model's list get a 0 score for the other).
4. Compute the weighted blend.
5. Return the top-N by `hybrid_score`.

| α value | Behaviour |
|---|---|
| `1.0` | Pure content-based |
| `0.0` | Pure collaborative |
| `0.5` | Equal blend (default) |

### Sentiment Analysis

`src/models/sentiment/textblob_model.py` and `src/models/sentiment/bert_model.py`

**TextBlob baseline** — rule-based lexicon approach applied per review:

| Output | Range | Meaning |
|---|---|---|
| `polarity` | `[−1, 1]` | Negative → Positive |
| `subjectivity` | `[0, 1]` | Objective → Subjective |
| `tb_label` | `{0, 1, 2}` | Negative / Neutral / Positive |

Label thresholds:
- `polarity > 0.1` → **Positive** (2)
- `polarity < −0.1` → **Negative** (0)
- otherwise → **Neutral** (1)

Scores are aggregated per drama as: mean polarity, mean subjectivity, percentage positive/neutral/negative. The resulting sentiment features are incorporated into the feature store as additional signals for content-based similarity.

**BERT model** — a transformer-based classifier (`src/models/sentiment/bert_model.py`) is available as a higher-accuracy alternative for batch inference on review text.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                              │
│           (Browser / curl / Streamlit UI)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS
┌──────────────────────────▼──────────────────────────────────┐
│                     FastAPI (uvicorn)                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Middleware: CORS  │  Rate Limit  │  Process-Time    │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  POST /auth/token          →  JWT issuance           │   │
│  │  GET  /health              →  model readiness        │   │
│  │  GET  /recommend           →  CB / CF / Hybrid recs  │   │
│  │  GET  /search              →  drama catalogue search │   │
│  │  GET  /sentiment/{name}    →  per-drama sentiment    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
┌──────────────▼──────────┐  ┌────────────▼────────────────┐
│   Model Layer           │  │   Data Layer                │
│                         │  │                             │
│  ContentBasedRecommender│  │  data/processed/            │
│  CollaborativeRecommender  │   ├── cleaned_dramas.csv    │
│  HybridRecommender      │  │   ├── cleaned_reviews.csv   │
│  TextBlob / BERT        │  │   ├── feature_store.csv     │
│                         │  │   └── sentiment_results.csv │
│  models/artifacts/      │  │                             │
│   ├── content_based/    │  │  PostgreSQL (Docker)        │
│   ├── collaborative/    │  │                             │
│   └── hybrid/           │  └─────────────────────────────┘
└──────────────┬──────────┘
               │
┌──────────────▼──────────┐
│   MLflow Tracking        │
│   (experiment registry,  │
│    metric logging,       │
│    model versioning)     │
└─────────────────────────┘
```

**Key design decisions:**

- `create_app()` factory pattern — enables test isolation; each test gets a fresh app instance.
- `lru_cache` on model loaders — models are loaded from disk once and cached in memory.
- `pydantic-settings` `BaseSettings` — all configuration (secret key, CORS origins, rate limits, paths) is driven by environment variables with sane defaults.
- JWT authentication — `python-jose` + `passlib[bcrypt]` with configurable token expiry.
- Per-request `X-Process-Time-Ms` header — surfaced on all responses for latency observability.

---

## API Reference

All endpoints that return data require a valid Bearer token unless noted.

### `POST /auth/token`

Obtain a JWT access token.

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=demo&password=kdrama123"
```

```json
{ "access_token": "<jwt>", "token_type": "bearer" }
```

### `GET /health`

No authentication required.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": {
    "content_based": true,
    "collaborative": true,
    "hybrid": true
  }
}
```

### `GET /recommend`

```
GET /recommend?drama_name=Crash+Landing+on+You&model=hybrid&top_n=10
Authorization: Bearer <token>
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `drama_name` | string | required | Source drama title |
| `model` | string | `hybrid` | `content_based`, `collaborative`, or `hybrid` |
| `top_n` | int | `10` | Number of results (1–50) |

```json
{
  "query": "Crash Landing on You",
  "model": "hybrid",
  "top_n": 10,
  "recommendations": [
    { "drama_name": "Goblin", "score": 0.912 },
    { "drama_name": "My Love from the Star", "score": 0.887 }
  ]
}
```

### `GET /search`

```
GET /search?q=goblin
```

Returns dramas whose name, genre, or synopsis matches the query string.

### `GET /sentiment/{drama_name}`

```
GET /sentiment/Goblin
```

```json
{
  "drama_name": "Goblin",
  "polarity": 0.34,
  "subjectivity": 0.52,
  "label": "positive",
  "review_count": 1482
}
```

---

## Evaluation & Outcomes

The recommender is evaluated using the following metrics, computed over a random sample of 50 query dramas:

| Metric | Description |
|---|---|
| **RMSE** | Rating prediction error for collaborative filtering (lower = better) |
| **Precision@K** | Fraction of top-K recommendations that appear in the user's co-watch history |
| **Recall@K** | Fraction of relevant items retrieved in top-K |
| **NDCG@K** | Ranking quality — penalises relevant items appearing lower in the list |
| **MAP@K** | Mean Average Precision — summary of precision across recall levels |
| **Catalog coverage** | Fraction of the full drama catalogue appearing in at least one recommendation list |
| **Intra-list diversity** | Average pairwise cosine distance within a recommendation list |

All metrics are logged to **MLflow** during training, enabling comparison across model variants, α values, and feature configurations.

---

## Quickstart

### Prerequisites

- Python 3.9+ and `uv` (or `pip`)
- PostgreSQL (optional — only needed for user data persistence)
- Docker (optional — for containerised deployment)

### Install

```bash
# Clone
git clone https://github.com/Pooja2420/K-Drama-Compass-Data-Driven-Recommender-System-for-Korean-Dramas.git
cd KDrama

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# Install all dependencies
pip install -e ".[dev]"

# Copy and edit environment file
cp .env.example .env

# Install pre-commit hooks
pre-commit install
```

### Train Models

Place raw CSVs in `data/raw/`, then run:

```bash
python scripts/run_pipeline.py
```

This executes: ingest → ETL → feature engineering → sentiment analysis → fit all three recommenders → log to MLflow → save artifacts.

### Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Using the Makefile

```bash
make install       # install production deps
make install-dev   # install with dev extras
make lint          # run Ruff linter
make format        # run Ruff formatter
make test          # run full test suite
make serve         # uvicorn --reload
make train         # run full training pipeline
```

---

## Docker

```bash
# Build and start API + MLflow + PostgreSQL
docker compose -f docker/docker-compose.yml up --build

# Stop
docker compose -f docker/docker-compose.yml down
```

Services:

| Service | Port | Description |
|---|---|---|
| `api` | `8000` | FastAPI application |
| `mlflow` | `5001` | MLflow tracking UI |
| `postgres` | `5432` | PostgreSQL 16 |

The Dockerfile uses a multi-stage build (builder → runtime), runs as a non-root user, and includes a `HEALTHCHECK` against `/health`.

---

## Running Tests

```bash
# Full suite with coverage
pytest

# Unit tests only
pytest tests/unit/

# A single file
pytest tests/unit/test_recommender.py -v
```

**Test coverage areas:**

| Module | Tests |
|---|---|
| `ContentBasedRecommender` | fit, recommend, top-N, unknown drama, score range, save/load roundtrip |
| `precision_at_k`, `catalog_coverage` | perfect, zero, partial, edge-case k |
| API `/health` | status, version, models_loaded dict, process-time header |
| API `/auth/token` | valid login, invalid credentials, missing credentials |
| API `/recommend` | auth guard, bad model, no model (503), top_n validation, mocked model |
| API `/search` | no data (503), missing query (422), mocked data |
| API `/sentiment` | no data (503), mocked data |
| Schemas | `HealthResponse`, `Token`, `DramaOut`, `RecommendResponse` defaults |
| Settings | loads, app_name, port, rate_limit, model weights, Path types, env override |
| Auth utils | create/decode token, invalid token, password verification, demo user |

---

## Project Structure

```
KDrama/
├── src/
│   ├── data/
│   │   ├── ingest.py              # Load raw CSVs
│   │   ├── etl.py                 # Clean & transform
│   │   └── validate.py            # Schema validation
│   ├── features/
│   │   ├── feature_engineering.py # Unified feature store
│   │   └── eda.py                 # Exploratory analysis helpers
│   ├── models/
│   │   ├── recommender/
│   │   │   ├── content_based.py   # Cosine similarity recommender
│   │   │   ├── collaborative.py   # SVD matrix factorisation
│   │   │   └── hybrid.py          # Weighted blend
│   │   ├── sentiment/
│   │   │   ├── textblob_model.py  # Rule-based sentiment
│   │   │   ├── bert_model.py      # Transformer-based sentiment
│   │   │   └── preprocessor.py    # Text cleaning
│   │   ├── evaluate.py            # RMSE, Precision@K, NDCG, MAP, coverage
│   │   ├── mlflow_tracker.py      # MLflow logging helpers
│   │   └── registry.py            # Model versioning
│   ├── api/
│   │   ├── main.py                # create_app() factory, middleware
│   │   ├── auth.py                # JWT creation & verification
│   │   ├── dependencies.py        # Shared FastAPI dependencies
│   │   ├── schemas.py             # Pydantic request/response schemas
│   │   ├── middleware/
│   │   │   └── rate_limit.py      # Token-bucket rate limiter
│   │   └── routers/
│   │       ├── auth.py            # POST /auth/token
│   │       ├── health.py          # GET /health
│   │       ├── recommend.py       # GET /recommend
│   │       ├── search.py          # GET /search
│   │       └── sentiment.py       # GET /sentiment/{name}
│   └── utils/
│       ├── config.py              # pydantic-settings Settings
│       └── logger.py              # loguru logger factory
├── data/
│   ├── raw/                       # Source CSVs (not committed)
│   └── processed/                 # Cleaned CSVs + feature store
├── models/
│   └── artifacts/                 # Serialised .joblib model files
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── unit/
│   │   ├── test_api.py
│   │   ├── test_recommender.py
│   │   └── test_data.py
│   └── integration/
├── scripts/
│   └── run_pipeline.py            # End-to-end training script
├── notebooks/                     # EDA & experiments
├── docker/
│   ├── Dockerfile                 # Multi-stage build
│   ├── docker-compose.yml         # API + MLflow + Postgres
│   └── init-multiple-dbs.sh       # DB initialisation script
├── .github/workflows/ci.yml       # Lint → test → Docker CI
├── .pre-commit-config.yaml        # Ruff lint + format hooks
├── pyproject.toml                 # hatchling build, all deps, Ruff + pytest config
├── Makefile
└── .env.example
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Build | hatchling |
| Linting / Formatting | Ruff |
| API framework | FastAPI + uvicorn |
| Data | pandas, NumPy, SciPy |
| ML / NLP | scikit-learn, TextBlob, Transformers (BERT), NLTK |
| Auth | python-jose, passlib[bcrypt] |
| Config | pydantic-settings |
| MLOps | MLflow |
| Database | PostgreSQL + SQLAlchemy |
| Containerisation | Docker (multi-stage), Docker Compose |
| Testing | pytest, pytest-asyncio, pytest-cov |
| CI/CD | GitHub Actions |
| Package management | uv (recommended) / pip |

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request:

1. **Lint** — Ruff check on `src/` and `tests/`
2. **Test** — Full pytest suite on Python 3.10 and 3.11 with coverage upload to Codecov
3. **Docker** — Build image to verify the Dockerfile is valid

---

## Contributing

1. Fork the repo and create a feature branch.
2. Run `make install-dev` and `pre-commit install`.
3. Make changes, add tests, run `make test`.
4. Open a pull request — CI runs automatically.

---

## License

MIT License — see `LICENSE` for details.

*Built by Pooja Mohan*
