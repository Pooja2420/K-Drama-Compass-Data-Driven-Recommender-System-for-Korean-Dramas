# Branching Strategy

## Branches
- `main`     — production-ready code only. Protected branch.
- `dev`      — integration branch for completed features.
- `feature/*` — one branch per feature/phase (e.g. `feature/data-pipeline`)
- `fix/*`    — bug fixes (e.g. `fix/sentiment-scoring`)

## Workflow
1. Branch off `dev` for each new feature: `git checkout -b feature/my-feature dev`
2. Open a Pull Request from `feature/*` → `dev`
3. After testing, merge `dev` → `main` for releases
