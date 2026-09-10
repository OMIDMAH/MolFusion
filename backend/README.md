# MolFusion Backend (v2)

Python 3.11 FastAPI service hosting the real, RDKit-based feature-extraction
agent registry. Replaces the mocked feature generation in the legacy v1
prototype (see `main` branch / `v1-legacy` tag).

Status: Phase 0 scaffold only. No agents, API routes, or dependencies are
implemented yet.

## Environment

- Python 3.11 (this repo assumes the `research311` conda environment or an
  equivalent 3.11 interpreter)
- Dependency/env management: [uv](https://docs.astral.sh/uv/) (not yet
  installed on this machine as of Phase 0 — see project notes)

## Layout

```
backend/
├── pyproject.toml
├── .python-version
└── src/
    └── molfusion_backend/   # package root, empty until Phase 1
```
