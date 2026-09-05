# MolFusion Frontend (v2)

Vite + React + TypeScript SPA for MolFusion v2. This is a thin client: all
chemistry (SMILES validation, fingerprint/descriptor computation) is done by
the FastAPI/RDKit backend. The frontend only calls the backend API and
renders/exports what it returns.

Requires Node.js >=20.19 or >=22.12 (Vite's minimum).

## Setup

```
npm install
npm run dev      # starts the dev server, default http://localhost:5173
```

The backend (from `../backend`) must be running separately, e.g.:

```
uvicorn molfusion_backend.main:app --host 127.0.0.1 --port 8000
```

## Configuring the backend URL

The frontend talks to the backend via a single configured base URL
(`src/config.ts`), defaulting to `http://127.0.0.1:8000` in development.

To override it, copy `.env.example` to `.env.local` and set:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

(`.env.local` is not committed; adjust the host/port for your setup.)

## Scripts

```
npm run dev        # dev server
npm run build       # tsc typecheck + production build (dist/)
npm run preview     # preview the production build
npm run test        # run Vitest once
npm run test:watch  # run Vitest in watch mode
npm run lint        # oxlint
```

## Structure

- `src/api/client.ts` — typed fetch wrapper for the backend API, with
  runtime validation of response shapes.
- `src/types/api.ts` — TypeScript types mirroring the backend's Pydantic
  schemas (`AgentMetadata`, `ValidateRequest`/`Response`, `ComputeRequest`,
  `FeatureVector`, `MoleculeResult`, ...).
- `src/lib/csv.ts` — CSV parsing for SMILES uploads and CSV export of
  `/features/compute` results (no chemistry logic).
- `src/components/` — `MoleculeInputPanel`, `MoleculeQueue`,
  `AgentSelector`, `ResultsPanel`, `ExportButton`.
- `src/App.tsx` — orchestrates the molecule queue, agent selection, and
  compute/export flow.
