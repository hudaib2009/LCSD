# CSD Frontend

Doctor-facing Next.js frontend for the Clinical Support Dashboard.

## Notes For The Public Repo

- `frontend/storage/` is runtime-only and is not included here.
- bundled sample medical images are not included in the public repo.
- the real inference API lives in `backend/app/`
- `frontend/server/` remains only as a compatibility shim

## Setup

From the repo root, start the backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python main.py
```

Then from `frontend/`:

```bash
npm install
npm run dev
```

Optional frontend env vars:

- `FASTAPI_BASE_URL`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `OPENROUTER_FALLBACK_MODEL`
- `OPENROUTER_FALLBACK_MODEL_2`
- `CSD_ALLOW_UNAUTHENTICATED`
- `CSD_BASIC_AUTH_USERNAME`
- `CSD_BASIC_AUTH_PASSWORD`

For any public or shared deployment, keep `CSD_ALLOW_UNAUTHENTICATED=0` and configure both basic-auth credentials so the case-management and file-serving routes are not publicly reachable.

## Runtime Data

- `data/cases.json` stores app metadata bundled with the frontend
- case uploads, explainability files, summaries, and reports are created at runtime and should stay out of git

## Useful Commands

- `npm run dev`
- `npm run build`
- `npm run start`
- `npm run lint`
