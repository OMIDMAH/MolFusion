from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from molfusion_backend.api.routes import router

app = FastAPI(title="MolFusion v2 Backend")

# Allow the local Vite dev server (any port) to call this API cross-origin.
# No cookies/credentials are used anywhere in this API, so this is safe for
# local development; tighten allow_origin_regex before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
