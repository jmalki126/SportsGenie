from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (so your frontend can call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/simulate")
def simulate(
    game_id: str = Query(...),
    season: int = Query(...),
    week: int = Query(...),
    model_version: str = Query("v1"),
    n_sims: int = Query(10000),
):
    # Temporary placeholder so the frontend stops failing with 422
    # Replace this with your real simulation logic later.
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "model_version": model_version,
        "n_sims": n_sims,
        "status": "simulate endpoint ok"
    }