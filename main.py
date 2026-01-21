# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import requests
from dotenv import load_dotenv
from typing import Optional, Dict, Any

load_dotenv()

app = FastAPI()

# Allow Lovable (and browsers) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# SportRadar config
# ------------------------------------------------------------
SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY", "").strip()
SPORTRADAR_ACCESS_LEVEL = os.getenv("SPORTRADAR_ACCESS_LEVEL", "trial").strip()
SPORTRADAR_LANGUAGE = os.getenv("SPORTRADAR_LANGUAGE", "en").strip()

# SportRadar NFL v7 base
# Example: https://api.sportradar.com/nfl/official/trial/v7/en
SR_BASE = f"https://api.sportradar.com/nfl/official/{SPORTRADAR_ACCESS_LEVEL}/v7/{SPORTRADAR_LANGUAGE}"


def sr_get(path: str, params: Optional[Dict[str, Any]] = None):
    """
    Small helper that calls SportRadar and returns:
      (json_data, None) on success
      (None, JSONResponse) on error
    """
    if not SPORTRADAR_API_KEY:
        return None, JSONResponse(status_code=400, content={"error": "SportRadar API key missing"})

    p: Dict[str, Any] = params.copy() if params else {}
    p["api_key"] = SPORTRADAR_API_KEY

    url = f"{SR_BASE}{path}"
    try:
        r = requests.get(url, params=p, timeout=20)
        if r.status_code != 200:
            return None, JSONResponse(
                status_code=r.status_code,
                content={
                    "error": "SportRadar request failed",
                    "status": r.status_code,
                    "url": url,
                    "body": r.text[:500],
                },
            )
        return r.json(), None
    except Exception as e:
        return None, JSONResponse(
            status_code=500,
            content={"error": "SportRadar request exception", "message": str(e)},
        )


def safe_team_abbr(game: Dict[str, Any], side: str) -> str:
    """
    SportRadar schedule games usually include:
      game["home"]["abbr"] or ["alias"]
      game["away"]["abbr"] or ["alias"]
    We fall back safely.
    """
    t = game.get(side) or {}
    return t.get("abbr") or t.get("alias") or t.get("name") or ""


def map_schedule_game_to_frontend(game: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize SportRadar schedule game into what your frontend expects.
    """
    scheduled = game.get("scheduled") or ""
    return {
        "game_id": game.get("id") or game.get("game_id") or "",
        "away_team": safe_team_abbr(game, "away"),
        "home_team": safe_team_abbr(game, "home"),
        "kickoff_utc": scheduled,
        "kickoff_display": scheduled,  # keep simple for now
        "confidence": 0.50,           # placeholder until your model fills this
        "source": "sportradar",
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/weeks")
def weeks(season: int = 2024):
    # Keep simple for now (frontend just needs something)
    return [
        {"week": 1, "season_type": "REG", "label": "Week 1", "game_count": 16},
        {"week": 2, "season_type": "REG", "label": "Week 2", "game_count": 16},
        {"week": 3, "season_type": "REG", "label": "Week 3", "game_count": 16},
    ]


@app.get("/games")
def games(season: int = 2024, week: int = 1, season_type: str = "REG"):
    """
    Returns schedule for a week.

    Example:
      /games?season=2024&week=1&season_type=REG
    """
    path = f"/games/{season}/{season_type}/{week}/schedule.json"
    data, err = sr_get(path)
    if err:
        return err

    week_obj = data.get("week") or {}
    raw_games = week_obj.get("games") or []

    games_list = [map_schedule_game_to_frontend(g) for g in raw_games]

    return {"season": season, "week": week, "season_type": season_type, "games": games_list}


@app.get("/game")
def game(
    game_id: str,
    season: int = 2024,
    week: int = 1,
    season_type: str = "REG",
    home_team_key: str = "",
    away_team_key: str = "",
    as_of_date: str = "",
):
    """
    Looks up a single game by ID by searching within the weekly schedule.
    """
    path = f"/games/{season}/{season_type}/{week}/schedule.json"
    data, err = sr_get(path)
    if err:
        return err

    week_obj = data.get("week") or {}
    raw_games = week_obj.get("games") or []

    for g in raw_games:
        if g.get("id") == game_id:
            mapped = map_schedule_game_to_frontend(g)

            venue = g.get("venue") or {}
            mapped["venue_name"] = venue.get("name") or ""
            mapped["venue_city"] = venue.get("city") or ""
            mapped["venue_state"] = venue.get("state") or ""

            mapped["status"] = g.get("status") or ""
            mapped["scheduled"] = g.get("scheduled") or ""

            return mapped

    return JSONResponse(
        status_code=404,
        content={"detail": "Game not found in that week. Check season, week, season_type, and game_id."},
    )


@app.get("/simulate")
def simulate(
    game_id: str,
    season: int = 2024,
    week: int = 1,
    model_version: str = "v1",
    n_sims: int = 10000,
):
    # Placeholder until your model is wired
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "model_version": model_version,
        "n_sims": n_sims,
        "home_win_prob": 0.57,
        "proj_spread_home": -2.5,
        "proj_total": 46.0,
        "source": "placeholder",
    }


@app.get("/explain")
def explain(
    game_id: str,
    season: int = 2024,
    week: int = 1,
    model_version: str = "v1",
):
    # Placeholder until your model explanations are wired
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "model_version": model_version,
        "drivers": [
            {"name": "Team strength gap", "impact_points": 2.1, "note": "Placeholder explanation."},
            {"name": "Home field", "impact_points": 1.5, "note": "Placeholder explanation."},
        ],
        "source": "placeholder",
    }