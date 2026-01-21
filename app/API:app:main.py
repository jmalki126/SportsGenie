import os
import datetime
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# CONFIG
# ----------------------------

def load_dotenv_simple(path: str = ".env"):
    """Minimal .env loader (no extra dependency)."""
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

load_dotenv_simple()

SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY", "")
SPORTRADAR_ACCESS_LEVEL = os.getenv("SPORTRADAR_ACCESS_LEVEL", "trial")

# NOTE:
# Sportradar NFL endpoints vary by product/version.
# This endpoint works for many NFL trial keys:
# https://api.sportradar.com/nfl/official/trial/v7/en/games/{season}/{season_type}/{week}/schedule.json?api_key=...
#
# If your product uses a different path/version, we will adjust after you test once.

SPORTRADAR_BASE = f"https://api.sportradar.com/nfl/official/{SPORTRADAR_ACCESS_LEVEL}/v7/en"

# ----------------------------
# MOCK FALLBACK
# ----------------------------

MOCK_WEEKS = [
    {"week": 1, "season_type": "REG", "label": "Week 1", "game_count": 16}
]

MOCK_GAMES = [
    {
        "game_id": "2024_w10_buf_kc",
        "away_team": "BUF",
        "home_team": "KC",
        "kickoff_utc": "2024-11-10T21:25:00Z",
        "kickoff_display": "Sun 4:25 PM ET",
        "confidence": 0.62,
    }
]

def _coerce_week(value, default: int = 1) -> int:
    if value is None:
        return int(default)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip().replace("Week", "").replace("week", "").strip()
        return int(v)
    return int(value)

def mock_bundle(game_id: str, season: int, week: int, n_sims: int):
    g = next((x for x in MOCK_GAMES if x["game_id"] == game_id), None)
    if not g:
        g = {
            "game_id": game_id,
            "away_team": "AWAY",
            "home_team": "HOME",
            "kickoff_utc": None,
            "kickoff_display": None,
            "confidence": 0.55,
        }

    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "snapshot": {
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "kickoff_utc": g.get("kickoff_utc"),
            "kickoff_display": g.get("kickoff_display"),
            "proj_spread_home": -2.5,
            "proj_total": 46.0,
            "win_prob_home": 0.56,
            "win_prob_away": 0.44,
            "confidence": g.get("confidence", 0.60),
        },
        "explain": {
            "drivers": [
                {"name": "Team strength gap", "impact_points": 2.1, "reason": "Baseline team rating differential favors the home team."},
                {"name": "Home field", "impact_points": 1.5, "reason": "Home field advantage."},
            ],
            "flip_conditions": [
                {"condition": "Turnover margin swings by 2", "swing_points": 4.0},
                {"condition": "Wind above 18 mph", "swing_total_points": -3.0},
            ],
        },
        "simulate": {
            "n_sims": n_sims,
            "home_score_mean": 24.4,
            "away_score_mean": 22.0,
            "home_win_prob_sim": 0.57,
            "score_histogram": [
                {"home": 24, "away": 21, "p": 0.052},
                {"home": 27, "away": 20, "p": 0.041},
                {"home": 23, "away": 17, "p": 0.038},
            ],
        },
    }

# ----------------------------
# SPORTRADAR FETCH
# ----------------------------

def fetch_sport_radar_games(season: int, week: int, season_type: str = "REG"):
    """
    Returns a list of games in our frontend format:
    {game_id, away_team, home_team, kickoff_utc, kickoff_display, confidence}
    """
    if not SPORTRADAR_API_KEY:
        raise RuntimeError("Missing SPORTRADAR_API_KEY in environment")

    url = f"{SPORTRADAR_BASE}/games/{season}/{season_type}/{week}/schedule.json"
    r = requests.get(url, params={"api_key": SPORTRADAR_API_KEY}, timeout=20)

    if r.status_code != 200:
        raise RuntimeError(f"Sportradar error {r.status_code}: {r.text[:200]}")

    data = r.json()

    games_out = []
    games = data.get("week", {}).get("games", []) or data.get("games", [])

    for g in games:
        gid = g.get("id") or g.get("game", {}).get("id") or ""
        scheduled = g.get("scheduled") or g.get("game", {}).get("scheduled")
        home = g.get("home") or g.get("game", {}).get("home") or {}
        away = g.get("away") or g.get("game", {}).get("away") or {}

        home_abbr = home.get("alias") or home.get("name") or "HOME"
        away_abbr = away.get("alias") or away.get("name") or "AWAY"

        kickoff_utc = scheduled
        kickoff_display = scheduled

        # You can later compute real confidence from your model.
        confidence = 0.55

        games_out.append(
            {
                "game_id": gid,
                "away_team": away_abbr,
                "home_team": home_abbr,
                "kickoff_utc": kickoff_utc,
                "kickoff_display": kickoff_display,
                "confidence": confidence,
            }
        )

    return games_out

# ----------------------------
# ROUTES
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True, "build": "v2", "file": "app/main.py"}

@app.get("/health")
def health():
    return {"ok": True, "sportradar_key_loaded": bool(SPORTRADAR_API_KEY)}

@app.get("/weeks")
def weeks(season: int):
    # For now keep mock weeks. Next step we will pull weeks too.
    return {"season": season, "weeks": MOCK_WEEKS}

@app.get("/games")
def games(season: int, week: int, model_version: str = "v1"):
    week_num = _coerce_week(week, 1)

    try:
        real_games = fetch_sport_radar_games(season=season, week=week_num, season_type="REG")
        return {"season": season, "week": week_num, "games": real_games, "source": "sportradar"}
    except Exception as e:
        # fallback to mocks so the app never breaks
        return {"season": season, "week": week_num, "games": MOCK_GAMES, "source": f"mock_fallback: {str(e)}"}

class BundleReq(BaseModel):
    game_id: str
    season: int
    week: int
    model_version: str = "v1"
    n_sims: int = 10000

@app.post("/bundle")
def bundle(req: BundleReq):
    return mock_bundle(req.game_id, req.season, _coerce_week(req.week, 1), req.n_sims)

# ----------------------------
# LOVABLE POST COMPATIBILITY
# ----------------------------

class GamesReq(BaseModel):
    season: int
    week: int
    model_version: str = "v1"

class GameReq(BaseModel):
    game_id: str
    season: int
    week: int
    model_version: str = "v1"
    n_sims: int = 10000

@app.post("/games")
def games_post(req: GamesReq):
    return games(req.season, req.week, req.model_version)

@app.post("/game")
def game_post(req: GameReq):
    return mock_bundle(req.game_id, req.season, _coerce_week(req.week, 1), req.n_sims)["snapshot"]

@app.post("/simulate")
def simulate_post(req: GameReq):
    return mock_bundle(req.game_id, req.season, _coerce_week(req.week, 1), req.n_sims)["simulate"]

@app.post("/explain")
def explain_post(req: GameReq):
    return mock_bundle(req.game_id, req.season, _coerce_week(req.week, 1), req.n_sims)["explain"]