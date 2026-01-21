import os
from typing import List, Optional
from datetime import datetime

import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="SportsGenie API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ok for now; later you can lock this down to your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPORTTRADAR_API_KEY = os.getenv("SPORTTRADAR_API_KEY", "")


# -----------------------------
# Response models (clean API)
# -----------------------------
class Team(BaseModel):
    code: str
    name: Optional[str] = None


class GenieInfo(BaseModel):
    pick: str
    win_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    model_version: str = "v1"


class MarketInfo(BaseModel):
    spread: Optional[float] = None
    total: Optional[float] = None
    source: Optional[str] = None
    last_updated_utc: Optional[datetime] = None


class ExplainItem(BaseModel):
    key: str
    label: str
    impact_points: float


class GameOut(BaseModel):
    game_id: str
    season: int
    week: int
    season_type: str
    status: str
    kickoff_utc: str
    kickoff_display: str
    home: Team
    away: Team
    genie: GenieInfo
    market: MarketInfo
    explain: List[ExplainItem]


class GamesResponse(BaseModel):
    season: int
    week: int
    season_type: str
    games: List[GameOut]


# -----------------------------
# Helpers
# -----------------------------
def _require_api_key() -> str:
    if not SPORTTRADAR_API_KEY:
        raise HTTPException(status_code=500, detail="SPORTTRADAR_API_KEY not set on server")
    return SPORTTRADAR_API_KEY


def _sr_get_json(url: str, params: Optional[dict] = None) -> dict:
    key = _require_api_key()
    p = dict(params or {})
    p["api_key"] = key

    try:
        r = requests.get(url, params=p, timeout=20)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"SportRadar request failed: {str(e)}")

    if r.status_code >= 400:
        # keep it simple but readable
        raise HTTPException(status_code=502, detail=f"SportRadar error {r.status_code}: {r.text[:250]}")

    try:
        return r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="SportRadar returned non JSON response")


def _iso(dt_str: str) -> str:
    # If SportRadar returns ISO already, pass it through.
    # If empty or None, return empty string.
    return dt_str or ""


def _to_game_out(raw: dict, season: int, week: int, season_type: str) -> GameOut:
    game_id = raw.get("game_id") or raw.get("id") or ""
    home_code = raw.get("home_team") or raw.get("home") or ""
    away_code = raw.get("away_team") or raw.get("away") or ""

    kickoff_utc = _iso(raw.get("kickoff_utc") or raw.get("scheduled") or "")
    kickoff_display = raw.get("kickoff_display") or kickoff_utc

    # Defaults for now (your model can replace these later)
    win_probability = float(raw.get("win_probability", 0.50))
    confidence = float(raw.get("confidence", 0.50))

    # Simple pick default
    pick = raw.get("pick") or home_code or "TBD"

    return GameOut(
        game_id=game_id,
        season=season,
        week=week,
        season_type=season_type,
        status=raw.get("status", "scheduled"),
        kickoff_utc=kickoff_utc,
        kickoff_display=kickoff_display,
        home=Team(code=home_code, name=raw.get("home_name")),
        away=Team(code=away_code, name=raw.get("away_name")),
        genie=GenieInfo(
            pick=pick,
            win_probability=win_probability,
            confidence=confidence,
            model_version=raw.get("model_version", "v1"),
        ),
        market=MarketInfo(),
        explain=[
            ExplainItem(key="team_strength_gap", label="Team strength gap", impact_points=float(raw.get("impact_team_strength_gap", 0.0))),
            ExplainItem(key="home_field", label="Home field", impact_points=float(raw.get("impact_home_field", 0.0))),
        ],
    )


def _extract_games(sr_payload: dict) -> List[dict]:
    """
    Your SportRadar payload shape can vary depending on endpoint/package.
    We try common patterns and fall back gracefully.
    """
    if isinstance(sr_payload, dict):
        if isinstance(sr_payload.get("games"), list):
            return sr_payload["games"]
        if isinstance(sr_payload.get("week"), dict) and isinstance(sr_payload["week"].get("games"), list):
            return sr_payload["week"]["games"]
        if isinstance(sr_payload.get("data"), dict) and isinstance(sr_payload["data"].get("games"), list):
            return sr_payload["data"]["games"]
    return []


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/games", response_model=GamesResponse)
def games(
    season: int = Query(..., description="NFL season year, ex: 2024"),
    week: int = Query(..., description="NFL week number, ex: 1"),
    season_type: str = Query("REG", description="REG, PRE, PST"),
):
    """
    Returns a clean, frontend friendly list of games.

    IMPORTANT:
    - You must be on a SportRadar package that supports the endpoint used below.
    - If this endpoint differs from your current one, tell me what URL you were using and I will swap it.
    """

    # SportRadar NFL weekly schedule endpoint (commonly used):
    # If your account uses a different endpoint, paste yours and I will update it.
    url = f"https://api.sportradar.com/nfl/official/trial/v7/en/games/{season}/{season_type}/{week}/schedule.json"

    sr = _sr_get_json(url)
    raw_games = _extract_games(sr)

    if not raw_games:
        # Some SR endpoints put games at top level, some nest differently.
        # If you hit this, show me the sr keys and I will adjust.
        return GamesResponse(season=season, week=week, season_type=season_type, games=[])

    cleaned = [_to_game_out(g, season, week, season_type) for g in raw_games]
    return GamesResponse(season=season, week=week, season_type=season_type, games=cleaned)


@app.get("/games/{game_id}", response_model=GameOut)
def game_by_id(
    game_id: str,
    season: int = Query(..., description="Season year used to locate the game"),
    week: int = Query(..., description="Week used to locate the game"),
    season_type: str = Query("REG", description="REG, PRE, PST"),
):
    """
    Fetch a week schedule and return one game by id.
    This is simple and reliable for now.
    """
    url = f"https://api.sportradar.com/nfl/official/trial/v7/en/games/{season}/{season_type}/{week}/schedule.json"
    sr = _sr_get_json(url)
    raw_games = _extract_games(sr)

    for g in raw_games:
        gid = g.get("game_id") or g.get("id")
        if gid == game_id:
            return _to_game_out(g, season, week, season_type)

    raise HTTPException(status_code=404, detail="Game not found")