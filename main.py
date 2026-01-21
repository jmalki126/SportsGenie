import os
import math
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SportsGenie API", version="1.0.0")
@app.get("/")
def root():
    return {"status": "ok"}

# Allow your Lovable frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for now. Later you can lock to your domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY", "").strip()

# ---- Helpers ----

TEAM_NAME_MAP = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _stable_number(seed: str, lo: float, hi: float) -> float:
    """
    Deterministic pseudo random number between lo and hi based on seed.
    """
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    n = int(h[:8], 16) / 0xFFFFFFFF
    return lo + (hi - lo) * n

def _team_full_name(team_key: str) -> str:
    return TEAM_NAME_MAP.get(team_key, team_key)

def _team_ratings(team_key: str, season: int) -> Dict[str, Any]:
    """
    Placeholder ratings. Deterministic so it feels consistent.
    """
    power = _stable_number(f"power:{season}:{team_key}", 68, 92)
    offense = _stable_number(f"offense:{season}:{team_key}", 65, 95)
    defense = _stable_number(f"defense:{season}:{team_key}", 65, 95)
    return {
        "team_key": team_key,
        "team_name": _team_full_name(team_key),
        "power": round(power, 1),
        "offense": round(offense, 1),
        "defense": round(defense, 1),
    }

def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _compute_placeholder_prediction(
    season: int,
    home_team_key: str,
    away_team_key: str,
    model_version: str = "v1",
) -> Dict[str, Any]:
    """
    Produces placeholder model outputs that look realistic.
    """
    home = _team_ratings(home_team_key, season)
    away = _team_ratings(away_team_key, season)

    home_field_adjustment = 1.5  # points, simple placeholder
    power_diff = (home["power"] - away["power"]) + home_field_adjustment

    # win probability derived from power_diff (scaled)
    win_prob_home = _logistic(power_diff / 6.5)

    # spreads and totals as placeholders
    projected_spread_home = -round(power_diff / 2.2, 1)  # negative means home favored
    projected_total = round(_stable_number(f"total:{season}:{home_team_key}:{away_team_key}", 38, 54), 1)

    # confidence (0.45–0.75)
    confidence = float(round(0.45 + abs(power_diff) / 40.0, 2))
    confidence = max(0.45, min(confidence, 0.75))

    drivers = [
        {"name": "Team strength gap", "points": round((home["power"] - away["power"]) / 10.0, 2)},
        {"name": "Home field", "points": round(home_field_adjustment / 1.0, 2)},
        {"name": "Offense vs defense", "points": round((home["offense"] - away["defense"]) / 15.0, 2)},
    ]

    return {
        "model_version": model_version,
        "home_team_key": home_team_key,
        "away_team_key": away_team_key,
        "team_ratings": {"home": home, "away": away},
        "home_field_adjustment": home_field_adjustment,
        "win_prob_home": round(win_prob_home, 4),
        "projected_spread_home": projected_spread_home,
        "projected_total": projected_total,
        "confidence": confidence,
        "drivers": drivers,
    }

def _sportradar_schedule_url(season: int, week: int, season_type: str) -> Optional[str]:
    """
    SportRadar endpoint patterns can vary by plan/version.
    This tries a commonly used path. If it fails, we fallback.
    """
    if not SPORTRADAR_API_KEY:
        return None

    # Common pattern for NFL schedule by week
    # If this ever 404s, the fallback still keeps your UI working.
    season_type = season_type.upper()
    return (
        f"https://api.sportradar.com/nfl/official/trial/v7/en/games/"
        f"{season}/{season_type}/{week}/schedule.json?api_key={SPORTRADAR_API_KEY}"
    )

def _fetch_games_from_sportradar(season: int, week: int, season_type: str) -> List[Dict[str, Any]]:
    url = _sportradar_schedule_url(season, week, season_type)
    if not url:
        return []

    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()

        games: List[Dict[str, Any]] = []

        # SportRadar schedules often contain "week" -> "games"
        # This parsing is defensive.
        raw_games = []
        if isinstance(data, dict):
            if "week" in data and isinstance(data["week"], dict) and "games" in data["week"]:
                raw_games = data["week"]["games"] or []
            elif "games" in data:
                raw_games = data["games"] or []

        for g in raw_games:
            try:
                # Try a few common shapes
                game_id = g.get("id") or g.get("game", {}).get("id") or g.get("game_id")
                scheduled = g.get("scheduled") or g.get("game", {}).get("scheduled")
                home = g.get("home") or g.get("game", {}).get("home") or {}
                away = g.get("away") or g.get("game", {}).get("away") or {}

                home_key = home.get("alias") or home.get("id") or home.get("abbreviation") or "HOME"
                away_key = away.get("alias") or away.get("id") or away.get("abbreviation") or "AWAY"

                kickoff_utc = scheduled or _now_iso()
                # Try making a friendly display string
                kickoff_display = kickoff_utc.replace("T", " ").replace("Z", " UTC")

                games.append({
                    "game_id": game_id or f"{season}_{season_type}_wk{week}_{away_key}_at_{home_key}",
                    "home_team": home_key,
                    "away_team": away_key,
                    "kickoff_utc": kickoff_utc,
                    "kickoff_display": kickoff_display,
                    "source": "sportradar",
                })
            except Exception:
                continue

        return games
    except Exception:
        return []

def _fallback_games(season: int, week: int, season_type: str) -> List[Dict[str, Any]]:
    # Simple fallback so your UI always has something
    sample = [
        ("BUF", "KC", "2024-09-08T00:20:00+00:00"),
        ("SF", "DAL", "2024-09-08T20:25:00+00:00"),
        ("MIA", "NYJ", "2024-09-08T17:00:00+00:00"),
    ]
    out = []
    for away, home, kickoff in sample:
        out.append({
            "game_id": f"{season}_{season_type}_wk{week}_{away}_at_{home}",
            "home_team": home,
            "away_team": away,
            "kickoff_utc": kickoff,
            "kickoff_display": kickoff.replace("T", " ").replace("+00:00", " UTC"),
            "source": "fallback",
        })
    return out

# ---- Routes ----

@app.get("/health")
def health():
    return {"ok": True, "time_utc": _now_iso()}

@app.get("/games")
def games(
    season: int = Query(...),
    week: int = Query(...),
    season_type: str = Query("REG"),
):
    # Prefer SportRadar, fallback if unavailable
    schedule = _fetch_games_from_sportradar(season, week, season_type)
    if not schedule:
        schedule = _fallback_games(season, week, season_type)

    # Add the new fields the frontend wants at list level (safe + helpful)
    enriched = []
    for g in schedule:
        pred = _compute_placeholder_prediction(
            season=season,
            home_team_key=g["home_team"],
            away_team_key=g["away_team"],
            model_version="v1",
        )
        enriched.append({
            **g,
            "confidence": pred["confidence"],
            "model_version": pred["model_version"],
            "team_ratings": pred["team_ratings"],
            "home_field_adjustment": pred["home_field_adjustment"],
        })

    return {
        "season": season,
        "week": week,
        "season_type": season_type.upper(),
        "games": enriched,
    }

@app.get("/game")
def game(
    game_id: str = Query(...),
    season: int = Query(...),
    week: int = Query(...),
    season_type: str = Query("REG"),
    home_team_key: str = Query(...),
    away_team_key: str = Query(...),
    as_of_date: Optional[str] = Query(None),
    model_version: str = Query("v1"),
):
    pred = _compute_placeholder_prediction(
        season=season,
        home_team_key=home_team_key,
        away_team_key=away_team_key,
        model_version=model_version,
    )

    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "season_type": season_type.upper(),
        "as_of_date": as_of_date,
        "home_team_key": home_team_key,
        "away_team_key": away_team_key,
        "home_team_name": _team_full_name(home_team_key),
        "away_team_name": _team_full_name(away_team_key),
        **pred,
    }

@app.get("/simulate")
def simulate(
    game_id: str = Query(...),
    season: int = Query(...),
    week: int = Query(...),
    season_type: str = Query("REG"),
    home_team_key: str = Query(...),
    away_team_key: str = Query(...),
    model_version: str = Query("v1"),
    n_sims: int = Query(10000),
):
    pred = _compute_placeholder_prediction(
        season=season,
        home_team_key=home_team_key,
        away_team_key=away_team_key,
        model_version=model_version,
    )

    # Placeholder simulation summary
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "season_type": season_type.upper(),
        "model_version": pred["model_version"],
        "n_sims": n_sims,
        "win_prob_home": pred["win_prob_home"],
        "projected_spread_home": pred["projected_spread_home"],
        "projected_total": pred["projected_total"],
        "confidence": pred["confidence"],
        "team_ratings": pred["team_ratings"],
        "home_field_adjustment": pred["home_field_adjustment"],
    }

@app.get("/explain")
def explain(
    game_id: str = Query(...),
    season: int = Query(...),
    week: int = Query(...),
    season_type: str = Query("REG"),
    home_team_key: str = Query(...),
    away_team_key: str = Query(...),
    model_version: str = Query("v1"),
):
    pred = _compute_placeholder_prediction(
        season=season,
        home_team_key=home_team_key,
        away_team_key=away_team_key,
        model_version=model_version,
    )

    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "season_type": season_type.upper(),
        "model_version": pred["model_version"],
        "drivers": pred["drivers"],
        "notes": "Placeholder explanation. Replace with real model feature contributions later.",
    }