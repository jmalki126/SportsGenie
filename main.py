import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SportsGenie API", version="0.1.0")

# CORS: allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY", "").strip()
SPORTRADAR_BASE = os.getenv("SPORTRADAR_BASE_URL", "https://api.sportradar.com").rstrip("/")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_rand(seed: str) -> float:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _mock_games(season: int, week: int, season_type: str) -> List[Dict[str, Any]]:
    # Small deterministic list so the UI always has something to render.
    pairs = [
        ("BAL", "KC"),
        ("GB", "PHI"),
        ("CAR", "NO"),
        ("TEN", "CHI"),
        ("HOU", "IND"),
        ("ARI", "BUF"),
        ("NE", "CIN"),
        ("MIA", "NYJ"),
        ("SF", "DAL"),
    ]
    base_date = datetime(season, 9, 5, 20, 20, tzinfo=timezone.utc)
    games: List[Dict[str, Any]] = []

    for i, (away, home) in enumerate(pairs):
        game_id = f"mock-{season}-{season_type}-{week}-{away}-{home}"
        kickoff = base_date.replace(day=min(28, base_date.day + i))
        games.append(
            {
                "game_id": game_id,
                "away_team": away,
                "home_team": home,
                "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z"),
                "kickoff_display": kickoff.strftime("%Y-%m-%d %H:%M UTC"),
                "confidence": round(0.45 + 0.45 * _stable_rand(game_id), 2),
                "source": "mock",
            }
        )
    return games


def _sportradar_schedule_url(season: int, week: int, season_type: str) -> str:
    # Common-ish NFL schedule path. If your plan/path differs, code falls back to mock data.
    # Example:
    # https://api.sportradar.com/nfl/official/trial/v7/en/games/2024/REG/1/schedule.json?api_key=...
    return (
        f"{SPORTRADAR_BASE}/nfl/official/trial/v7/en/games/"
        f"{season}/{season_type}/{week}/schedule.json"
    )


def _try_fetch_sportradar_games(season: int, week: int, season_type: str) -> Optional[List[Dict[str, Any]]]:
    if not SPORTRADAR_API_KEY:
        return None

    url = _sportradar_schedule_url(season, week, season_type)
    try:
        r = requests.get(url, params={"api_key": SPORTRADAR_API_KEY}, timeout=20)
        if r.status_code != 200:
            return None

        data = r.json()

        # Try to locate games in a few common shapes.
        raw_games = None
        if isinstance(data, dict):
            raw_games = data.get("games") or data.get("week", {}).get("games")
            if raw_games is None and "weeks" in data:
                for w in data.get("weeks", []):
                    if str(w.get("sequence")) == str(week) or str(w.get("number")) == str(week):
                        raw_games = w.get("games")
                        break

        if not isinstance(raw_games, list):
            return None

        games: List[Dict[str, Any]] = []
        for g in raw_games:
            gid = g.get("id") or g.get("game", {}).get("id")

            # home/away can be nested objects or strings depending on endpoint version
            home_obj = g.get("home") or g.get("home_team") or {}
            away_obj = g.get("away") or g.get("away_team") or {}

            home = home_obj.get("alias") if isinstance(home_obj, dict) else str(home_obj)
            away = away_obj.get("alias") if isinstance(away_obj, dict) else str(away_obj)

            scheduled = g.get("scheduled") or g.get("kickoff") or g.get("scheduled_at")
            if not gid or not home or not away:
                continue

            kickoff_utc = scheduled
            kickoff_display = (
                kickoff_utc.replace("T", " ").replace("Z", " UTC")
                if isinstance(kickoff_utc, str) and kickoff_utc.endswith("Z")
                else str(kickoff_utc)
            )

            games.append(
                {
                    "game_id": gid,
                    "away_team": away,
                    "home_team": home,
                    "kickoff_utc": kickoff_utc,
                    "kickoff_display": kickoff_display,
                    "confidence": round(0.45 + 0.45 * _stable_rand(gid), 2),
                    "source": "sportradar",
                }
            )

        return games if games else None
    except Exception:
        return None


@app.get("/best-bets")
def best_bets(
    season: int = Query(2024),
    week: int = Query(1),
    season_type: str = Query("REG"),
    top_n: int = Query(5),
):
    """
    Returns the top games for the week by 'edge'.

    Placeholder now (no sportsbook market lines yet):
      edge = confidence * 100

    Later, when you add sportsbook lines:
      edge = abs(model_spread - market_spread) * confidence
    """
    sr_games = _try_fetch_sportradar_games(season, week, season_type)
    games_list = sr_games if sr_games is not None else _mock_games(season, week, season_type)

    bets: List[Dict[str, Any]] = []
    for g in games_list:
        confidence = float(g.get("confidence", 0.0))
        edge_score = round(confidence * 100.0, 1)

        bets.append(
            {
                "game_id": g.get("game_id"),
                "away_team": g.get("away_team"),
                "home_team": g.get("home_team"),
                "kickoff_utc": g.get("kickoff_utc"),
                "kickoff_display": g.get("kickoff_display"),
                "confidence": confidence,
                "edge": edge_score,
                "source": g.get("source", "unknown"),
            }
        )

    bets.sort(key=lambda x: x.get("edge", 0), reverse=True)
    bets = bets[: max(1, min(20, int(top_n)))]

    return {
        "season": season,
        "week": week,
        "season_type": season_type,
        "top_n": top_n,
        "best_bets": bets,
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "sportsgenie", "time": _now_iso()}


@app.head("/")
def root_head():
    # Render sometimes does a HEAD / check. Returning empty is fine.
    return {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/games")
def games(
    season: int = Query(2024),
    week: int = Query(1),
    season_type: str = Query("REG"),
    as_of_date: Optional[str] = Query(None),
):
    # Return schedule games the frontend can render.
    # as_of_date is accepted for compatibility even if not used.
    sr_games = _try_fetch_sportradar_games(season, week, season_type)
    out_games = sr_games if sr_games is not None else _mock_games(season, week, season_type)

    return {
        "season": season,
        "week": week,
        "season_type": season_type,
        "as_of_date": as_of_date,
        "games": out_games,
    }


@app.get("/simulate")
def simulate(
    game_id: str = Query(...),
    season: int = Query(2024),
    week: int = Query(1),
    model_version: str = Query("v1"),
    n_sims: int = Query(10000),
):
    # Deterministic "model" so UI works even before real model wiring is done.
    if not game_id or game_id.strip() in {"...", "null", "undefined"}:
        raise HTTPException(status_code=400, detail="game_id is required")

    seed = f"{game_id}:{season}:{week}:{model_version}:{n_sims}"
    r1 = _stable_rand(seed + ":a")
    r2 = _stable_rand(seed + ":b")
    r3 = _stable_rand(seed + ":c")

    win_prob = round(0.40 + 0.20 * r1 + 0.30 * (r2 - 0.5), 2)
    win_prob = max(0.01, min(0.99, win_prob))

    spread = round(-7 + 14 * (r2 - 0.5), 1)
    total = round(38 + 20 * r3, 1)
    confidence = round(0.50 + 0.45 * _stable_rand(seed + ":conf"), 2)

    team_ratings = {
        "home_power": round(60 + 30 * _stable_rand(seed + ":hp"), 1),
        "away_power": round(60 + 30 * _stable_rand(seed + ":ap"), 1),
        "home_off": round(60 + 35 * _stable_rand(seed + ":ho"), 1),
        "home_def": round(60 + 35 * _stable_rand(seed + ":hd"), 1),
        "away_off": round(60 + 35 * _stable_rand(seed + ":ao"), 1),
        "away_def": round(60 + 35 * _stable_rand(seed + ":ad"), 1),
    }
    home_field_adjustment = round(0.5 + 2.0 * _stable_rand(seed + ":hfa"), 1)

    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "model_version": model_version,
        "n_sims": n_sims,
        "win_prob": win_prob,
        "spread": spread,
        "total": total,
        "confidence": confidence,
        "team_ratings": team_ratings,
        "home_field_adjustment": home_field_adjustment,
        "status": "ok",
    }


@app.get("/explain")
def explain(
    game_id: str = Query(...),
    season: int = Query(2024),
    week: int = Query(1),
    model_version: str = Query("v1"),
):
    if not game_id or game_id.strip() in {"...", "null", "undefined"}:
        raise HTTPException(status_code=400, detail="game_id is required")

    seed = f"{game_id}:{season}:{week}:{model_version}:explain"
    f1 = round(0.5 + 3.0 * _stable_rand(seed + ":1"), 1)
    f2 = round(0.5 + 3.0 * _stable_rand(seed + ":2"), 1)
    f3 = round(0.5 + 3.0 * _stable_rand(seed + ":3"), 1)

    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "model_version": model_version,
        "factors": [
            {"rank": 1, "label": "Team strength gap", "points": f1},
            {"rank": 2, "label": "Home field", "points": f2},
            {"rank": 3, "label": "Injuries and availability", "points": f3},
        ],
        "status": "ok",
    }