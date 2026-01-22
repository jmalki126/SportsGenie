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


def _to_display(dt_str: Optional[str]) -> str:
    """
    Converts ISO strings into a friendly display.
    Examples:
      2024-09-08T17:00:00+00:00
      2024-09-08T17:00:00Z
    """
    if not dt_str:
        return ""
    s = str(dt_str).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%a %b %d, %I:%M %p UTC")
    except Exception:
        return str(dt_str)


def _mock_games(season: int, week: int, season_type: str) -> List[Dict[str, Any]]:
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
        kickoff_utc = kickoff.isoformat().replace("+00:00", "Z")

        games.append(
            {
                "game_id": game_id,
                "away_team": away,
                "home_team": home,
                "kickoff_utc": kickoff_utc,
                "kickoff_display": _to_display(kickoff_utc),
                "confidence": round(0.45 + 0.45 * _stable_rand(game_id), 2),
                "source": "mock",
            }
        )

    return games


def _sportradar_schedule_url(season: int, week: int, season_type: str) -> str:
    # Common NFL schedule path for Sportradar trial feeds.
    # If your plan uses a different path, the code will fall back to mock data.
    return (
        f"{SPORTRADAR_BASE}/nfl/official/trial/v7/en/games/"
        f"{season}/{season_type}/{week}/schedule.json"
    )


def _try_fetch_sportradar_games(
    season: int, week: int, season_type: str
) -> Optional[List[Dict[str, Any]]]:
    if not SPORTRADAR_API_KEY:
        return None

    url = _sportradar_schedule_url(season, week, season_type)

    try:
        r = requests.get(url, params={"api_key": SPORTRADAR_API_KEY}, timeout=20)
        if r.status_code != 200:
            return None

        data = r.json()

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

            home_obj = g.get("home") or g.get("home_team") or {}
            away_obj = g.get("away") or g.get("away_team") or {}

            home = home_obj.get("alias") if isinstance(home_obj, dict) else str(home_obj)
            away = away_obj.get("alias") if isinstance(away_obj, dict) else str(away_obj)

            scheduled = g.get("scheduled") or g.get("kickoff") or g.get("scheduled_at")
            kickoff_utc = str(scheduled) if scheduled is not None else ""

            if not gid or not home or not away:
                continue

            games.append(
                {
                    "game_id": gid,
                    "away_team": away,
                    "home_team": home,
                    "kickoff_utc": kickoff_utc,
                    "kickoff_display": _to_display(kickoff_utc),
                    "confidence": round(0.45 + 0.45 * _stable_rand(gid), 2),
                    "source": "sportradar",
                }
            )

        return games if games else None

    except Exception:
        return None


@app.get("/")
def root():
    return {"status": "ok", "service": "sportsgenie", "time": _now_iso()}


@app.head("/")
def root_head():
    # Render may probe with HEAD requests
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


@app.get("/accuracy")
def accuracy(
    season: int = Query(2024),
    season_type: str = Query("REG"),
    weeks_back: int = Query(10),
    model_version: str = Query("v1"),
):
    """
    Historical accuracy summary.
    This is deterministic mock data until real results are stored.
    """

    total_games = weeks_back * 14  # approx games per week
    rng_seed = f"{season}:{season_type}:{weeks_back}:{model_version}"

    def win_rate(seed: str, base: float):
        return round(min(0.75, max(0.45, base + (_stable_rand(seed) - 0.5) * 0.1)), 2)

    ats_accuracy = win_rate(rng_seed + ":ats", 0.57)
    high_conf_accuracy = win_rate(rng_seed + ":high", 0.66)

    confidence_buckets = [
        ("0.50–0.59", win_rate(rng_seed + ":50", 0.52)),
        ("0.60–0.69", win_rate(rng_seed + ":60", 0.57)),
        ("0.70–0.79", win_rate(rng_seed + ":70", 0.62)),
        ("0.80+", win_rate(rng_seed + ":80", 0.68)),
    ]

    return {
        "season": season,
        "season_type": season_type,
        "weeks_back": weeks_back,
        "model_version": model_version,
        "summary": {
            "games": total_games,
            "ats_accuracy": ats_accuracy,
            "high_confidence_accuracy": high_conf_accuracy,
        },
        "by_confidence": [
            {"bucket": b, "accuracy": a} for b, a in confidence_buckets
        ],
        "status": "ok",
    }

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

    bets.sort(key=lambda x: x.get("edge", 0.0), reverse=True)
    n = max(1, min(20, int(top_n)))
    bets = bets[:n]

    return {
        "season": season,
        "week": week,
        "season_type": season_type,
        "top_n": n,
        "best_bets": bets,
        "status": "ok",
    }


@app.get("/accuracy")
def accuracy(
    season: int = Query(2024),
    season_type: str = Query("REG"),
    weeks_back: int = Query(10),
    model_version: str = Query("v1"),
    min_confidence: float = Query(0.60),
):
    """
    Placeholder historical accuracy metrics.
    This returns stable, deterministic numbers so the UI can be built now.
    Later, replace this with real graded results.
    """
    weeks_back = max(1, min(18, int(weeks_back)))
    seed_base = f"acc:{season}:{season_type}:{model_version}"

    ats = round(0.52 + 0.10 * (_stable_rand(seed_base + ":ats") - 0.5), 3)
    totals = round(0.51 + 0.10 * (_stable_rand(seed_base + ":totals") - 0.5), 3)
    high_conf_ats = round(ats + 0.03 + 0.05 * (_stable_rand(seed_base + ":hc") - 0.5), 3)

    ats = max(0.45, min(0.65, ats))
    totals = max(0.45, min(0.65, totals))
    high_conf_ats = max(0.48, min(0.70, high_conf_ats))

    buckets = [
        {
            "range": "50-59%",
            "pred": 0.55,
            "actual": round(0.53 + 0.06 * (_stable_rand(seed_base + ":b1") - 0.5), 3),
        },
        {
            "range": "60-69%",
            "pred": 0.65,
            "actual": round(0.64 + 0.06 * (_stable_rand(seed_base + ":b2") - 0.5), 3),
        },
        {
            "range": "70-79%",
            "pred": 0.75,
            "actual": round(0.73 + 0.06 * (_stable_rand(seed_base + ":b3") - 0.5), 3),
        },
        {
            "range": "80-89%",
            "pred": 0.85,
            "actual": round(0.82 + 0.06 * (_stable_rand(seed_base + ":b4") - 0.5), 3),
        },
    ]
    for b in buckets:
        b["actual"] = max(0.45, min(0.95, b["actual"]))

    return {
        "season": season,
        "season_type": season_type,
        "weeks_back": weeks_back,
        "model_version": model_version,
        "summary": {
            "ats_win_rate": ats,
            "totals_accuracy": totals,
            "high_confidence_ats_win_rate": high_conf_ats,
            "min_confidence": float(min_confidence),
        },
        "calibration": buckets,
        "status": "ok",
    }