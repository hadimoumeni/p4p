"""
Phase 3 — FastAPI backend for P4P rankings and predictions.

Endpoints:
  GET  /rankings              — current division rankings (champ + 15)
  GET  /rankings/{division}   — single division ranking
  GET  /p4p                   — pound-for-pound top 15
  GET  /goat                  — greatest of all time top 50
  GET  /predict               — predict fight outcome (?a=...&b=...)
  GET  /fighter/{name}        — fighter profile with rating, stats, recent fights
  GET  /disagreements         — where P4P rankings disagree with UFC official
"""

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from db import get_connection
from glicko2 import (
    ACTIVE_THRESHOLD_DAYS, MENS_DIVISIONS, FighterRating, build_ratings,
)
from predict import predict_fight

# ---------------------------------------------------------------------------
# App state — precompute ratings on startup
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Computing Glicko-2 ratings...")
    _state["ratings"] = build_ratings()
    print("API ready.")
    yield
    _state.clear()


app = FastAPI(
    title="P4P — UFC Algorithmic Rankings",
    description="Glicko-2 ratings, matchup predictions, and ranking disagreements",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_active(r: FighterRating) -> bool:
    if not r.last_fight_date:
        return False
    days = (datetime.now() - datetime.strptime(r.last_fight_date, "%Y-%m-%d")).days
    return days <= ACTIVE_THRESHOLD_DAYS


def _fighter_to_dict(r: FighterRating, rank: int | None = None) -> dict:
    d = {
        "fighter_id": r.fighter_id,
        "name": r.name,
        "rating": round(r.rating),
        "rd": round(r.rd),
        "volatility": round(r.volatility, 4),
        "division": r.division,
        "peak_rating": round(r.peak_rating),
        "fights": r.fights_count,
        "last_fight": r.last_fight_date,
        "active": _is_active(r),
    }
    if rank is not None:
        d["rank"] = rank
    return d


def _division_ranking(division: str) -> dict:
    ratings = _state["ratings"]
    active = [r for r in ratings.values() if _is_active(r) and r.division == division]
    active.sort(key=lambda r: r.rating, reverse=True)

    result = {"division": division, "champ": None, "ranked": []}
    if active:
        result["champ"] = _fighter_to_dict(active[0], rank=0)
        result["ranked"] = [_fighter_to_dict(f, rank=i) for i, f in enumerate(active[1:16], 1)]
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

DIVISION_NAMES = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
]


@app.get("/rankings")
def get_rankings():
    """Current division rankings for all 8 men's divisions."""
    return [_division_ranking(div) for div in DIVISION_NAMES]


@app.get("/rankings/{division}")
def get_division_ranking(division: str):
    """Ranking for a single division."""
    # Fuzzy match division name
    match = None
    for d in DIVISION_NAMES:
        if d.lower().replace(" ", "") == division.lower().replace(" ", "").replace("-", ""):
            match = d
            break
    if not match:
        raise HTTPException(404, f"Division not found: {division}. Options: {DIVISION_NAMES}")
    return _division_ranking(match)


@app.get("/p4p")
def get_p4p():
    """Pound-for-pound top 15."""
    ratings = _state["ratings"]
    active = [r for r in ratings.values() if _is_active(r) and r.division in MENS_DIVISIONS]
    active.sort(key=lambda r: r.rating, reverse=True)
    return [_fighter_to_dict(f, rank=i) for i, f in enumerate(active[:15], 1)]


@app.get("/goat")
def get_goat():
    """Greatest of all time top 50 by peak rating."""
    ratings = _state["ratings"]
    eligible = [
        r for r in ratings.values()
        if r.division in MENS_DIVISIONS and r.fights_count >= 3
    ]
    eligible.sort(key=lambda r: r.peak_rating, reverse=True)
    result = []
    for i, f in enumerate(eligible[:50], 1):
        d = _fighter_to_dict(f, rank=i)
        d["peak_rating"] = round(f.peak_rating)
        result.append(d)
    return result


@app.get("/predict")
def get_prediction(a: str = Query(..., description="Fighter A name"),
                   b: str = Query(..., description="Fighter B name")):
    """Predict fight outcome between two fighters."""
    result = predict_fight(a, b)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/fighter/{name}")
def get_fighter(name: str):
    """Fighter profile with rating, stats, and recent fights."""
    conn = get_connection()

    # Find fighter
    row = conn.execute(
        "SELECT fighter_id, name, nickname, height, reach, stance, dob, record "
        "FROM fighters WHERE name LIKE ?",
        (f"%{name}%",),
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(404, f"Fighter not found: {name}")

    fighter_id = row[0]
    profile = {
        "fighter_id": row[0],
        "name": row[1],
        "nickname": row[2],
        "height": row[3],
        "reach": row[4],
        "stance": row[5],
        "dob": row[6],
        "record": row[7],
    }

    # Rating info
    ratings = _state["ratings"]
    r = ratings.get(fighter_id)
    if r:
        profile["rating"] = round(r.rating)
        profile["rd"] = round(r.rd)
        profile["division"] = r.division
        profile["peak_rating"] = round(r.peak_rating)
        profile["active"] = _is_active(r)

    # Recent fights
    fights = conn.execute("""
        SELECT f.date, fa.name, fb.name, fw.name as winner,
               f.method, f.round, f.time, f.weight_class
        FROM fights f
        JOIN fighters fa ON f.fighter_a_id = fa.fighter_id
        JOIN fighters fb ON f.fighter_b_id = fb.fighter_id
        LEFT JOIN fighters fw ON f.winner_id = fw.fighter_id
        WHERE f.fighter_a_id = ? OR f.fighter_b_id = ?
        ORDER BY f.date DESC
        LIMIT 10
    """, (fighter_id, fighter_id)).fetchall()

    profile["recent_fights"] = [
        {
            "date": f[0],
            "fighter_a": f[1],
            "fighter_b": f[2],
            "winner": f[3],
            "method": f[4],
            "round": f[5],
            "time": f[6],
            "weight_class": f[7],
        }
        for f in fights
    ]

    # Career stats averages
    stats = conn.execute("""
        SELECT COUNT(*) as fights,
               AVG(knockdowns),
               AVG(significant_strikes_landed),
               AVG(significant_strikes_attempted),
               AVG(takedowns_landed),
               AVG(takedowns_attempted),
               AVG(submissions_attempted),
               AVG(control_time_seconds)
        FROM fight_stats
        WHERE fighter_id = ?
    """, (fighter_id,)).fetchone()

    if stats and stats[0] > 0:
        profile["career_averages"] = {
            "fights_with_stats": stats[0],
            "knockdowns_per_fight": round(stats[1] or 0, 2),
            "sig_strikes_landed_per_fight": round(stats[2] or 0, 1),
            "sig_strikes_attempted_per_fight": round(stats[3] or 0, 1),
            "sig_strike_accuracy": round((stats[2] or 0) / max(stats[3] or 1, 1) * 100, 1),
            "takedowns_landed_per_fight": round(stats[4] or 0, 2),
            "takedowns_attempted_per_fight": round(stats[5] or 0, 2),
            "takedown_accuracy": round((stats[4] or 0) / max(stats[5] or 1, 1) * 100, 1),
            "submissions_per_fight": round(stats[6] or 0, 2),
            "control_time_avg_seconds": round(stats[7] or 0, 0),
        }

    conn.close()
    return profile


@app.get("/disagreements")
def get_disagreements():
    """
    Where P4P algorithmic rankings most disagree with UFC official rankings.
    Returns divisions where the champ or top-5 differ significantly.
    """
    # UFC official champs as of late March 2026
    # This would ideally be scraped/updated, but hardcoding for now
    ufc_official = {
        "Flyweight": ["Alexandre Pantoja", "Brandon Royval", "Amir Albazi",
                       "Kai Kara-France", "Tatsuro Taira", "Manel Kape"],
        "Bantamweight": ["Merab Dvalishvili", "Umar Nurmagomedov", "Sean O'Malley",
                          "Petr Yan", "Cory Sandhagen", "Deiveson Figueiredo"],
        "Featherweight": ["Ilia Topuria", "Alexander Volkanovski", "Movsar Evloev",
                           "Diego Lopes", "Yair Rodriguez", "Lerone Murphy"],
        "Lightweight": ["Islam Makhachev", "Arman Tsarukyan", "Charles Oliveira",
                         "Justin Gaethje", "Dustin Poirier", "Benoit Saint Denis"],
        "Welterweight": ["Belal Muhammad", "Shavkat Rakhmonov", "Jack Della Maddalena",
                          "Kamaru Usman", "Ian Machado Garry", "Leon Edwards"],
        "Middleweight": ["Dricus Du Plessis", "Khamzat Chimaev", "Sean Strickland",
                          "Robert Whittaker", "Nassourdine Imavov", "Israel Adesanya"],
        "Light Heavyweight": ["Alex Pereira", "Magomed Ankalaev", "Jiri Prochazka",
                               "Jamahal Hill", "Khalil Rountree Jr.", "Carlos Ulberg"],
        "Heavyweight": ["Jon Jones", "Tom Aspinall", "Ciryl Gane",
                         "Stipe Miocic", "Sergei Pavlovich", "Alexander Volkov"],
    }

    disagreements = []

    for div in DIVISION_NAMES:
        algo = _division_ranking(div)
        official = ufc_official.get(div, [])
        if not official or not algo["champ"]:
            continue

        algo_names = [algo["champ"]["name"]] + [f["name"] for f in algo["ranked"][:5]]

        # Find disagreements
        for i, official_name in enumerate(official[:6]):
            rank_label = "C" if i == 0 else f"#{i}"
            # Where is this fighter in algo ranking?
            algo_pos = None
            for j, algo_name in enumerate(algo_names):
                if official_name.lower() in algo_name.lower() or algo_name.lower() in official_name.lower():
                    algo_pos = j
                    break

            if algo_pos is None:
                disagreements.append({
                    "division": div,
                    "fighter": official_name,
                    "ufc_rank": rank_label,
                    "algo_rank": "unranked in top 16",
                    "delta": "missing",
                    "note": f"UFC ranks {official_name} at {rank_label} but algo doesn't have them in top 16",
                })
            elif abs(algo_pos - i) >= 3:
                algo_label = "C" if algo_pos == 0 else f"#{algo_pos}"
                disagreements.append({
                    "division": div,
                    "fighter": official_name,
                    "ufc_rank": rank_label,
                    "algo_rank": algo_label,
                    "delta": algo_pos - i,
                    "note": f"UFC: {rank_label}, Algo: {algo_label}",
                })

    # Sort by magnitude of disagreement
    disagreements.sort(key=lambda d: abs(d["delta"]) if isinstance(d["delta"], int) else 99, reverse=True)
    return disagreements


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
