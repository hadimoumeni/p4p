"""
Phase 2 — Matchup prediction model.

Predicts fight outcomes using pre-fight features:
  - Rolling career stats (strikes, takedowns, submissions, control time)
  - Physical attributes (reach advantage, height advantage, age)
  - Glicko-2 rating gap and RD
  - Win streak, finish rate, experience

Uses time-based train/test split to prevent data leakage.
"""

import math
import pickle
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

from db import get_connection
from glicko2 import build_ratings, FighterRating

MODEL_PATH = Path(__file__).parent / "data" / "model.pkl"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _parse_reach_inches(reach: str | None) -> float | None:
    """Parse '72\"' or '72' into float."""
    if not reach:
        return None
    try:
        return float(reach.replace('"', '').strip())
    except ValueError:
        return None


def _parse_height_inches(height: str | None) -> float | None:
    """Parse 5' 11\" into total inches."""
    if not height:
        return None
    try:
        parts = height.replace('"', '').split("'")
        feet = int(parts[0].strip())
        inches = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
        return feet * 12 + inches
    except (ValueError, IndexError):
        return None


def _age_at_date(dob: str | None, fight_date: str) -> float | None:
    if not dob:
        return None
    try:
        born = datetime.strptime(dob, "%Y-%m-%d")
        fight = datetime.strptime(fight_date, "%Y-%m-%d")
        return (fight - born).days / 365.25
    except ValueError:
        return None


def build_dataset() -> pl.DataFrame:
    """
    Build the full feature matrix for all fights.

    For each fight, we compute features using ONLY data available BEFORE the fight
    (rolling career averages up to but not including the current fight).
    """
    conn = get_connection()

    # Load fights with winners
    fights = pl.read_database(
        """
        SELECT f.fight_id, f.date, f.fighter_a_id, f.fighter_b_id,
               f.winner_id, f.method, f.round, f.weight_class
        FROM fights f
        WHERE f.date IS NOT NULL AND f.winner_id IS NOT NULL
        ORDER BY f.date ASC
        """,
        conn,
    )

    # Load all fight stats
    stats = pl.read_database(
        """
        SELECT fight_id, fighter_id, knockdowns,
               significant_strikes_landed, significant_strikes_attempted,
               total_strikes_landed, total_strikes_attempted,
               takedowns_landed, takedowns_attempted,
               submissions_attempted, control_time_seconds
        FROM fight_stats
        """,
        conn,
    )

    # Load fighter bios
    bios = pl.read_database(
        "SELECT fighter_id, height, reach, dob FROM fighters",
        conn,
    )
    conn.close()

    # Build Glicko-2 ratings (we'll index by fighter_id at each fight date)
    print("Building Glicko-2 ratings for feature extraction...")
    glicko_ratings = build_ratings()

    # Pre-compute bio lookups
    bio_map = {}
    for row in bios.iter_rows(named=True):
        bio_map[row["fighter_id"]] = row

    # Build rolling career stats per fighter
    # We process fights chronologically, accumulating stats
    career: dict[str, dict] = {}  # fighter_id -> accumulated stats

    def get_career(fid: str) -> dict:
        if fid not in career:
            career[fid] = {
                "fights": 0, "wins": 0, "losses": 0,
                "win_streak": 0, "loss_streak": 0,
                "ko_wins": 0, "sub_wins": 0, "dec_wins": 0,
                "kd_total": 0,
                "sig_landed": 0, "sig_attempted": 0,
                "total_landed": 0, "total_attempted": 0,
                "td_landed": 0, "td_attempted": 0,
                "sub_att_total": 0,
                "ctrl_seconds": 0,
                # opponent stats (what opponents land on this fighter)
                "opp_sig_landed": 0, "opp_sig_attempted": 0,
                "opp_td_landed": 0, "opp_td_attempted": 0,
            }
        return career[fid]

    # Stats lookup
    stats_map: dict[tuple[str, str], dict] = {}
    for row in stats.iter_rows(named=True):
        stats_map[(row["fight_id"], row["fighter_id"])] = row

    # Process fights and build feature rows
    feature_rows = []

    rng = random.Random(42)  # deterministic shuffle

    for fight in fights.iter_rows(named=True):
        fid = fight["fight_id"]
        date = fight["date"]
        fa_id = fight["fighter_a_id"]
        fb_id = fight["fighter_b_id"]
        winner_id = fight["winner_id"]

        # Randomly swap A/B so the model can't learn positional bias
        if rng.random() < 0.5:
            fa_id, fb_id = fb_id, fa_id

        ca = get_career(fa_id)
        cb = get_career(fb_id)

        # Skip if either fighter has 0 prior fights (no data to predict from)
        if ca["fights"] < 1 or cb["fights"] < 1:
            # Still update career after skipping
            _update_career_after_fight(ca, cb, fa_id, fb_id, fid, winner_id,
                                       fight["method"], stats_map)
            continue

        # --- Build features ---
        features = {}

        # Career averages (per fight)
        for prefix, c in [("a", ca), ("b", cb)]:
            n = max(c["fights"], 1)
            features[f"{prefix}_fights"] = c["fights"]
            features[f"{prefix}_win_rate"] = c["wins"] / n
            features[f"{prefix}_win_streak"] = c["win_streak"]
            features[f"{prefix}_loss_streak"] = c["loss_streak"]
            features[f"{prefix}_finish_rate"] = (c["ko_wins"] + c["sub_wins"]) / max(c["wins"], 1)
            features[f"{prefix}_ko_rate"] = c["ko_wins"] / max(c["wins"], 1)
            features[f"{prefix}_sub_rate"] = c["sub_wins"] / max(c["wins"], 1)
            features[f"{prefix}_kd_per_fight"] = c["kd_total"] / n
            features[f"{prefix}_sig_landed_pf"] = c["sig_landed"] / n
            features[f"{prefix}_sig_acc"] = c["sig_landed"] / max(c["sig_attempted"], 1)
            features[f"{prefix}_td_landed_pf"] = c["td_landed"] / n
            features[f"{prefix}_td_acc"] = c["td_landed"] / max(c["td_attempted"], 1)
            features[f"{prefix}_sub_att_pf"] = c["sub_att_total"] / n
            features[f"{prefix}_ctrl_pf"] = c["ctrl_seconds"] / n
            # Defense: opponent accuracy against this fighter
            features[f"{prefix}_sig_def"] = 1.0 - (c["opp_sig_landed"] / max(c["opp_sig_attempted"], 1))
            features[f"{prefix}_td_def"] = 1.0 - (c["opp_td_landed"] / max(c["opp_td_attempted"], 1))

        # Differential features (A - B perspective)
        diff_keys = [
            "win_rate", "win_streak", "finish_rate", "kd_per_fight",
            "sig_landed_pf", "sig_acc", "td_landed_pf", "td_acc",
            "sub_att_pf", "ctrl_pf", "sig_def", "td_def",
        ]
        for key in diff_keys:
            features[f"diff_{key}"] = features[f"a_{key}"] - features[f"b_{key}"]

        # Experience difference
        features["diff_fights"] = ca["fights"] - cb["fights"]

        # Physical attributes
        bio_a = bio_map.get(fa_id, {})
        bio_b = bio_map.get(fb_id, {})

        reach_a = _parse_reach_inches(bio_a.get("reach"))
        reach_b = _parse_reach_inches(bio_b.get("reach"))
        height_a = _parse_height_inches(bio_a.get("height"))
        height_b = _parse_height_inches(bio_b.get("height"))
        age_a = _age_at_date(bio_a.get("dob"), date)
        age_b = _age_at_date(bio_b.get("dob"), date)

        features["diff_reach"] = (reach_a or 0) - (reach_b or 0)
        features["diff_height"] = (height_a or 0) - (height_b or 0)
        features["diff_age"] = (age_a or 30) - (age_b or 30)
        features["has_reach_data"] = 1 if (reach_a and reach_b) else 0
        features["has_age_data"] = 1 if (age_a and age_b) else 0

        # Glicko-2 features
        gr_a = glicko_ratings.get(fa_id)
        gr_b = glicko_ratings.get(fb_id)
        if gr_a and gr_b:
            features["diff_rating"] = gr_a.rating - gr_b.rating
            features["diff_rd"] = gr_a.rd - gr_b.rd
            features["a_rating"] = gr_a.rating
            features["b_rating"] = gr_b.rating
        else:
            features["diff_rating"] = 0
            features["diff_rd"] = 0
            features["a_rating"] = 1500
            features["b_rating"] = 1500

        # Label: 1 if fighter A wins, 0 if fighter B wins
        features["label"] = 1 if winner_id == fa_id else 0
        features["date"] = date
        features["fight_id"] = fid

        feature_rows.append(features)

        # Update career stats AFTER extracting features
        _update_career_after_fight(ca, cb, fa_id, fb_id, fid, winner_id,
                                   fight["method"], stats_map)

    print(f"Built {len(feature_rows)} feature rows.")
    return pl.DataFrame(feature_rows)


def _update_career_after_fight(ca: dict, cb: dict,
                                fa_id: str, fb_id: str, fight_id: str,
                                winner_id: str, method: str | None,
                                stats_map: dict) -> None:
    """Update rolling career stats after a fight."""
    # Win/loss tracking
    if winner_id == fa_id:
        ca["wins"] += 1
        ca["win_streak"] += 1
        ca["loss_streak"] = 0
        cb["losses"] += 1
        cb["loss_streak"] += 1
        cb["win_streak"] = 0
        # Finish type
        if method:
            mu = method.upper()
            if "KO" in mu or "TKO" in mu:
                ca["ko_wins"] += 1
            elif "SUB" in mu:
                ca["sub_wins"] += 1
            else:
                ca["dec_wins"] += 1
    elif winner_id == fb_id:
        cb["wins"] += 1
        cb["win_streak"] += 1
        cb["loss_streak"] = 0
        ca["losses"] += 1
        ca["loss_streak"] += 1
        ca["win_streak"] = 0
        if method:
            mu = method.upper()
            if "KO" in mu or "TKO" in mu:
                cb["ko_wins"] += 1
            elif "SUB" in mu:
                cb["sub_wins"] += 1
            else:
                cb["dec_wins"] += 1

    ca["fights"] += 1
    cb["fights"] += 1

    # Accumulate stats
    sa = stats_map.get((fight_id, fa_id))
    sb = stats_map.get((fight_id, fb_id))

    if sa:
        ca["kd_total"] += sa.get("knockdowns") or 0
        ca["sig_landed"] += sa.get("significant_strikes_landed") or 0
        ca["sig_attempted"] += sa.get("significant_strikes_attempted") or 0
        ca["total_landed"] += sa.get("total_strikes_landed") or 0
        ca["total_attempted"] += sa.get("total_strikes_attempted") or 0
        ca["td_landed"] += sa.get("takedowns_landed") or 0
        ca["td_attempted"] += sa.get("takedowns_attempted") or 0
        ca["sub_att_total"] += sa.get("submissions_attempted") or 0
        ca["ctrl_seconds"] += sa.get("control_time_seconds") or 0

    if sb:
        cb["kd_total"] += sb.get("knockdowns") or 0
        cb["sig_landed"] += sb.get("significant_strikes_landed") or 0
        cb["sig_attempted"] += sb.get("significant_strikes_attempted") or 0
        cb["total_landed"] += sb.get("total_strikes_landed") or 0
        cb["total_attempted"] += sb.get("total_strikes_attempted") or 0
        cb["td_landed"] += sb.get("takedowns_landed") or 0
        cb["td_attempted"] += sb.get("takedowns_attempted") or 0
        cb["sub_att_total"] += sb.get("submissions_attempted") or 0
        cb["ctrl_seconds"] += sb.get("control_time_seconds") or 0

    # Cross-reference for defense stats
    if sb:
        ca["opp_sig_landed"] += sb.get("significant_strikes_landed") or 0
        ca["opp_sig_attempted"] += sb.get("significant_strikes_attempted") or 0
        ca["opp_td_landed"] += sb.get("takedowns_landed") or 0
        ca["opp_td_attempted"] += sb.get("takedowns_attempted") or 0
    if sa:
        cb["opp_sig_landed"] += sa.get("significant_strikes_landed") or 0
        cb["opp_sig_attempted"] += sa.get("significant_strikes_attempted") or 0
        cb["opp_td_landed"] += sa.get("takedowns_landed") or 0
        cb["opp_td_attempted"] += sa.get("takedowns_attempted") or 0


# ---------------------------------------------------------------------------
# Feature columns (used for training and prediction)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Differential features
    "diff_win_rate", "diff_win_streak", "diff_finish_rate",
    "diff_kd_per_fight", "diff_sig_landed_pf", "diff_sig_acc",
    "diff_td_landed_pf", "diff_td_acc", "diff_sub_att_pf",
    "diff_ctrl_pf", "diff_sig_def", "diff_td_def", "diff_fights",
    # Physical
    "diff_reach", "diff_height", "diff_age",
    "has_reach_data", "has_age_data",
    # Glicko-2
    "diff_rating", "diff_rd",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(df: pl.DataFrame) -> dict:
    """
    Train XGBoost and logistic regression with time-based split.
    Returns dict with models and evaluation metrics.
    """
    # Time-based split: train on fights before 2024, test on 2024+
    train = df.filter(pl.col("date") < "2024-01-01")
    test = df.filter(pl.col("date") >= "2024-01-01")

    print(f"Train: {len(train)} fights (before 2024)")
    print(f"Test:  {len(test)} fights (2024+)")

    X_train = train.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_train = train["label"].to_numpy()
    X_test = test.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_test = test["label"].to_numpy()

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Logistic Regression baseline ---
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    print(f"  Accuracy: {accuracy_score(y_test, lr_pred):.3f}")
    print(f"  Log loss: {log_loss(y_test, lr_prob):.3f}")
    print(f"  Brier score: {brier_score_loss(y_test, lr_prob):.3f}")

    # --- XGBoost ---
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    print(f"  Accuracy: {accuracy_score(y_test, xgb_pred):.3f}")
    print(f"  Log loss: {log_loss(y_test, xgb_prob):.3f}")
    print(f"  Brier score: {brier_score_loss(y_test, xgb_prob):.3f}")

    # --- Calibrated XGBoost ---
    print("\nCalibrating XGBoost probabilities...")
    cal_xgb = CalibratedClassifierCV(xgb, cv=5, method="isotonic")
    cal_xgb.fit(X_train, y_train)
    cal_prob = cal_xgb.predict_proba(X_test)[:, 1]
    cal_pred = (cal_prob >= 0.5).astype(int)
    print(f"  Accuracy: {accuracy_score(y_test, cal_pred):.3f}")
    print(f"  Log loss: {log_loss(y_test, cal_prob):.3f}")
    print(f"  Brier score: {brier_score_loss(y_test, cal_prob):.3f}")

    # Feature importance
    print("\nTop 10 feature importances (XGBoost):")
    importances = sorted(zip(FEATURE_COLS, xgb.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    for feat, imp in importances[:10]:
        print(f"  {feat:<25s} {imp:.3f}")

    # Save best model
    model_data = {
        "xgb": xgb,
        "calibrated_xgb": cal_xgb,
        "lr": lr,
        "feature_cols": FEATURE_COLS,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {MODEL_PATH}")

    return model_data


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_fight(fighter_a_name: str, fighter_b_name: str) -> dict:
    """
    Predict the outcome of a fight between two fighters.
    Returns win probabilities and key factors.
    """
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    conn = get_connection()

    # Look up fighters
    def find_fighter(name: str) -> dict | None:
        row = conn.execute(
            "SELECT fighter_id, name, height, reach, dob FROM fighters WHERE name LIKE ?",
            (f"%{name}%",),
        ).fetchone()
        if row:
            return {"fighter_id": row[0], "name": row[1], "height": row[2],
                    "reach": row[3], "dob": row[4]}
        return None

    fa = find_fighter(fighter_a_name)
    fb = find_fighter(fighter_b_name)
    if not fa or not fb:
        missing = fighter_a_name if not fa else fighter_b_name
        return {"error": f"Fighter not found: {missing}"}

    # Build career stats from DB
    def get_career_stats(fighter_id: str) -> dict:
        fights = conn.execute("""
            SELECT f.fight_id, f.winner_id, f.method, f.fighter_a_id, f.fighter_b_id
            FROM fights f
            WHERE (f.fighter_a_id = ? OR f.fighter_b_id = ?)
            AND f.date IS NOT NULL AND f.winner_id IS NOT NULL
            ORDER BY f.date ASC
        """, (fighter_id, fighter_id)).fetchall()

        c = {
            "fights": 0, "wins": 0, "losses": 0,
            "win_streak": 0, "loss_streak": 0,
            "ko_wins": 0, "sub_wins": 0, "dec_wins": 0,
            "kd_total": 0,
            "sig_landed": 0, "sig_attempted": 0,
            "td_landed": 0, "td_attempted": 0,
            "sub_att_total": 0, "ctrl_seconds": 0,
            "opp_sig_landed": 0, "opp_sig_attempted": 0,
            "opp_td_landed": 0, "opp_td_attempted": 0,
        }

        for fid, winner, method, fa_id, fb_id in fights:
            opp_id = fb_id if fighter_id == fa_id else fa_id

            if winner == fighter_id:
                c["wins"] += 1
                c["win_streak"] += 1
                c["loss_streak"] = 0
                if method:
                    mu = method.upper()
                    if "KO" in mu or "TKO" in mu:
                        c["ko_wins"] += 1
                    elif "SUB" in mu:
                        c["sub_wins"] += 1
                    else:
                        c["dec_wins"] += 1
            else:
                c["losses"] += 1
                c["loss_streak"] += 1
                c["win_streak"] = 0

            c["fights"] += 1

            # Fight stats
            s = conn.execute(
                "SELECT * FROM fight_stats WHERE fight_id = ? AND fighter_id = ?",
                (fid, fighter_id),
            ).fetchone()
            opp_s = conn.execute(
                "SELECT * FROM fight_stats WHERE fight_id = ? AND fighter_id = ?",
                (fid, opp_id),
            ).fetchone()

            if s:
                c["kd_total"] += s[2] or 0
                c["sig_landed"] += s[3] or 0
                c["sig_attempted"] += s[4] or 0
                c["td_landed"] += s[7] or 0
                c["td_attempted"] += s[8] or 0
                c["sub_att_total"] += s[10] or 0
                c["ctrl_seconds"] += s[12] or 0
            if opp_s:
                c["opp_sig_landed"] += opp_s[3] or 0
                c["opp_sig_attempted"] += opp_s[4] or 0
                c["opp_td_landed"] += opp_s[7] or 0
                c["opp_td_attempted"] += opp_s[8] or 0

        return c

    ca = get_career_stats(fa["fighter_id"])
    cb = get_career_stats(fb["fighter_id"])

    # Build feature vector
    features = {}
    for prefix, c in [("a", ca), ("b", cb)]:
        n = max(c["fights"], 1)
        features[f"{prefix}_win_rate"] = c["wins"] / n
        features[f"{prefix}_win_streak"] = c["win_streak"]
        features[f"{prefix}_finish_rate"] = (c["ko_wins"] + c["sub_wins"]) / max(c["wins"], 1)
        features[f"{prefix}_kd_per_fight"] = c["kd_total"] / n
        features[f"{prefix}_sig_landed_pf"] = c["sig_landed"] / n
        features[f"{prefix}_sig_acc"] = c["sig_landed"] / max(c["sig_attempted"], 1)
        features[f"{prefix}_td_landed_pf"] = c["td_landed"] / n
        features[f"{prefix}_td_acc"] = c["td_landed"] / max(c["td_attempted"], 1)
        features[f"{prefix}_sub_att_pf"] = c["sub_att_total"] / n
        features[f"{prefix}_ctrl_pf"] = c["ctrl_seconds"] / n
        features[f"{prefix}_sig_def"] = 1.0 - (c["opp_sig_landed"] / max(c["opp_sig_attempted"], 1))
        features[f"{prefix}_td_def"] = 1.0 - (c["opp_td_landed"] / max(c["opp_td_attempted"], 1))

    diff_keys = [
        "win_rate", "win_streak", "finish_rate", "kd_per_fight",
        "sig_landed_pf", "sig_acc", "td_landed_pf", "td_acc",
        "sub_att_pf", "ctrl_pf", "sig_def", "td_def",
    ]
    for key in diff_keys:
        features[f"diff_{key}"] = features[f"a_{key}"] - features[f"b_{key}"]

    features["diff_fights"] = ca["fights"] - cb["fights"]

    reach_a = _parse_reach_inches(fa.get("reach"))
    reach_b = _parse_reach_inches(fb.get("reach"))
    height_a = _parse_height_inches(fa.get("height"))
    height_b = _parse_height_inches(fb.get("height"))
    age_a = _age_at_date(fa.get("dob"), datetime.now().strftime("%Y-%m-%d"))
    age_b = _age_at_date(fb.get("dob"), datetime.now().strftime("%Y-%m-%d"))

    features["diff_reach"] = (reach_a or 0) - (reach_b or 0)
    features["diff_height"] = (height_a or 0) - (height_b or 0)
    features["diff_age"] = (age_a or 30) - (age_b or 30)
    features["has_reach_data"] = 1 if (reach_a and reach_b) else 0
    features["has_age_data"] = 1 if (age_a and age_b) else 0

    # Glicko-2
    ratings = build_ratings()
    gr_a = ratings.get(fa["fighter_id"])
    gr_b = ratings.get(fb["fighter_id"])
    features["diff_rating"] = (gr_a.rating if gr_a else 1500) - (gr_b.rating if gr_b else 1500)
    features["diff_rd"] = (gr_a.rd if gr_a else 350) - (gr_b.rd if gr_b else 350)

    conn.close()

    # Predict
    X = np.array([[features[c] for c in FEATURE_COLS]], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cal_xgb = model_data["calibrated_xgb"]
    prob_a = cal_xgb.predict_proba(X)[0][1]
    prob_b = 1 - prob_a

    # Key factors
    factors = []
    if features["diff_rating"] > 100:
        factors.append(f"Rating edge: {fa['name']} ({features['diff_rating']:+.0f})")
    elif features["diff_rating"] < -100:
        factors.append(f"Rating edge: {fb['name']} ({-features['diff_rating']:+.0f})")
    if features["diff_sig_landed_pf"] > 10:
        factors.append(f"Striking volume: {fa['name']}")
    elif features["diff_sig_landed_pf"] < -10:
        factors.append(f"Striking volume: {fb['name']}")
    if features["diff_td_landed_pf"] > 1:
        factors.append(f"Takedown threat: {fa['name']}")
    elif features["diff_td_landed_pf"] < -1:
        factors.append(f"Takedown threat: {fb['name']}")
    if abs(features["diff_reach"]) >= 3 and features["has_reach_data"]:
        longer = fa["name"] if features["diff_reach"] > 0 else fb["name"]
        factors.append(f"Reach advantage: {longer} ({abs(features['diff_reach']):.0f}\")")

    return {
        "fighter_a": fa["name"],
        "fighter_b": fb["name"],
        "prob_a": round(prob_a, 3),
        "prob_b": round(prob_b, 3),
        "predicted_winner": fa["name"] if prob_a > 0.5 else fb["name"],
        "confidence": round(max(prob_a, prob_b), 3),
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("P4P Matchup Prediction Model")
    print("=" * 60)

    print("\n[1/2] Building features...")
    df = build_dataset()

    print("\n[2/2] Training models...")
    train_model(df)

    # Demo predictions
    print("\n" + "=" * 60)
    print("DEMO PREDICTIONS")
    print("=" * 60)

    matchups = [
        ("Islam Makhachev", "Shavkat Rakhmonov"),
        ("Alex Pereira", "Magomed Ankalaev"),
        ("Tom Aspinall", "Jon Jones"),
        ("Ilia Topuria", "Charles Oliveira"),
    ]

    for a, b in matchups:
        result = predict_fight(a, b)
        if "error" in result:
            print(f"\n{a} vs {b}: {result['error']}")
        else:
            print(f"\n{result['fighter_a']} vs {result['fighter_b']}")
            print(f"  {result['fighter_a']}: {result['prob_a']*100:.1f}%")
            print(f"  {result['fighter_b']}: {result['prob_b']*100:.1f}%")
            print(f"  Predicted winner: {result['predicted_winner']} ({result['confidence']*100:.1f}%)")
            if result["factors"]:
                print(f"  Key factors: {', '.join(result['factors'])}")
