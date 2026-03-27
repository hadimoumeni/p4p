"""
Glicko-2 rating engine for UFC fighters with a performance multiplier layer.

References:
- Mark Glickman, "Example of the Glicko-2 system" (2013)
- http://www.glicko.net/glicko/glicko2.pdf

The Glicko-2 system tracks three values per fighter:
  μ  (mu)    — rating (skill estimate)
  φ  (phi)   — rating deviation (uncertainty)
  σ  (sigma) — volatility (consistency of performance)
"""

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from db import get_connection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Glicko-2 scale conversion (Glicko-2 works on a different scale internally)
GLICKO2_SCALE = 173.7178

# Default rating values (on the Glicko-1 scale for display)
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0  # high uncertainty for new fighters
DEFAULT_VOLATILITY = 0.06

# System constant τ — constrains volatility change. Lower = more conservative.
# Glickman recommends 0.3–1.2. MMA is volatile, so we use 0.5.
TAU = 0.5

# Convergence tolerance for volatility iteration
EPSILON = 1e-6

# RD inflation per rating period (days) of inactivity
# After ~18 months of inactivity, RD approaches the default (high uncertainty)
RD_INFLATION_PER_DAY = 0.5

# Men's divisions we track
MENS_DIVISIONS = {
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
}

# Active fighter threshold (days since last fight)
ACTIVE_THRESHOLD_DAYS = 548  # ~18 months


# ---------------------------------------------------------------------------
# Performance multiplier — how much a result moves the rating
# ---------------------------------------------------------------------------

def performance_multiplier(method: str | None, fight_round: int | None,
                           stats: dict | None) -> float:
    """
    Returns a multiplier (0.8–1.5) that scales how much a win/loss moves ratings.

    - KO/TKO finish:        base 1.2, early round bonus up to 1.5
    - Submission finish:     base 1.15, early round bonus up to 1.4
    - Decision (unanimous):  1.0
    - Decision (split):      0.85 (close fight = less informative)
    - DQ / Doctor stoppage:  0.8 (not very informative)

    Dominance bonus from stats: up to +0.1 for extreme stat dominance.
    """
    if not method:
        return 1.0

    method_upper = method.upper()

    # Base multiplier by method
    if "KO" in method_upper or "TKO" in method_upper:
        base = 1.2
        # Early finish bonus
        if fight_round and fight_round == 1:
            base = 1.5
        elif fight_round and fight_round == 2:
            base = 1.35
    elif "SUB" in method_upper:
        base = 1.15
        if fight_round and fight_round == 1:
            base = 1.4
        elif fight_round and fight_round == 2:
            base = 1.25
    elif "S-DEC" in method_upper or "SPLIT" in method_upper:
        base = 0.85
    elif "M-DEC" in method_upper or "MAJORITY" in method_upper:
        base = 0.9
    elif "U-DEC" in method_upper or "UNANIMOUS" in method_upper:
        base = 1.0
    elif "DQ" in method_upper or "DISQUALIFICATION" in method_upper:
        base = 0.8
    elif "OVERTURNED" in method_upper:
        base = 0.7
    else:
        base = 1.0

    # Dominance bonus from fight stats
    if stats:
        sig_landed = stats.get("significant_strikes_landed", 0) or 0
        sig_att = stats.get("significant_strikes_attempted", 0) or 0
        opp_sig_landed = stats.get("opp_significant_strikes_landed", 0) or 0

        # Strike differential bonus
        if sig_landed > 0 and opp_sig_landed >= 0:
            diff_ratio = (sig_landed - opp_sig_landed) / max(sig_landed, 1)
            if diff_ratio > 0.5:
                base += 0.1  # dominant striking performance

        # Takedown dominance
        td_landed = stats.get("takedowns_landed", 0) or 0
        if td_landed >= 5:
            base += 0.05

    return min(base, 1.5)  # cap at 1.5


# ---------------------------------------------------------------------------
# Glicko-2 math
# ---------------------------------------------------------------------------

@dataclass
class FighterRating:
    fighter_id: str
    name: str = ""
    # Glicko-1 scale (for display)
    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    volatility: float = DEFAULT_VOLATILITY
    # Track history
    last_fight_date: str | None = None
    division: str | None = None
    peak_rating: float = DEFAULT_RATING
    fights_count: int = 0
    division_counts: dict = field(default_factory=dict)  # weight_class -> count

    @property
    def mu(self) -> float:
        """Convert to Glicko-2 scale."""
        return (self.rating - 1500) / GLICKO2_SCALE

    @mu.setter
    def mu(self, value: float):
        self.rating = value * GLICKO2_SCALE + 1500

    @property
    def phi(self) -> float:
        """Convert RD to Glicko-2 scale."""
        return self.rd / GLICKO2_SCALE

    @phi.setter
    def phi(self, value: float):
        self.rd = value * GLICKO2_SCALE


def _g(phi: float) -> float:
    """Glicko-2 g function — reduces impact of opponent with high RD."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / (math.pi ** 2))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Expected score given ratings."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _compute_new_volatility(sigma: float, phi: float, delta: float, v: float) -> float:
    """
    Iterative algorithm (Illinois method) to find new volatility σ'.
    This is Step 5 of the Glicko-2 algorithm.
    """
    a = math.log(sigma ** 2)
    tau_sq = TAU ** 2

    def f(x):
        ex = math.exp(x)
        denom = 2.0 * (phi ** 2 + v + ex) ** 2
        return (ex * (delta ** 2 - phi ** 2 - v - ex)) / denom - (x - a) / tau_sq

    # Find bounds
    A = a
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU

    # Illinois algorithm
    fA = f(A)
    fB = f(B)
    for _ in range(100):
        if abs(B - A) < EPSILON:
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A = B
            fA = fB
        else:
            fA = fA / 2.0
        B = C
        fB = fC

    return math.exp(A / 2.0)


def update_rating(player: FighterRating, opponent: FighterRating,
                  score: float, multiplier: float = 1.0) -> None:
    """
    Update a single player's rating after one fight.

    score: 1.0 = win, 0.0 = loss, 0.5 = draw
    multiplier: performance multiplier that scales the update magnitude
    """
    mu = player.mu
    phi = player.phi
    sigma = player.volatility

    mu_j = opponent.mu
    phi_j = opponent.phi

    # Step 3: Compute variance (v) and improvement (delta)
    g_j = _g(phi_j)
    E_j = _E(mu, mu_j, phi_j)

    v = 1.0 / (g_j ** 2 * E_j * (1.0 - E_j))
    delta = v * g_j * (score - E_j) * multiplier

    # Step 5: New volatility
    new_sigma = _compute_new_volatility(sigma, phi, delta, v)

    # Step 6: Update phi with new volatility
    phi_star = math.sqrt(phi ** 2 + new_sigma ** 2)

    # Step 7: New phi and mu
    new_phi = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    new_mu = mu + new_phi ** 2 * g_j * (score - E_j) * multiplier

    player.mu = new_mu
    player.phi = new_phi
    player.volatility = new_sigma
    player.fights_count += 1
    player.peak_rating = max(player.peak_rating, player.rating)


def inflate_rd(player: FighterRating, days_inactive: int) -> None:
    """Increase RD for time between rating periods (inactivity)."""
    if days_inactive <= 0:
        return
    inflation = RD_INFLATION_PER_DAY * days_inactive
    new_rd = min(math.sqrt(player.rd ** 2 + inflation ** 2), DEFAULT_RD)
    player.rd = new_rd


# ---------------------------------------------------------------------------
# Engine: process all fights
# ---------------------------------------------------------------------------

def build_ratings() -> dict[str, FighterRating]:
    """
    Process all fights chronologically and compute Glicko-2 ratings.
    Returns dict of fighter_id -> FighterRating.
    """
    conn = get_connection()

    # Load all fights ordered by date
    fights = conn.execute("""
        SELECT f.fight_id, f.date, f.fighter_a_id, f.fighter_b_id,
               f.winner_id, f.method, f.round, f.weight_class
        FROM fights f
        WHERE f.date IS NOT NULL
          AND f.weight_class IN ('Flyweight','Bantamweight','Featherweight',
               'Lightweight','Welterweight','Middleweight',
               'Light Heavyweight','Heavyweight',
               'Open Weight','Catch Weight','Super Heavyweight')
        ORDER BY f.date ASC, f.fight_id ASC
    """).fetchall()

    # Load fighter names
    fighter_names = {}
    for fid, name in conn.execute("SELECT fighter_id, name FROM fighters"):
        fighter_names[fid] = name

    # Load division overrides
    division_overrides = {}
    for fid, div in conn.execute("SELECT fighter_id, division FROM division_overrides"):
        division_overrides[fid] = div

    # Pre-load all fight stats for performance multiplier
    stats_by_fight = {}
    for row in conn.execute("""
        SELECT fight_id, fighter_id, significant_strikes_landed,
               significant_strikes_attempted, takedowns_landed
        FROM fight_stats
    """):
        fid, fighter_id = row[0], row[1]
        if fid not in stats_by_fight:
            stats_by_fight[fid] = {}
        stats_by_fight[fid][fighter_id] = {
            "significant_strikes_landed": row[2],
            "significant_strikes_attempted": row[3],
            "takedowns_landed": row[4],
        }

    conn.close()

    ratings: dict[str, FighterRating] = {}

    def get_or_create(fighter_id: str) -> FighterRating:
        if fighter_id not in ratings:
            ratings[fighter_id] = FighterRating(
                fighter_id=fighter_id,
                name=fighter_names.get(fighter_id, "Unknown"),
            )
        return ratings[fighter_id]

    print(f"Processing {len(fights)} fights...")

    for fight_id, date, fa_id, fb_id, winner_id, method, rnd, weight_class in fights:
        fa = get_or_create(fa_id)
        fb = get_or_create(fb_id)

        # Inflate RD for inactivity
        if date:
            for fighter in (fa, fb):
                if fighter.last_fight_date:
                    days = (datetime.strptime(date, "%Y-%m-%d") -
                            datetime.strptime(fighter.last_fight_date, "%Y-%m-%d")).days
                    inflate_rd(fighter, days)
                fighter.last_fight_date = date

        # Track division: most recent + counts for fallback
        if weight_class in MENS_DIVISIONS:
            for fighter in (fa, fb):
                fighter.division = weight_class
                fighter.division_counts[weight_class] = fighter.division_counts.get(weight_class, 0) + 1

        # Determine scores
        if winner_id == fa_id:
            score_a, score_b = 1.0, 0.0
        elif winner_id == fb_id:
            score_a, score_b = 0.0, 1.0
        else:
            score_a, score_b = 0.5, 0.5  # draw/NC

        # Compute performance multiplier
        fight_stats = stats_by_fight.get(fight_id, {})
        stats_a = fight_stats.get(fa_id)
        stats_b = fight_stats.get(fb_id)

        # Add opponent stats for dominance calc
        if stats_a and stats_b:
            stats_a["opp_significant_strikes_landed"] = stats_b.get("significant_strikes_landed", 0)
            stats_b["opp_significant_strikes_landed"] = stats_a.get("significant_strikes_landed", 0)

        mult_a = performance_multiplier(method, rnd, stats_a) if score_a == 1.0 else performance_multiplier(method, rnd, stats_b)
        mult_b = performance_multiplier(method, rnd, stats_b) if score_b == 1.0 else performance_multiplier(method, rnd, stats_a)

        # For draws, use base multiplier
        if score_a == 0.5:
            mult_a = mult_b = 1.0

        # Update both fighters (must snapshot opponent before mutating)
        fa_snapshot = FighterRating(
            fighter_id=fa.fighter_id, rating=fa.rating,
            rd=fa.rd, volatility=fa.volatility,
        )
        fb_snapshot = FighterRating(
            fighter_id=fb.fighter_id, rating=fb.rating,
            rd=fb.rd, volatility=fb.volatility,
        )

        update_rating(fa, fb_snapshot, score_a, mult_a)
        update_rating(fb, fa_snapshot, score_b, mult_b)

    # For inactive fighters, use most common division (legacy assignment)
    today = datetime.now()
    for r in ratings.values():
        if r.last_fight_date and r.division_counts:
            days = (today - datetime.strptime(r.last_fight_date, "%Y-%m-%d")).days
            if days > ACTIVE_THRESHOLD_DAYS:
                r.division = max(r.division_counts, key=r.division_counts.get)

    # Apply manual division overrides (takes priority over everything)
    for fid, div in division_overrides.items():
        if fid in ratings:
            ratings[fid].division = div

    print(f"Ratings computed for {len(ratings)} fighters.")
    return ratings


# ---------------------------------------------------------------------------
# Rankings output
# ---------------------------------------------------------------------------

def generate_rankings(ratings: dict[str, FighterRating]) -> None:
    """Print current division rankings, P4P, and GOAT top 50."""
    today = datetime.now()

    def is_active(r: FighterRating) -> bool:
        if not r.last_fight_date:
            return False
        days = (today - datetime.strptime(r.last_fight_date, "%Y-%m-%d")).days
        return days <= ACTIVE_THRESHOLD_DAYS

    active = [r for r in ratings.values() if is_active(r)]

    # --- Division rankings ---
    print("\n" + "=" * 70)
    print("CURRENT DIVISION RANKINGS (Glicko-2)")
    print("=" * 70)

    for div in ["Flyweight", "Bantamweight", "Featherweight", "Lightweight",
                "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]:
        div_fighters = sorted(
            [r for r in active if r.division == div],
            key=lambda r: r.rating, reverse=True,
        )

        print(f"\n{'─' * 50}")
        print(f"  {div.upper()}")
        print(f"{'─' * 50}")

        if div_fighters:
            champ = div_fighters[0]
            print(f"  C  {champ.name:<30s}  {champ.rating:.0f} ± {champ.rd:.0f}")
            for i, f in enumerate(div_fighters[1:16], 1):
                print(f"  {i:>2d}  {f.name:<30s}  {f.rating:.0f} ± {f.rd:.0f}")

    # --- Pound for Pound ---
    print(f"\n{'=' * 70}")
    print("POUND-FOR-POUND TOP 15")
    print(f"{'=' * 70}")

    p4p = sorted(
        [r for r in active if r.division in MENS_DIVISIONS],
        key=lambda r: r.rating, reverse=True,
    )
    for i, f in enumerate(p4p[:15], 1):
        print(f"  {i:>2d}  {f.name:<30s}  {f.rating:.0f} ± {f.rd:.0f}  ({f.division})")

    # --- GOAT Top 50 ---
    print(f"\n{'=' * 70}")
    print("GREATEST OF ALL TIME — TOP 50 (by peak rating)")
    print(f"{'=' * 70}")

    goat = sorted(
        [r for r in ratings.values() if r.division in MENS_DIVISIONS and r.fights_count >= 3],
        key=lambda r: r.peak_rating, reverse=True,
    )
    for i, f in enumerate(goat[:50], 1):
        active_tag = " *" if is_active(f) else ""
        print(f"  {i:>2d}  {f.name:<30s}  peak {f.peak_rating:.0f}  (current {f.rating:.0f}, {f.fights_count} fights){active_tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ratings = build_ratings()
    generate_rankings(ratings)
