"""UFCStats.com scraper with caching, rate limiting, and incremental updates."""

import hashlib
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests
from bs4 import BeautifulSoup

from db import get_connection, init_db

BASE_URL = "http://ufcstats.com"
CACHE_DIR = Path(__file__).parent / "cache"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
MIN_REQUEST_INTERVAL = 1.0  # seconds between requests

_last_request_time = 0.0


# ---------------------------------------------------------------------------
# HTTP + Caching
# ---------------------------------------------------------------------------

def _cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{h}.html"


def fetch(url: str, use_cache: bool = True) -> str:
    """Fetch a URL with local HTML caching and rate limiting."""
    global _last_request_time

    cp = _cache_path(url)
    if use_cache and cp.exists():
        return cp.read_text(encoding="utf-8")

    for attempt in range(2):
        elapsed = time.time() - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        resp = requests.get(url, headers=HEADERS, timeout=30)
        _last_request_time = time.time()

        if resp.status_code >= 500 and attempt == 0:
            time.sleep(3)  # brief backoff before retry
            continue
        resp.raise_for_status()
        break

    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(resp.text, encoding="utf-8")
    return resp.text


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_id(url: str) -> str:
    """Extract the hex ID from a ufcstats URL."""
    return url.strip().rstrip("/").split("/")[-1]


def _parse_of(text: str) -> tuple[int | None, int | None]:
    """Parse '36 of 55' into (36, 55). Returns (None, None) on failure."""
    text = text.strip()
    if " of " not in text:
        return None, None
    parts = text.split(" of ")
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        return None, None


def _parse_pct(text: str) -> float | None:
    text = text.strip().replace("%", "")
    if text in ("", "---"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_ctrl_time(text: str) -> int | None:
    """Parse '4:32' control time into seconds."""
    text = text.strip()
    if text in ("", "---"):
        return None
    parts = text.split(":")
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return None


def _parse_date(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Scrape: Events list
# ---------------------------------------------------------------------------

def scrape_event_list() -> pl.DataFrame:
    """Scrape all completed events from the events listing page."""
    html = fetch(f"{BASE_URL}/statistics/events/completed?page=all", use_cache=False)
    soup = BeautifulSoup(html, "lxml")

    rows = []
    for tr in soup.select("tr.b-statistics__table-row_type_first, tr.b-statistics__table-row"):
        link = tr.select_one("a.b-link")
        date_el = tr.select_one("span.b-statistics__date")
        cols = tr.select("td.b-statistics__table-col")
        if not link:
            continue
        location = cols[1].text.strip() if len(cols) > 1 else None
        rows.append({
            "event_id": _extract_id(link["href"]),
            "name": link.text.strip(),
            "date": _parse_date(date_el.text) if date_el else None,
            "location": location,
            "url": link["href"],
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scrape: Event details (fight list for one event)
# ---------------------------------------------------------------------------

def scrape_event_fights(event_url: str, event_id: str, event_date: str | None) -> list[dict]:
    """Parse fight rows from an event detail page. Returns list of fight dicts."""
    html = fetch(event_url)
    soup = BeautifulSoup(html, "lxml")

    fights = []
    for tr in soup.select("tr.b-fight-details__table-row.js-fight-details-click"):
        fight_url = tr.get("data-link", "")
        if not fight_url:
            continue
        fight_id = _extract_id(fight_url)

        # Fighters: two <p> tags in the fighter column
        fighter_col = tr.select("td")[1]
        fighter_links = fighter_col.select("a.b-link")
        if len(fighter_links) < 2:
            continue
        fighter_a_id = _extract_id(fighter_links[0]["href"])
        fighter_b_id = _extract_id(fighter_links[1]["href"])

        # Winner: check W/L column
        wl_col = tr.select("td")[0]
        win_flags = wl_col.select("i.b-flag__text")
        winner_id = None
        if win_flags:
            first_flag = win_flags[0].text.strip().lower()
            if first_flag == "win":
                winner_id = fighter_a_id
            elif len(win_flags) > 1 and win_flags[1].text.strip().lower() == "win":
                winner_id = fighter_b_id
            # draw/NC → winner_id stays None

        # Weight class column
        cols = tr.select("td")
        weight_class = None
        if len(cols) > 6:
            weight_class = cols[6].text.strip() or None

        # Method, Round, Time — method cell contains method + details on separate lines
        method = None
        if len(cols) > 7:
            method_ps = cols[7].select("p")
            if method_ps:
                method = method_ps[0].text.strip() or None
            else:
                method = cols[7].text.strip().split("\n")[0].strip() or None
        fight_round = None
        if len(cols) > 8:
            try:
                fight_round = int(cols[8].text.strip())
            except ValueError:
                pass
        fight_time = cols[9].text.strip() or None if len(cols) > 9 else None

        # Title fight heuristic: weight class text contains "Title"
        title_fight = False
        if weight_class and "title" in weight_class.lower():
            title_fight = True
            weight_class = weight_class.replace("Title Bout", "").replace("title bout", "").strip()

        fights.append({
            "fight_id": fight_id,
            "fight_url": fight_url,
            "event_id": event_id,
            "date": event_date,
            "fighter_a_id": fighter_a_id,
            "fighter_b_id": fighter_b_id,
            "winner_id": winner_id,
            "method": method,
            "round": fight_round,
            "time": fight_time,
            "title_fight": 1 if title_fight else 0,
            "weight_class": weight_class,
        })

    return fights


# ---------------------------------------------------------------------------
# Scrape: Fight details (per-fight stats)
# ---------------------------------------------------------------------------

def scrape_fight_stats(fight_url: str, fight_id: str) -> list[dict]:
    """Parse totals stats table from a fight detail page."""
    html = fetch(fight_url)
    soup = BeautifulSoup(html, "lxml")

    # Find the totals table — it's the first table with the stats headers
    tables = soup.select("table")
    totals_table = None
    for t in tables:
        headers = [th.text.strip() for th in t.select("th")]
        if "KD" in headers and "Sig. str." in headers:
            totals_table = t
            break

    if totals_table is None:
        return []

    # Each row has two <p> tags per cell — one per fighter
    row = totals_table.select_one("tbody tr")
    if not row:
        return []

    cells = row.select("td")
    if len(cells) < 10:
        return []

    # Fighter links in first cell
    fighter_links = cells[0].select("a.b-link")
    if len(fighter_links) < 2:
        return []

    stats = []
    for idx in range(2):
        fid = _extract_id(fighter_links[idx]["href"])

        def cell_text(col_idx: int) -> str:
            ps = cells[col_idx].select("p")
            return ps[idx].text.strip() if idx < len(ps) else ""

        kd_text = cell_text(1)
        sig_str_text = cell_text(2)
        sig_pct_text = cell_text(3)
        total_str_text = cell_text(4)
        td_text = cell_text(5)
        td_pct_text = cell_text(6)
        sub_text = cell_text(7)
        rev_text = cell_text(8)
        ctrl_text = cell_text(9)

        sig_landed, sig_att = _parse_of(sig_str_text)
        total_landed, total_att = _parse_of(total_str_text)
        td_landed, td_att = _parse_of(td_text)

        stats.append({
            "fight_id": fight_id,
            "fighter_id": fid,
            "knockdowns": int(kd_text) if kd_text.isdigit() else None,
            "significant_strikes_landed": sig_landed,
            "significant_strikes_attempted": sig_att,
            "significant_strike_pct": _parse_pct(sig_pct_text),
            "total_strikes_landed": total_landed,
            "total_strikes_attempted": total_att,
            "takedowns_landed": td_landed,
            "takedowns_attempted": td_att,
            "takedown_pct": _parse_pct(td_pct_text),
            "submissions_attempted": int(sub_text) if sub_text.isdigit() else None,
            "reversals": int(rev_text) if rev_text.isdigit() else None,
            "control_time_seconds": _parse_ctrl_time(ctrl_text),
        })

    return stats


# ---------------------------------------------------------------------------
# Scrape: Fighter details
# ---------------------------------------------------------------------------

def scrape_fighter(fighter_url: str, fighter_id: str) -> dict:
    """Parse fighter bio from their detail page."""
    html = fetch(fighter_url)
    soup = BeautifulSoup(html, "lxml")

    name_el = soup.select_one("span.b-content__title-highlight")
    nick_el = soup.select_one("p.b-content__Nickname")
    record_el = soup.select_one("span.b-content__title-record")

    info = {}
    for li in soup.select("li.b-list__box-list-item"):
        label_el = li.select_one("i.b-list__box-item-title")
        if not label_el:
            continue
        label = label_el.text.strip().rstrip(":").upper()
        value = li.text.replace(label_el.text, "").strip()
        if value in ("", "--"):
            value = None
        info[label] = value

    record_text = None
    if record_el:
        record_text = record_el.text.strip().replace("Record:", "").strip()

    return {
        "fighter_id": fighter_id,
        "name": name_el.text.strip() if name_el else None,
        "nickname": nick_el.text.strip() if nick_el else None,
        "weight_class": info.get("WEIGHT CLASS") or info.get("WEIGHT"),
        "height": info.get("HEIGHT"),
        "reach": info.get("REACH"),
        "stance": info.get("STANCE"),
        "dob": _parse_date(info["DOB"]) if info.get("DOB") else None,
        "record": record_text,
    }


# ---------------------------------------------------------------------------
# Database insert helpers
# ---------------------------------------------------------------------------

def _upsert_event(cur: sqlite3.Cursor, event: dict) -> None:
    cur.execute(
        "INSERT OR REPLACE INTO events (event_id, name, date, location) VALUES (?, ?, ?, ?)",
        (event["event_id"], event["name"], event["date"], event["location"]),
    )


def _upsert_fighter(cur: sqlite3.Cursor, fighter: dict) -> None:
    cur.execute(
        """INSERT OR REPLACE INTO fighters
           (fighter_id, name, nickname, weight_class, height, reach, stance, dob, record)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            fighter["fighter_id"], fighter["name"], fighter["nickname"],
            fighter["weight_class"], fighter["height"], fighter["reach"],
            fighter["stance"], fighter["dob"], fighter["record"],
        ),
    )


def _upsert_fight(cur: sqlite3.Cursor, fight: dict) -> None:
    cur.execute(
        """INSERT OR REPLACE INTO fights
           (fight_id, event_id, date, fighter_a_id, fighter_b_id, winner_id,
            method, round, time, title_fight, weight_class)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            fight["fight_id"], fight["event_id"], fight["date"],
            fight["fighter_a_id"], fight["fighter_b_id"], fight["winner_id"],
            fight["method"], fight["round"], fight["time"],
            fight["title_fight"], fight["weight_class"],
        ),
    )


def _upsert_fight_stats(cur: sqlite3.Cursor, stat: dict) -> None:
    cur.execute(
        """INSERT OR REPLACE INTO fight_stats
           (fight_id, fighter_id, knockdowns,
            significant_strikes_landed, significant_strikes_attempted, significant_strike_pct,
            total_strikes_landed, total_strikes_attempted,
            takedowns_landed, takedowns_attempted, takedown_pct,
            submissions_attempted, reversals, control_time_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            stat["fight_id"], stat["fighter_id"], stat["knockdowns"],
            stat["significant_strikes_landed"], stat["significant_strikes_attempted"],
            stat["significant_strike_pct"],
            stat["total_strikes_landed"], stat["total_strikes_attempted"],
            stat["takedowns_landed"], stat["takedowns_attempted"], stat["takedown_pct"],
            stat["submissions_attempted"], stat["reversals"], stat["control_time_seconds"],
        ),
    )


# ---------------------------------------------------------------------------
# Main scrape pipeline
# ---------------------------------------------------------------------------

def run_scrape(force: bool = False) -> None:
    """
    Full scrape pipeline:
    1. Fetch all events
    2. For each un-scraped event, fetch fights + stats + fighters
    3. Store everything in SQLite
    """
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    print("=" * 60)
    print("P4P Data Scraper — UFCStats.com")
    print("=" * 60)

    # Step 1: Get all events
    print("\n[1/3] Fetching event list...")
    events_df = scrape_event_list()
    total_events = len(events_df)
    print(f"  Found {total_events} events")

    # Determine which events are already scraped
    already_scraped = set()
    if not force:
        cur.execute("SELECT event_id FROM scrape_log")
        already_scraped = {row[0] for row in cur.fetchall()}
        print(f"  Already scraped: {len(already_scraped)} events")

    events_to_scrape = events_df.filter(~pl.col("event_id").is_in(list(already_scraped)))
    print(f"  Events to scrape: {len(events_to_scrape)}")

    # Insert all events into DB
    for row in events_df.iter_rows(named=True):
        _upsert_event(cur, row)
    conn.commit()

    # Step 2: Scrape each event
    print("\n[2/3] Scraping event details, fights, and stats...")
    total_fights_added = 0
    total_stats_added = 0
    fighters_seen = set()
    errors = []

    for i, event in enumerate(events_to_scrape.iter_rows(named=True)):
        eid = event["event_id"]
        ename = event["name"]
        edate = event["date"]
        eurl = event["url"]

        pct = (i + 1) / len(events_to_scrape) * 100
        print(f"  [{i+1}/{len(events_to_scrape)}] ({pct:.1f}%) {ename} ({edate or '?'})", end="", flush=True)

        try:
            fights = scrape_event_fights(eurl, eid, edate)
        except Exception as e:
            errors.append(f"Event {eid} fights: {e}")
            print(f" — ERROR: {e}")
            continue

        event_fights = 0
        event_stats = 0

        for fight in fights:
            # Scrape fighter details first (FK constraint requires fighters exist)
            for fid_key in ("fighter_a_id", "fighter_b_id"):
                fid = fight[fid_key]
                if fid not in fighters_seen:
                    fighters_seen.add(fid)
                    try:
                        fdata = scrape_fighter(f"{BASE_URL}/fighter-details/{fid}", fid)
                        _upsert_fighter(cur, fdata)
                    except Exception as e:
                        errors.append(f"Fighter {fid}: {e}")
                        # Insert stub so FK constraint doesn't block the fight
                        cur.execute(
                            "INSERT OR IGNORE INTO fighters (fighter_id, name) VALUES (?, ?)",
                            (fid, f"Unknown ({fid[:8]})"),
                        )

            _upsert_fight(cur, fight)
            event_fights += 1

            # Scrape fight stats
            try:
                stats = scrape_fight_stats(fight["fight_url"], fight["fight_id"])
                for s in stats:
                    _upsert_fight_stats(cur, s)
                    event_stats += 1
            except Exception as e:
                errors.append(f"Fight stats {fight['fight_id']}: {e}")

        total_fights_added += event_fights
        total_stats_added += event_stats

        # Mark event as scraped
        cur.execute(
            "INSERT OR REPLACE INTO scrape_log (event_id, scraped_at) VALUES (?, ?)",
            (eid, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        print(f" — {event_fights} fights, {event_stats} stat rows")

    print(f"\n  Total: {total_fights_added} fights, {total_stats_added} stat rows, {len(fighters_seen)} fighters")

    if errors:
        print(f"\n  ⚠ {len(errors)} errors encountered:")
        for e in errors[:20]:
            print(f"    - {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")

    conn.close()

    # Step 3: Validate
    print("\n[3/3] Running validation...")
    validate()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate() -> None:
    """Check data completeness and flag anomalies."""
    conn = get_connection()

    # Basic counts
    counts = {}
    for table in ("events", "fighters", "fights", "fight_stats"):
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = row[0]
        print(f"  {table}: {row[0]:,} rows")

    # Date range
    row = conn.execute("SELECT MIN(date), MAX(date) FROM events WHERE date IS NOT NULL").fetchone()
    print(f"  Date range: {row[0]} to {row[1]}")

    # Fights without stats
    row = conn.execute("""
        SELECT COUNT(*) FROM fights f
        WHERE NOT EXISTS (SELECT 1 FROM fight_stats fs WHERE fs.fight_id = f.fight_id)
    """).fetchone()
    fights_no_stats = row[0]
    total_fights = counts["fights"]
    stats_pct = ((total_fights - fights_no_stats) / total_fights * 100) if total_fights else 0
    print(f"  Fights with stats: {total_fights - fights_no_stats}/{total_fights} ({stats_pct:.1f}%)")

    # Per-column completeness in fight_stats
    print("\n  fight_stats column completeness:")
    stat_cols = [
        "knockdowns", "significant_strikes_landed", "significant_strikes_attempted",
        "total_strikes_landed", "total_strikes_attempted",
        "takedowns_landed", "takedowns_attempted",
        "submissions_attempted", "control_time_seconds",
    ]
    total_stat_rows = counts["fight_stats"]
    if total_stat_rows > 0:
        for col in stat_cols:
            row = conn.execute(f"SELECT COUNT(*) FROM fight_stats WHERE {col} IS NOT NULL").fetchone()
            pct = row[0] / total_stat_rows * 100
            print(f"    {col}: {pct:.1f}%")

    # Fighter completeness
    print("\n  fighter column completeness:")
    fighter_cols = ["height", "reach", "stance", "dob", "record"]
    total_fighters = counts["fighters"]
    if total_fighters > 0:
        for col in fighter_cols:
            row = conn.execute(f"SELECT COUNT(*) FROM fighters WHERE {col} IS NOT NULL AND {col} != ''").fetchone()
            pct = row[0] / total_fighters * 100
            print(f"    {col}: {pct:.1f}%")

    # Anomaly checks
    print("\n  Anomaly checks:")

    # Fights where winner is not one of the two fighters
    row = conn.execute("""
        SELECT COUNT(*) FROM fights
        WHERE winner_id IS NOT NULL
        AND winner_id != fighter_a_id
        AND winner_id != fighter_b_id
    """).fetchone()
    print(f"    Fights with invalid winner_id: {row[0]}")

    # Fights with no winner (draws/NC)
    row = conn.execute("SELECT COUNT(*) FROM fights WHERE winner_id IS NULL").fetchone()
    print(f"    Fights with no winner (draw/NC/ongoing): {row[0]}")

    # Duplicate fights
    row = conn.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT fight_id) FROM fights
    """).fetchone()
    print(f"    Duplicate fight_ids: {row[0]}")

    conn.close()
    print("\n" + "=" * 60)
    print("Scrape complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_scrape()
