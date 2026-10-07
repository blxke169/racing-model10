from __future__ import annotations
import io, re, requests, pandas as pd
from datetime import date, timedelta
from typing import Optional

BETFAIR_PROMO = "https://promo.betfair.com/betfairsp/prices"
FORMFAV_BASE = "https://api.formfav.com"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def betfair_filename(day: date, country: str = "aus", market: str = "win") -> str:
    # Betfair SP promo files follow dwbfprices{country}{market}DDMMYYYY.csv
    return f"dwbfprices{country.lower()}{market.lower()}{day:%d%m%Y}.csv"


def fetch_betfair_bsp_day(day: date, country: str = "aus", timeout: int = 25) -> pd.DataFrame:
    """Download a public Betfair Starting Price daily CSV.

    Betfair occasionally changes availability/naming. Fail loudly so the UI can
    show the exact date that could not be retrieved rather than silently faking data.
    """
    fname = betfair_filename(day, country=country)
    url = f"{BETFAIR_PROMO}/{fname}"
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if not r.content or b"<html" in r.content[:500].lower():
        raise RuntimeError(f"Betfair returned non-CSV content for {url}")
    df = pd.read_csv(io.BytesIO(r.content))
    df["source_url"] = url
    return df


def fetch_betfair_range(start: date, end: date, country: str = "aus") -> tuple[pd.DataFrame, list[str]]:
    frames, errors = [], []
    d = start
    while d <= end:
        try:
            x = fetch_betfair_bsp_day(d, country=country)
            x["file_date"] = d.isoformat()
            frames.append(x)
        except Exception as e:
            errors.append(f"{d.isoformat()}: {e}")
        d += timedelta(days=1)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), errors


def formfav_headers(api_key: str) -> dict:
    return {"X-API-Key": api_key, "Accept": "application/json"}


def fetch_formfav_meetings(api_key: str, race_date: str, country: str = "au", timezone: str = "Australia/Melbourne") -> dict:
    params = {"date": race_date, "race_code": "gallops", "country": country, "timezone": timezone}
    r = requests.get(f"{FORMFAV_BASE}/v1/meetings", params=params, headers=formfav_headers(api_key), timeout=25)
    r.raise_for_status()
    return r.json()


def fetch_formfav_form(api_key: str, race_date: str, track: str, race: int, country: str = "au", timezone: str = "Australia/Melbourne") -> dict:
    params = {"date": race_date, "track": track, "race": race, "country": country, "timezone": timezone}
    r = requests.get(f"{FORMFAV_BASE}/v1/form", params=params, headers=formfav_headers(api_key), timeout=25)
    r.raise_for_status()
    return r.json()


def flatten_formfav_race(payload: dict) -> pd.DataFrame:
    race_meta = {k: payload.get(k) for k in ["date","track","raceNumber","raceName","distance","condition","weather","raceClass","startTime","timezone"]}
    rows = []
    for runner in payload.get("runners", []):
        row = dict(race_meta)
        row.update(runner)
        # Flatten common nested stat blocks while retaining raw JSON-compatible values elsewhere.
        for block_name in ["careerStats","stats","career","trackStats","distanceStats","conditionStats"]:
            block = runner.get(block_name)
            if isinstance(block, dict):
                for k, v in block.items():
                    if not isinstance(v, (dict, list)):
                        row[f"{block_name}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def standardize_betfair(df: pd.DataFrame) -> pd.DataFrame:
    """Map Betfair promo variants into a stable modelling schema."""
    if df.empty:
        return df.copy()
    cols = {_norm(c): c for c in df.columns}
    aliases = {
        "event_id": ["eventid","event_id"],
        "event_name": ["eventname","event_name","event"],
        "market_id": ["marketid","market_id"],
        "market_name": ["marketname","market_name","market"],
        "selection_id": ["selectionid","selection_id"],
        "runner": ["selectionname","runnername","runner","selection"],
        "bsp": ["bsp","actualsp","actual_sp"],
        "ppwap": ["ppwap","weightedaverageprice","wap"],
        "morning_wap": ["morningwap","morning_wap"],
        "winner": ["winner","win","status"],
        "event_date": ["eventdate","event_date","date"],
    }
    out = pd.DataFrame(index=df.index)
    for target, opts in aliases.items():
        src = next((cols.get(_norm(o)) for o in opts if _norm(o) in cols), None)
        if src is not None:
            out[target] = df[src]
    if "runner" not in out:
        raise ValueError(f"Could not identify runner name. Columns: {list(df.columns)}")
    for c in ["bsp","ppwap","morning_wap"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "winner" in out:
        vals = out["winner"].astype(str).str.lower().str.strip()
        out["winner"] = vals.isin(["1","true","winner","win","w"]).astype(int)
    elif "bsp" in out:
        out["winner"] = 0
    return out
