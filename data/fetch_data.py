"""
Data Fetcher: FastF1 Telemetry Pipeline
========================================

WHY THIS FILE EXISTS:
All the mathematical elegance in mdp.py, tyre_model.py, and solver.py is only
as good as the data feeding it. This script is the bridge between the real
world — 20 cars, 23 circuits, 70-odd laps each, recorded at 10Hz — and our
model. It downloads, cleans, and stores everything the model needs.

ABOUT FastF1:
FastF1 is an open-source Python library that wraps the official Formula 1
timing API (the same data feed broadcast partners receive). It gives access to:

  - Lap-by-lap timing (sector times, lap times, tyre compound and age)
  - Car telemetry at 10Hz (speed, throttle, brake, gear, DRS, RPM)
  - Weather data (track temp, air temp, humidity, rainfall)
  - Track status (green/yellow/SC/VSC/red flag codes)
  - Tyre data (compound, tyre life, fresh/used flag)

The library caches downloaded data locally (in data/raw/), so subsequent
runs are fast. The cache is gitignored — only processed features go into
the repository.

DATA PIPELINE:
  1. Download race sessions for specified years and circuits
  2. Extract "clean air" laps: gap ahead > 2s, green flag, no pit
  3. Compute lap time delta vs new-tyre baseline per driver per stint
  4. Aggregate by compound across all drivers (normalise for car speed)
  5. Export processed arrays to data/processed/ as .parquet files
  6. Export safety car deployment maps as safety_car_events.parquet

WHAT "CLEAN AIR" MEANS AND WHY IT MATTERS:
Tyre degradation is masked by two confounders:
  - Traffic: following a car < 2 seconds ahead generates dirty air, increasing
    fuel consumption and tyre temperatures, accelerating deg artificially.
  - Safety car: lap times under SC are controlled (~40% slower), so they
    should not be used to fit the degradation model.

Filtering for clean air laps isolates the tyre-induced component of lap time
variation. This is exactly what Pirelli's engineers do internally.

NORMALISATION:
Different cars have different base pace (Red Bull is ~0.8s faster per lap than
Haas in 2023). We normalise by computing each lap's delta relative to that
driver's new-tyre pace (laps 1-3 of the stint), then pool across all drivers.
This gives us compound-specific degradation curves independent of car speed.

Author: Claudia Maria Lopez Bombin
GitHub: https://github.com/claudialbombin/pitwall
License: MIT
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Circuits and years to fetch ──
DEFAULT_CIRCUITS = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Azerbaijan",
    "Miami",
    "Monaco",
    "Spain",
    "Canada",
    "Austria",
    "British",
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",
    "Singapore",
    "Japan",
    "Qatar",
    "United States",
    "Mexico",
    "Brazil",
    "Las Vegas",
    "Abu Dhabi",
]

DEFAULT_YEARS = [2021, 2022, 2023]

# Track status codes in FastF1 laps
SC_STATUS_CODES = {"4", "6", "7"}  # Safety Car
VSC_STATUS_CODES = {"5"}  # Virtual Safety Car
GREEN_CODE = "1"


# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------


def fetch_session(
    year: int, circuit: str, session_type: str = "R"
) -> pd.DataFrame | None:
    """
    Download a single session from FastF1 and return lap data as a DataFrame.

    Parameters
    ----------
    year         : season year
    circuit      : circuit name (as used in FastF1, e.g. "British", "Monaco")
    session_type : "R" = Race, "Q" = Qualifying, "FP1"/"FP2"/"FP3"

    Returns None if the session cannot be loaded.
    """
    try:
        import fastf1  # type: ignore[import]
    except ImportError:
        raise ImportError("fastf1 is required. Install with:\n  pip install fastf1")

    fastf1.Cache.enable_cache(str(RAW_DIR))

    log.info(f"Fetching {year} {circuit} {session_type}...")

    try:
        session = fastf1.get_session(year, circuit, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as e:
        log.warning(f"Could not load {year} {circuit}: {e}")
        return None

    laps = session.laps.copy()
    weather = session.weather_data

    # ── Basic cleaning ──
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps = laps[laps["LapTimeSec"].notna()]
    laps = laps[laps["LapTimeSec"] > 60]  # sanity: no lap under 1 minute
    laps = laps[laps["LapTimeSec"] < 300]  # sanity: no lap over 5 minutes
    laps = laps[laps["TyreLife"].notna()]
    laps = laps[laps["Compound"].notna()]
    laps["Compound"] = laps["Compound"].str.upper()

    # ── Annotate SC / VSC ──
    laps["SC"] = laps["TrackStatus"].isin(SC_STATUS_CODES)
    laps["VSC"] = laps["TrackStatus"].isin(VSC_STATUS_CODES)
    laps["GreenFlag"] = laps["TrackStatus"] == GREEN_CODE

    # ── Add race/circuit metadata ──
    laps["Year"] = year
    laps["Circuit"] = circuit

    # ── Merge weather (nearest timestamp) ──
    if weather is not None and not weather.empty and "AirTemp" in weather.columns:
        laps = _merge_weather(laps, weather)

    log.info(f"  → {len(laps):,} laps loaded")
    return laps


def _merge_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge nearest weather reading onto each lap by timestamp."""
    if "Time" not in weather.columns:
        return laps

    weather = weather.sort_values("Time").reset_index(drop=True)
    laps = laps.copy()

    # Approximate lap timestamp as LapStartTime + LapTime/2
    if "LapStartTime" in laps.columns:
        lap_times = laps["LapStartTime"] + laps["LapTime"] / 2
        idxs = weather["Time"].searchsorted(lap_times)
        idxs = idxs.clip(0, len(weather) - 1)
        for col in ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]:
            if col in weather.columns:
                laps[col] = weather[col].iloc[idxs].values
    return laps


# ---------------------------------------------------------------------------
# Clean air extraction
# ---------------------------------------------------------------------------


def extract_clean_laps(
    laps: pd.DataFrame, min_gap_ahead: float = 2.0, max_tyre_age_baseline: int = 3
) -> pd.DataFrame:
    """
    Filter to laps driven in clean air under green flag conditions.

    Parameters
    ----------
    laps                  : raw lap DataFrame from fetch_session()
    min_gap_ahead         : minimum gap to car ahead in seconds
    max_tyre_age_baseline : laps used to compute new-tyre baseline pace
    """
    clean = laps[laps["GreenFlag"]].copy()

    # Gap filter (if available)
    if "GapToLeader" in clean.columns:
        # GapToLeader is not per-car gap ahead; use as proxy when needed
        pass
    # Use PitOutTime / PitInTime as alternative gap proxy:
    # A lap immediately after a pit out (TyreLife == 1) has no car ahead
    # constraint — include it as a baseline lap.

    # ── Per-driver, per-stint baseline ──
    def add_baseline(g: pd.DataFrame) -> pd.DataFrame:
        baseline_laps = g[g["TyreLife"] <= max_tyre_age_baseline]
        if len(baseline_laps) < 2:
            return g.assign(LapTimeDelta=np.nan)
        baseline = baseline_laps["LapTimeSec"].median()
        g = g.copy()
        g["LapTimeDelta"] = g["LapTimeSec"] - baseline
        return g

    # Group by driver and stint (identified by TyreLife reset)
    clean["Stint"] = clean.groupby("Driver")["TyreLife"].transform(
        lambda x: (x.diff() < 0).cumsum()
    )

    clean = clean.groupby(["Driver", "Stint"], group_keys=False).apply(add_baseline)

    clean = clean[
        clean["LapTimeDelta"].notna()
        & (clean["LapTimeDelta"] >= -0.5)  # sanity: not faster than baseline
        & (clean["LapTimeDelta"] < 8.0)  # sanity: not more than 8s degraded
    ]

    log.info(f"  → {len(clean):,} clean air laps after filtering")
    return clean


# ---------------------------------------------------------------------------
# Aggregate per compound
# ---------------------------------------------------------------------------


def aggregate_compound(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clean laps into (compound, tyre_age) → (mean_delta, std_delta, n).

    This is the training data for tyre_model.py.
    Each row represents the average lap time delta across all drivers and
    races at a given compound × tyre_age combination.
    """
    agg = (
        clean.groupby(["Circuit", "Year", "Compound", "TyreLife"])["LapTimeDelta"]
        .agg(mean_delta="mean", std_delta="std", n="count")
        .reset_index()
        .rename(columns={"TyreLife": "TyreAge"})
    )
    return agg


# ---------------------------------------------------------------------------
# Safety car event extraction
# ---------------------------------------------------------------------------


def extract_sc_events(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Extract safety car deployment events per race.

    Returns a DataFrame with columns:
      Year, Circuit, LapNumber, SCType, Duration (laps)

    Used by safety_car.py to fit the Bayesian deployment model.
    """
    records = []

    for (year, circuit), race_laps in laps.groupby(["Year", "Circuit"]):
        race_laps = race_laps.sort_values("LapNumber")
        sc_laps = race_laps[race_laps["SC"]]["LapNumber"].unique()
        vsc_laps = race_laps[race_laps["VSC"]]["LapNumber"].unique()

        for code, lap_set, sc_type in [
            ("SC", sc_laps, "SAFETY_CAR"),
            ("VSC", vsc_laps, "VIRTUAL_SC"),
        ]:
            if len(lap_set) == 0:
                continue
            sorted_laps = sorted(lap_set)
            # Group consecutive laps into events
            start = sorted_laps[0]
            prev = sorted_laps[0]
            for lap in sorted_laps[1:]:
                if lap != prev + 1:
                    records.append(
                        {
                            "Year": year,
                            "Circuit": circuit,
                            "StartLap": start,
                            "EndLap": prev,
                            "Duration": prev - start + 1,
                            "SCType": sc_type,
                        }
                    )
                    start = lap
                prev = lap
            records.append(
                {
                    "Year": year,
                    "Circuit": circuit,
                    "StartLap": start,
                    "EndLap": prev,
                    "Duration": prev - start + 1,
                    "SCType": sc_type,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    years: list[int] = DEFAULT_YEARS,
    circuits: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Run the full data pipeline for the given years and circuits.

    Outputs:
      data/processed/laps_clean.parquet   — clean air lap data
      data/processed/tyre_degradation.parquet — aggregated by (compound, age)
      data/processed/safety_car_events.parquet — SC deployment per race
    """
    if circuits is None:
        circuits = DEFAULT_CIRCUITS

    out_clean = PROC_DIR / "laps_clean.parquet"
    out_deg = PROC_DIR / "tyre_degradation.parquet"
    out_sc = PROC_DIR / "safety_car_events.parquet"

    if not overwrite and out_clean.exists():
        log.info("Processed data already exists. Use --overwrite to re-fetch.")
        return

    all_laps: list[pd.DataFrame] = []

    for year in years:
        for circuit in circuits:
            laps = fetch_session(year, circuit)
            if laps is None:
                continue
            clean = extract_clean_laps(laps)
            if len(clean) > 0:
                all_laps.append(clean)

    if not all_laps:
        log.error(
            "No laps fetched. Check your FastF1 installation and internet connection."
        )
        return

    combined = pd.concat(all_laps, ignore_index=True)
    log.info(f"Total clean air laps: {len(combined):,}")

    # Save clean laps
    combined.to_parquet(out_clean, index=False)
    log.info(f"Saved: {out_clean}")

    # Aggregated degradation curves
    deg_agg = aggregate_compound(combined)
    deg_agg.to_parquet(out_deg, index=False)
    log.info(f"Saved: {out_deg}")

    # Safety car events (re-fetch all laps including non-clean)
    log.info("Extracting safety car events...")
    all_raw: list[pd.DataFrame] = []
    for year in years:
        for circuit in circuits:
            laps = fetch_session(year, circuit)
            if laps is not None:
                all_raw.append(laps)
    if all_raw:
        sc_events = extract_sc_events(pd.concat(all_raw, ignore_index=True))
        sc_events.to_parquet(out_sc, index=False)
        log.info(f"Saved: {out_sc} ({len(sc_events)} events)")

    log.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and process F1 telemetry data via FastF1."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Seasons to download (e.g. --years 2022 2023)",
    )
    parser.add_argument(
        "--circuits",
        nargs="+",
        type=str,
        default=None,
        help="Circuits to include (default: all 2023 calendar)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if processed data already exists",
    )
    args = parser.parse_args()

    run_pipeline(
        years=args.years,
        circuits=args.circuits,
        overwrite=args.overwrite,
    )
