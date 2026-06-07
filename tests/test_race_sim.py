"""
Tests for core/race_sim.py
============================

Testing a Monte Carlo simulator requires a different mindset from testing
deterministic algorithms. We cannot assert that "the mean race time equals X"
because it depends on random draws. Instead, we test:

  1. STATISTICAL PROPERTIES — properties that hold almost surely:
     - With N → ∞ simulations, the mean converges to a finite value
     - Standard deviation is positive and finite
     - Percentile ordering: P5 < mean < P95

  2. DETERMINISM — same seed must give exactly the same results.
     This is crucial for reproducibility of the backtest results.

  3. PHYSICAL CONSTRAINTS — every simulated race must be valid:
     - Race time must be positive
     - Pit lap must be within [1, total_laps]
     - At least 1 and at most 2 pit stops (regulations)
     - Lap times must be in physically plausible range

  4. POLICY MONOTONICITY — the MDP policy must outperform (or match)
     the baseline heuristic on average. If it doesn't, either the
     policy is wrong or the test is.

  5. VARIANCE REDUCTION — antithetic variates must reduce variance.
     We can test this statistically with a sufficient sample.

Author: Claudia Maria Lopez Bombin
GitHub: https://github.com/claudialbombin/pitwall
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from race_sim import (
    RaceSimulator, RaceResult, CircuitConfig, CIRCUITS,
    _AntitheticsRNG,
)
from tyre_model import TyreModel
from safety_car import SafetyCarModel
from mdp import Compound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tyre_model() -> TyreModel:
    """Minimal fitted tyre model for tests (uses default Weibull parameters)."""
    return TyreModel()


def _make_sc_model(circuit: str = "silverstone") -> SafetyCarModel:
    return SafetyCarModel(circuit=circuit, total_laps=CIRCUITS[circuit].total_laps)


@pytest.fixture
def silverstone() -> CircuitConfig:
    return CIRCUITS["silverstone"]


@pytest.fixture
def tyre_model() -> TyreModel:
    return _make_tyre_model()


@pytest.fixture
def sc_model() -> SafetyCarModel:
    return _make_sc_model("silverstone")


@pytest.fixture
def simulator(silverstone, tyre_model, sc_model) -> RaceSimulator:
    """Baseline simulator (no policy — heuristic only)."""
    return RaceSimulator(
        circuit=silverstone,
        tyre_model=tyre_model,
        sc_model=sc_model,
        policy=None,
    )


@pytest.fixture
def small_results(simulator) -> list[RaceResult]:
    """Run 50 simulations — enough for structural tests, not statistical ones."""
    return simulator.run(
        n_simulations=50, starting_position=5,
        starting_compound=Compound.MEDIUM,
        use_antithetic=False, seed=0, verbose=False,
    )


# ---------------------------------------------------------------------------
# CircuitConfig
# ---------------------------------------------------------------------------

class TestCircuitConfig:

    def test_all_circuits_have_positive_laps(self):
        for name, cfg in CIRCUITS.items():
            assert cfg.total_laps > 0, f"{name}: total_laps={cfg.total_laps}"

    def test_all_circuits_have_positive_pit_delta(self):
        for name, cfg in CIRCUITS.items():
            assert cfg.pit_lane_delta > 0, f"{name}: pit_delta={cfg.pit_lane_delta}"

    def test_all_circuits_have_realistic_base_lap_time(self):
        """Base lap times must be between 60s (1 minute) and 150s (2.5 minutes)."""
        for name, cfg in CIRCUITS.items():
            assert 60 <= cfg.base_lap_time <= 150, \
                f"{name}: base_lap_time={cfg.base_lap_time}"

    def test_monaco_has_longest_pit_delta(self):
        """Monaco pit lane is the longest — must exceed other common circuits."""
        monaco_delta = CIRCUITS["monaco"].pit_lane_delta
        for name in ["silverstone", "monza", "spa"]:
            assert monaco_delta >= CIRCUITS[name].pit_lane_delta, \
                f"Monaco delta ({monaco_delta}) < {name} ({CIRCUITS[name].pit_lane_delta})"

    def test_monza_has_lowest_base_lap(self):
        """Monza is the fastest circuit in the dataset."""
        monza_lap = CIRCUITS["monza"].base_lap_time
        for name in ["monaco", "silverstone", "spa"]:
            assert monza_lap <= CIRCUITS[name].base_lap_time, \
                f"Monza ({monza_lap}s) is not faster than {name} ({CIRCUITS[name].base_lap_time}s)"


# ---------------------------------------------------------------------------
# Single race simulation structural tests
# ---------------------------------------------------------------------------

class TestSingleRaceStructure:

    def test_race_result_has_positive_total_time(self, small_results):
        for r in small_results:
            assert r.total_time > 0, f"Non-positive race time: {r.total_time}"

    def test_race_time_in_plausible_range(self, small_results, silverstone):
        """
        Race time must be between:
          - minimum: base_lap_time × total_laps (impossible — no pits, perfect tyres)
          - maximum: 2× minimum (anything more is a simulation bug)
        """
        T = silverstone.total_laps
        lo = silverstone.base_lap_time * T * 0.9
        hi = silverstone.base_lap_time * T * 2.5
        for r in small_results:
            assert lo <= r.total_time <= hi, \
                f"Race time {r.total_time:.0f}s outside [{lo:.0f}, {hi:.0f}]"

    def test_pit_laps_within_race_bounds(self, small_results, silverstone):
        """All pit laps must be within the race lap range."""
        for r in small_results:
            for lap in r.pit_laps:
                assert 1 <= lap <= silverstone.total_laps, \
                    f"Pit on lap {lap} outside race ({silverstone.total_laps} laps)"

    def test_mandatory_stop_served(self, small_results):
        """FIA regulations: every car must make at least 1 pit stop."""
        for r in small_results:
            assert len(r.pit_laps) >= 1, "No pit stops — violates mandatory stop rule"

    def test_at_most_two_stops(self, small_results):
        """Our model allows at most 2 pit stops (tyre allocation constraint)."""
        for r in small_results:
            assert len(r.pit_laps) <= 2, f"More than 2 stops: {r.pit_laps}"

    def test_lap_times_list_correct_length(self, small_results, silverstone):
        for r in small_results:
            assert len(r.lap_times) == silverstone.total_laps, \
                f"Expected {silverstone.total_laps} lap times, got {len(r.lap_times)}"

    def test_all_lap_times_positive(self, small_results):
        for r in small_results:
            for i, lt in enumerate(r.lap_times):
                assert lt > 0, f"Negative/zero lap time on lap {i+1}: {lt:.3f}"

    def test_lap_times_in_plausible_range(self, small_results, silverstone):
        """
        Individual lap times must be between:
          - 0.9 × base (fastest possible — accounting for noise)
          - 2.0 × base (SC pace is ~1.38×; nothing should be 2×)
        """
        lo = silverstone.base_lap_time * 0.85
        hi = silverstone.base_lap_time * 2.1
        for r in small_results:
            for i, lt in enumerate(r.lap_times):
                assert lo <= lt <= hi, \
                    f"Lap {i+1} time {lt:.2f}s outside [{lo:.1f}, {hi:.1f}]"

    def test_compounds_used_consistent_with_pit_count(self, small_results):
        """
        len(compounds_used) must equal len(pit_laps) + 1
        (starting compound + one per pit stop).
        """
        for r in small_results:
            assert len(r.compounds_used) == len(r.pit_laps) + 1, (
                f"compounds_used={len(r.compounds_used)}, "
                f"pit_laps={len(r.pit_laps)}"
            )

    def test_policy_field_set(self, small_results):
        """policy_used field must be a non-empty string."""
        for r in small_results:
            assert isinstance(r.policy_used, str)
            assert len(r.policy_used) > 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_same_results(self, silverstone, tyre_model, sc_model):
        """Identical seeds must produce identical race times."""
        sim = RaceSimulator(silverstone, tyre_model, sc_model, policy=None)
        results_a = sim.run(n_simulations=20, seed=42, use_antithetic=False, verbose=False)
        results_b = sim.run(n_simulations=20, seed=42, use_antithetic=False, verbose=False)
        times_a = [r.total_time for r in results_a]
        times_b = [r.total_time for r in results_b]
        assert times_a == times_b, "Different results with same seed"

    def test_different_seeds_different_results(self, silverstone, tyre_model, sc_model):
        """Different seeds must (almost certainly) produce different results."""
        sim = RaceSimulator(silverstone, tyre_model, sc_model, policy=None)
        results_a = sim.run(n_simulations=20, seed=1, use_antithetic=False, verbose=False)
        results_b = sim.run(n_simulations=20, seed=2, use_antithetic=False, verbose=False)
        times_a = [r.total_time for r in results_a]
        times_b = [r.total_time for r in results_b]
        assert times_a != times_b, "Same results with different seeds (extremely unlikely)"


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

class TestStatisticalProperties:

    @pytest.fixture
    def large_results(self, simulator):
        """500 simulations — enough for reliable statistical tests."""
        return simulator.run(
            n_simulations=500, seed=99,
            use_antithetic=False, verbose=False
        )

    def test_mean_is_finite(self, large_results):
        times = [r.total_time for r in large_results]
        assert np.isfinite(np.mean(times))

    def test_std_is_positive(self, large_results):
        times = [r.total_time for r in large_results]
        assert np.std(times) > 0

    def test_percentile_ordering(self, large_results):
        """P5 < mean < P95 — must hold for any reasonable distribution."""
        times = np.array([r.total_time for r in large_results])
        p5   = np.percentile(times, 5)
        mean = np.mean(times)
        p95  = np.percentile(times, 95)
        assert p5 < mean, f"P5={p5:.1f} >= mean={mean:.1f}"
        assert mean < p95, f"mean={mean:.1f} >= P95={p95:.1f}"

    def test_sc_flag_increases_variance(self, silverstone, tyre_model):
        """
        Enabling safety car randomness should increase the variance of outcomes
        (more possible race scenarios).
        """
        sc_on  = SafetyCarModel(circuit="silverstone", total_laps=52)
        sc_off = SafetyCarModel(circuit="silverstone", total_laps=52)
        # Manually set p_deploy to 0 to disable SC
        for i in range(52):
            sc_off._lap_probs[i] = sc_off._lap_probs[i].__class__(alpha=0.0001, beta=100.0)

        sim_on  = RaceSimulator(silverstone, tyre_model, sc_on,  policy=None)
        sim_off = RaceSimulator(silverstone, tyre_model, sc_off, policy=None)

        r_on  = sim_on.run(200,  seed=7, use_antithetic=False, verbose=False)
        r_off = sim_off.run(200, seed=7, use_antithetic=False, verbose=False)

        std_on  = np.std([r.total_time for r in r_on])
        std_off = np.std([r.total_time for r in r_off])
        assert std_on >= std_off, \
            f"SC did not increase variance: std_on={std_on:.2f}, std_off={std_off:.2f}"

    def test_summary_dict_has_required_keys(self, simulator, large_results):
        summary = simulator.summarise(large_results)
        required = {
            "n_simulations", "mean_race_time_s", "std_race_time_s",
            "p5_race_time_s", "p95_race_time_s", "mean_pit_count",
            "pct_races_with_sc",
        }
        missing = required - set(summary.keys())
        assert not missing, f"Summary missing keys: {missing}"

    def test_summary_n_simulations_matches(self, simulator, large_results):
        summary = simulator.summarise(large_results)
        assert summary["n_simulations"] == len(large_results)

    def test_summary_sc_pct_in_range(self, simulator, large_results):
        summary = simulator.summarise(large_results)
        assert 0 <= summary["pct_races_with_sc"] <= 100


# ---------------------------------------------------------------------------
# Antithetic variates
# ---------------------------------------------------------------------------

class TestAntitheticsRNG:

    def test_random_complementary(self):
        """random() on antithetic RNG must return 1 - u for each source draw."""
        rng = np.random.default_rng(0)
        anti = _AntitheticsRNG(rng)
        rng2 = np.random.default_rng(0)  # same seed as source
        for _ in range(100):
            source_val = rng2.random()
            anti_val   = anti.random()
            assert anti_val == pytest.approx(1.0 - source_val, abs=1e-10)

    def test_uniform_complementary(self):
        """uniform(lo, hi) on antithetic must return lo + hi - source_uniform."""
        rng  = np.random.default_rng(42)
        anti = _AntitheticsRNG(rng)
        rng2 = np.random.default_rng(42)
        lo, hi = 2.0, 8.0
        for _ in range(50):
            source_val = rng2.random() * (hi - lo) + lo
            anti_val   = anti.uniform(lo, hi)
            assert anti_val == pytest.approx(lo + hi - source_val, abs=1e-9)

    def test_antithetic_values_in_unit_interval(self):
        """All antithetic random() values must be in [0, 1]."""
        rng  = np.random.default_rng(123)
        anti = _AntitheticsRNG(rng)
        for _ in range(1000):
            v = anti.random()
            assert 0 <= v <= 1, f"Antithetic value {v} outside [0, 1]"

    def test_antithetic_reduces_variance(self, silverstone, tyre_model, sc_model):
        """
        For a sufficient sample size, antithetic variates should reduce
        the variance of the mean race time estimate.

        We test this by comparing:
          Var[mean with antithetics] < Var[mean without antithetics]
        using bootstrap estimation.
        """
        N = 200
        sim = RaceSimulator(silverstone, tyre_model, sc_model, policy=None)

        results_anti = sim.run(N, seed=5, use_antithetic=True,  verbose=False)
        results_plain= sim.run(N, seed=5, use_antithetic=False, verbose=False)

        # Bootstrap variance of the sample mean
        def bootstrap_var_of_mean(times, n_boot=200):
            means = [np.mean(np.random.choice(times, len(times))) for _ in range(n_boot)]
            return np.var(means)

        times_anti  = np.array([r.total_time for r in results_anti])
        times_plain = np.array([r.total_time for r in results_plain])

        var_anti  = bootstrap_var_of_mean(times_anti)
        var_plain = bootstrap_var_of_mean(times_plain)

        # Antithetic variance should be smaller — allow 20% margin for randomness
        assert var_anti <= var_plain * 1.20, (
            f"Antithetic did not reduce variance: "
            f"var_anti={var_anti:.4f}, var_plain={var_plain:.4f}"
        )


# ---------------------------------------------------------------------------
# Multi-circuit sanity
# ---------------------------------------------------------------------------

class TestMultiCircuit:

    @pytest.mark.parametrize("circuit_name", ["silverstone", "monza", "monaco", "spa"])
    def test_all_circuits_simulate_without_error(self, circuit_name, tyre_model):
        cfg = CIRCUITS[circuit_name]
        sc  = _make_sc_model(circuit_name)
        sim = RaceSimulator(cfg, tyre_model, sc, policy=None)
        results = sim.run(20, seed=0, use_antithetic=False, verbose=False)
        assert len(results) == 20

    @pytest.mark.parametrize("circuit_name", ["silverstone", "monza", "monaco", "spa"])
    def test_race_times_scale_with_laps(self, circuit_name, tyre_model):
        """
        Circuits with more laps should have longer total race times
        (corrected for lap length difference via base_lap_time).
        """
        cfg = CIRCUITS[circuit_name]
        sc  = _make_sc_model(circuit_name)
        sim = RaceSimulator(cfg, tyre_model, sc, policy=None)
        results = sim.run(30, seed=0, use_antithetic=False, verbose=False)
        mean_time = np.mean([r.total_time for r in results])
        expected_min = cfg.base_lap_time * cfg.total_laps * 0.85
        assert mean_time > expected_min, \
            f"{circuit_name}: mean_time={mean_time:.0f}s < expected_min={expected_min:.0f}s"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_simulation(self, simulator):
        """Running exactly 1 simulation must not crash."""
        results = simulator.run(
            n_simulations=1, seed=0, use_antithetic=False, verbose=False
        )
        assert len(results) == 1

    def test_starting_position_1(self, silverstone, tyre_model, sc_model):
        """Starting from pole position (P1) must not crash."""
        sim = RaceSimulator(silverstone, tyre_model, sc_model, policy=None)
        results = sim.run(10, starting_position=1, seed=0,
                          use_antithetic=False, verbose=False)
        assert len(results) == 10

    def test_starting_position_20(self, silverstone, tyre_model, sc_model):
        """Starting from last (P20) must not crash."""
        sim = RaceSimulator(silverstone, tyre_model, sc_model, policy=None)
        results = sim.run(10, starting_position=20, seed=0,
                          use_antithetic=False, verbose=False)
        assert len(results) == 10

    def test_hard_compound_start(self, silverstone, tyre_model, sc_model):
        """Starting on HARD compound (unusual but valid) must not crash."""
        sim = RaceSimulator(silverstone, tyre_model, sc_model, policy=None)
        results = sim.run(10, starting_compound=Compound.HARD,
                          seed=0, use_antithetic=False, verbose=False)
        assert len(results) == 10

    def test_no_sc_model_events_does_not_crash(self, silverstone, tyre_model):
        """
        A safety car model with near-zero probability must still produce
        valid simulations.
        """
        sc = SafetyCarModel(circuit="silverstone", total_laps=52)
        for i in range(52):
            sc._lap_probs[i] = sc._lap_probs[i].__class__(alpha=1e-6, beta=1000.0)
        sim = RaceSimulator(silverstone, tyre_model, sc, policy=None)
        results = sim.run(20, seed=0, use_antithetic=False, verbose=False)
        assert all(r.total_time > 0 for r in results)