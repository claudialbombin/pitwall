"""
Tests for core/tyre_model.py
==============================

Testing philosophy for statistical models:

The challenge with testing probabilistic and ML models is that we cannot
test exact outputs — they depend on data, random seeds, and numerical
optimisation. Instead we test:

  1. MATHEMATICAL CONSTRAINTS — properties that must hold regardless of data:
     - Degradation is non-negative
     - Degradation is monotonically non-decreasing (on average)
     - UCB >= mean (by definition)
     - GPR uncertainty is non-negative

  2. PHYSICAL PLAUSIBILITY — outputs must be in physically reasonable ranges:
     - Cliff lap must be between 5 and 55 laps
     - Maximum degradation must be between 0.5 and 6 seconds
     - Weibull shape k > 1 (accelerating wear — the whole premise of the model)

  3. QUALITATIVE ORDERING — even without ground truth, some orderings must hold:
     - SOFT degrades faster than MEDIUM, which degrades faster than HARD
     - UCB(β=2) >= UCB(β=1) >= mean (strict monotonicity in β)
     - Weibull mean life must be λ·Γ(1 + 1/k)

  4. NUMERICAL STABILITY — models must not produce NaN, inf, or negative variance.

Author: Claudia Maria Lopez Bombin
GitHub: https://github.com/claudialbombin/pitwall
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from tyre_model import (
    WeibullTyreModel,
    GPRTyreModel,
    TyreModel,
    _weibull_degradation,
    MAX_DEGRADATION,
)
from mdp import Compound


# ---------------------------------------------------------------------------
# Synthetic training data fixtures
# ---------------------------------------------------------------------------


def _synthetic_tyre_data(
    compound: Compound, n_points: int = 40, noise_std: float = 0.05, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (tyre_age, lap_time_delta) data from the true Weibull model.
    Used to test fitting routines against known ground truth.
    """
    rng = np.random.default_rng(seed)
    params = {
        Compound.SOFT: {"k": 2.8, "lam": 16, "d0": 2.8},
        Compound.MEDIUM: {"k": 2.4, "lam": 26, "d0": 1.9},
        Compound.HARD: {"k": 2.0, "lam": 40, "d0": 1.2},
    }
    p = params.get(compound, params[Compound.MEDIUM])
    ages = np.sort(rng.uniform(0.5, p["lam"] * 1.5, n_points))
    true_deg = _weibull_degradation(ages, p["k"], p["lam"], p["d0"])
    noisy_deg = np.clip(true_deg + rng.normal(0, noise_std, n_points), 0, None)
    return ages, noisy_deg


@pytest.fixture
def soft_data():
    return _synthetic_tyre_data(Compound.SOFT)


@pytest.fixture
def medium_data():
    return _synthetic_tyre_data(Compound.MEDIUM)


@pytest.fixture
def hard_data():
    return _synthetic_tyre_data(Compound.HARD)


@pytest.fixture
def fitted_soft_weibull(soft_data):
    ages, deltas = soft_data
    model = WeibullTyreModel(Compound.SOFT)
    model.fit(ages, deltas)
    return model


@pytest.fixture
def fitted_medium_weibull(medium_data):
    ages, deltas = medium_data
    model = WeibullTyreModel(Compound.MEDIUM)
    model.fit(ages, deltas)
    return model


@pytest.fixture
def fitted_hard_weibull(hard_data):
    ages, deltas = hard_data
    model = WeibullTyreModel(Compound.HARD)
    model.fit(ages, deltas)
    return model


# ---------------------------------------------------------------------------
# Weibull mathematical properties
# ---------------------------------------------------------------------------


class TestWeibullMathematics:
    def test_weibull_at_zero_is_zero(self):
        """D(0) = D₀ · (1 - exp(0)) = 0 for all k, λ, D₀."""
        for k in [1.5, 2.5, 4.0]:
            for lam in [10, 25, 40]:
                result = _weibull_degradation(np.array([0.0]), k, lam, 2.0)
                assert result[0] == pytest.approx(0.0, abs=1e-10), (
                    f"D(0) != 0 for k={k}, lam={lam}"
                )

    def test_weibull_approaches_d0_at_infinity(self):
        """D(t) → D₀ as t → ∞."""
        d0 = 2.5
        for k in [2.0, 3.0]:
            result = _weibull_degradation(np.array([1e6]), k, 20.0, d0)
            assert result[0] == pytest.approx(d0, rel=1e-4), f"D(∞) != D₀ for k={k}"

    def test_weibull_is_monotone(self):
        """D(t) must be non-decreasing for all t ≥ 0."""
        t = np.linspace(0, 60, 500)
        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            m = WeibullTyreModel(compound)
            d = m.predict(t)
            diffs = np.diff(d)
            assert np.all(diffs >= -1e-10), (
                f"Weibull not monotone for {compound.name}: min diff = {diffs.min():.6f}"
            )

    def test_weibull_rate_is_non_negative(self):
        """Instantaneous degradation rate must be ≥ 0."""
        t = np.linspace(0.01, 60, 500)
        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            m = WeibullTyreModel(compound)
            rate = m.degradation_rate(t)
            assert np.all(rate >= -1e-10), (
                f"Negative rate for {compound.name}: min = {rate.min():.6f}"
            )

    def test_weibull_mean_life_formula(self):
        """
        Weibull mean life = λ · Γ(1 + 1/k).
        We test this against scipy.special.gamma directly.
        """
        from scipy.special import gamma as gamma_fn

        for k, lam in [(2.5, 20), (3.0, 30), (1.8, 15)]:
            m = WeibullTyreModel(Compound.MEDIUM, k=k, lam=lam, d0=2.0)
            expected = lam * gamma_fn(1 + 1 / k)
            assert m.weibull_mean_life == pytest.approx(expected, rel=1e-6)

    def test_cliff_lap_formula_for_k_greater_than_1(self):
        """
        Cliff lap = λ · ((k-1)/k)^(1/k) for k > 1.
        This is the inflection point of the Weibull CDF.
        """
        k, lam = 2.5, 20.0
        m = WeibullTyreModel(Compound.SOFT, k=k, lam=lam, d0=2.0)
        expected = lam * ((k - 1) / k) ** (1 / k)
        assert m.cliff_lap() == pytest.approx(expected, rel=1e-6)

    def test_cliff_lap_infinity_when_k_equals_1(self):
        """For k = 1 (exponential distribution), there is no cliff — return inf."""
        m = WeibullTyreModel(Compound.HARD, k=1.0, lam=30.0, d0=1.5)
        assert m.cliff_lap() == float("inf")

    def test_compound_ordering_at_age_20(self):
        """
        At tyre age 20, default SOFT must degrade more than MEDIUM,
        which must degrade more than HARD.
        """
        p_soft = WeibullTyreModel(Compound.SOFT).predict(20)
        p_medium = WeibullTyreModel(Compound.MEDIUM).predict(20)
        p_hard = WeibullTyreModel(Compound.HARD).predict(20)
        assert p_soft[0] >= p_medium[0], (
            f"SOFT < MEDIUM at age 20: {p_soft[0]:.3f} < {p_medium[0]:.3f}"
        )
        assert p_medium[0] >= p_hard[0], (
            f"MEDIUM < HARD at age 20: {p_medium[0]:.3f} < {p_hard[0]:.3f}"
        )


# ---------------------------------------------------------------------------
# Weibull fitting
# ---------------------------------------------------------------------------


class TestWeibullFitting:
    def test_fit_does_not_raise(self, soft_data):
        ages, deltas = soft_data
        model = WeibullTyreModel(Compound.SOFT)
        model.fit(ages, deltas)  # should not raise

    def test_fit_parameters_in_physical_range(self, fitted_soft_weibull):
        """Fitted parameters must be physically plausible."""
        m = fitted_soft_weibull
        assert m.k > 1.0, f"k={m.k:.3f} ≤ 1 — implies non-accelerating wear"
        assert m.k < 10.0, f"k={m.k:.3f} unrealistically high"
        assert m.lam > 3.0, f"λ={m.lam:.1f} unrealistically low (< 3 laps)"
        assert m.lam < 60.0, f"λ={m.lam:.1f} unrealistically high (> 60 laps)"
        assert m.d0 > 0.3, f"D₀={m.d0:.3f} unrealistically low"
        assert m.d0 < 6.0, f"D₀={m.d0:.3f} unrealistically high"

    def test_fit_recovers_shape_approximately(self, medium_data):
        """
        Fitting to synthetic data from k=2.4, λ=26 should recover
        parameters within ±30% of ground truth.
        """
        ages, deltas = medium_data
        model = WeibullTyreModel(Compound.MEDIUM)
        model.fit(ages, deltas)
        assert abs(model.k - 2.4) / 2.4 < 0.30, f"k={model.k:.3f} off by >30%"
        assert abs(model.lam - 26) / 26 < 0.30, f"λ={model.lam:.1f} off by >30%"

    def test_fitted_model_predictions_non_negative(self, fitted_soft_weibull):
        """Predictions from a fitted model must always be ≥ 0."""
        ages = np.linspace(0, 50, 200)
        pred = fitted_soft_weibull.predict(ages)
        assert np.all(pred >= -1e-9), f"Negative prediction: min={pred.min():.4f}"

    def test_fit_is_idempotent(self, soft_data):
        """Fitting the same model twice should give the same result."""
        ages, deltas = soft_data
        m1 = WeibullTyreModel(Compound.SOFT)
        m1.fit(ages, deltas)
        m2 = WeibullTyreModel(Compound.SOFT)
        m2.fit(ages, deltas)
        assert m1.k == pytest.approx(m2.k, rel=1e-6)
        assert m1.lam == pytest.approx(m2.lam, rel=1e-6)
        assert m1.d0 == pytest.approx(m2.d0, rel=1e-6)


# ---------------------------------------------------------------------------
# GPR model properties
# ---------------------------------------------------------------------------


class TestGPRTyreModel:
    def test_gpr_predict_non_negative(self, medium_data):
        """GPR predictions must be ≥ 0 (degradation cannot be negative)."""
        ages, deltas = medium_data
        model = GPRTyreModel(Compound.MEDIUM)
        model.fit(ages, deltas)
        test_ages = np.linspace(0, 40, 100)
        pred = model.predict(test_ages)
        assert np.all(np.atleast_1d(pred) >= -1e-6), (
            f"Negative GPR prediction: min = {np.min(pred):.4f}"
        )

    def test_gpr_uncertainty_non_negative(self, medium_data):
        """Posterior standard deviation must be ≥ 0 everywhere."""
        ages, deltas = medium_data
        model = GPRTyreModel(Compound.MEDIUM)
        model.fit(ages, deltas)
        test_ages = np.linspace(0, 50, 100)
        _, std = model.predict(test_ages, return_std=True)
        assert np.all(np.atleast_1d(std) >= -1e-9), (
            f"Negative GPR std: min = {np.min(std):.6f}"
        )

    def test_ucb_exceeds_mean(self, medium_data):
        """UCB(β > 0) must always be ≥ mean prediction."""
        ages, deltas = medium_data
        model = GPRTyreModel(Compound.MEDIUM)
        model.fit(ages, deltas)
        test_ages = np.linspace(1, 40, 100)
        mean = np.atleast_1d(model.predict(test_ages))
        ucb = model.upper_confidence_bound(test_ages, beta=1.5)
        assert np.all(ucb >= mean - 1e-9), f"UCB < mean at {np.sum(ucb < mean)} points"

    def test_ucb_increases_with_beta(self, medium_data):
        """Larger β must produce a larger (more conservative) UCB."""
        ages, deltas = medium_data
        model = GPRTyreModel(Compound.MEDIUM)
        model.fit(ages, deltas)
        test_ages = np.linspace(5, 35, 50)
        ucb1 = model.upper_confidence_bound(test_ages, beta=1.0)
        ucb2 = model.upper_confidence_bound(test_ages, beta=2.0)
        assert np.all(ucb2 >= ucb1 - 1e-9), "UCB(β=2) < UCB(β=1) at some points"

    def test_gpr_predict_returns_correct_shape(self, medium_data):
        ages, deltas = medium_data
        model = GPRTyreModel(Compound.MEDIUM)
        model.fit(ages, deltas)

        # Scalar input → scalar-like output
        scalar_pred = model.predict(10.0)
        assert np.atleast_1d(scalar_pred).shape == (1,)

        # Array input → matching array
        arr = np.array([5.0, 10.0, 20.0])
        arr_pred = model.predict(arr)
        assert np.atleast_1d(arr_pred).shape == (3,)

    def test_gpr_cliff_lap_in_range(self, medium_data):
        """Cliff lap detected from GPR must be in a physically reasonable range."""
        ages, deltas = medium_data
        model = GPRTyreModel(Compound.MEDIUM)
        model.fit(ages, deltas)
        cliff = model.cliff_lap(method="inflection")
        assert 5 <= cliff <= 55, f"Cliff lap {cliff:.1f} outside range [5, 55]"

    def test_gpr_fallback_without_sklearn(self, medium_data, monkeypatch):
        """
        Without scikit-learn, GPR must fall back to Weibull predictions
        without raising an exception.
        """
        import tyre_model as tm

        monkeypatch.setattr(tm, "_GPR_AVAILABLE", False)
        ages, deltas = medium_data
        model = GPRTyreModel(Compound.MEDIUM)
        model.fit(ages, deltas)
        pred = model.predict(np.array([10.0, 20.0]))
        assert not np.any(np.isnan(np.atleast_1d(pred)))


# ---------------------------------------------------------------------------
# TyreModel unified interface
# ---------------------------------------------------------------------------


class TestTyreModelInterface:
    @pytest.fixture
    def fitted_tyre_model(self):
        m = TyreModel()
        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            ages, deltas = _synthetic_tyre_data(compound)
            m.fit(compound, ages, deltas)
        return m

    def test_predict_returns_array(self, fitted_tyre_model):
        pred = fitted_tyre_model.predict(Compound.MEDIUM, np.arange(0, 30))
        assert hasattr(pred, "__len__")
        assert len(pred) == 30  # type: ignore[arg-type]

    def test_predict_non_negative(self, fitted_tyre_model):
        pred = fitted_tyre_model.predict(Compound.SOFT, np.linspace(0, 40, 100))
        assert np.all(np.atleast_1d(pred) >= -1e-6)

    def test_risk_averse_exceeds_mean(self, fitted_tyre_model):
        ages = np.linspace(5, 35, 50)
        mean = np.atleast_1d(fitted_tyre_model.predict(Compound.MEDIUM, ages))
        ucb = np.atleast_1d(
            fitted_tyre_model.predict(Compound.MEDIUM, ages, risk_averse=True, beta=1.5)
        )
        assert np.all(ucb >= mean - 1e-9)

    def test_cliff_lap_all_compounds(self, fitted_tyre_model):
        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            cliff = fitted_tyre_model.cliff_lap(compound)
            assert 3 <= cliff <= 60, (
                f"Cliff lap {cliff:.1f} out of range for {compound.name}"
            )

    def test_cliff_ordering_soft_before_hard(self, fitted_tyre_model):
        """SOFT compound cliff must occur before HARD compound cliff."""
        cliff_soft = fitted_tyre_model.cliff_lap(Compound.SOFT)
        cliff_hard = fitted_tyre_model.cliff_lap(Compound.HARD)
        assert cliff_soft < cliff_hard, (
            f"SOFT cliff ({cliff_soft:.1f}) not before HARD cliff ({cliff_hard:.1f})"
        )

    def test_summary_contains_all_compounds(self, fitted_tyre_model):
        summary = fitted_tyre_model.summary()
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            assert compound in summary, f"{compound} missing from summary"

    def test_summary_has_required_keys(self, fitted_tyre_model):
        summary = fitted_tyre_model.summary()
        required = {
            "weibull_k",
            "weibull_lambda",
            "max_degradation",
            "mean_life_laps",
            "cliff_lap",
        }
        for compound_name, params in summary.items():
            missing = required - set(params.keys())
            assert not missing, f"{compound_name} missing keys: {missing}"

    def test_no_nan_in_predictions(self, fitted_tyre_model):
        """No NaN values should appear in predictions over the full age range."""
        ages = np.arange(0, 55, dtype=float)
        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            pred = np.atleast_1d(fitted_tyre_model.predict(compound, ages))
            assert not np.any(np.isnan(pred)), f"NaN in predictions for {compound.name}"

    def test_no_inf_in_predictions(self, fitted_tyre_model):
        ages = np.arange(0, 55, dtype=float)
        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            pred = np.atleast_1d(fitted_tyre_model.predict(compound, ages))
            assert not np.any(np.isinf(pred)), f"Inf in predictions for {compound.name}"


# ---------------------------------------------------------------------------
# Default parameter sanity (without fitting)
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    def test_default_weibull_k_greater_than_1(self):
        """All default Weibull models must have k > 1 (accelerating wear)."""
        for compound in Compound:
            m = WeibullTyreModel(compound)
            assert m.k > 1.0, f"Default k={m.k} ≤ 1 for {compound.name}"

    def test_default_cliff_laps_match_domain_knowledge(self):
        """
        Default cliff laps must be within ±5 laps of known empirical values.
        SOFT: ~18, MEDIUM: ~28, HARD: ~42
        """
        tolerances = {
            Compound.SOFT: (13, 23),
            Compound.MEDIUM: (23, 33),
            Compound.HARD: (37, 47),
        }
        for compound, (lo, hi) in tolerances.items():
            m = WeibullTyreModel(compound)
            cliff = m.cliff_lap()
            assert lo <= cliff <= hi, (
                f"Default cliff for {compound.name}: {cliff:.1f} not in [{lo}, {hi}]"
            )

    def test_max_degradation_values(self):
        """Maximum degradation must be in physically realistic range [0.5, 5.0] seconds."""
        for compound, d0 in MAX_DEGRADATION.items():
            assert 0.5 <= d0 <= 5.0, (
                f"MAX_DEGRADATION[{compound.name}] = {d0} outside [0.5, 5.0]"
            )

    def test_soft_has_highest_max_degradation(self):
        """SOFT must have the highest maximum degradation of the dry compounds."""
        assert MAX_DEGRADATION[Compound.SOFT] >= MAX_DEGRADATION[Compound.MEDIUM]
        assert MAX_DEGRADATION[Compound.MEDIUM] >= MAX_DEGRADATION[Compound.HARD]
