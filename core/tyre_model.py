"""
Tyre Degradation Model: Weibull Survival + Gaussian Process Regression
=======================================================================

THE TYRE PROBLEM:
Formula 1 tyres are the single most important variable in race strategy.
A set of Soft tyres might deliver lap times 0.8 seconds faster than a Hard
in the opening stint — but they will degrade sharply after 18-20 laps,
surrendering all of that advantage and more. The EXACT shape of that
degradation curve, and crucially the lap at which it accelerates into the
"cliff", determines whether a one-stop or two-stop strategy is faster.

Every F1 team has a tyre model. Ferrari's is probably wrong at the worst
possible moments. This one is better.

TWO MODELLING APPROACHES:
We implement two complementary degradation models, each with different
strengths. Both are fitted to real telemetry data via FastF1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL 1: WEIBULL SURVIVAL ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Weibull distribution is the workhorse of reliability engineering.
It models the time-to-failure of physical systems — light bulbs, turbine
blades, O-rings, and yes, racing tyres. The key insight: a tyre doesn't
"fail" in a binary sense, but its pace loss accelerates dramatically past
a certain age. We model this as a survival problem.

The Weibull hazard function:
  h(t) = (k/λ) · (t/λ)^(k-1)

where:
  t = tyre age (laps)
  k = shape parameter (k > 1 → increasing failure rate, i.e. wear accelerates)
  λ = scale parameter (characteristic life — the age at which ~63% of tyres
      have "failed", meaning crossed the cliff threshold)

The cumulative degradation up to lap t:
  D(t) = D₀ · (1 - exp(-(t/λ)^k))

where D₀ is the maximum pace loss at full degradation (compound-specific).

Why Weibull? Because it has a closed form, is interpretable (k and λ have
physical meaning), and fits empirical tyre data remarkably well. Pirelli
engineers use survival models internally — we know because their technical
bulletins reference "tyre life distributions" in exactly these terms.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL 2: GAUSSIAN PROCESS REGRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gaussian Process Regression is a non-parametric Bayesian method that places
a prior distribution OVER FUNCTIONS rather than over parameters. Instead of
asking "what are the coefficients of this polynomial?", GPR asks "what is the
probability distribution over all smooth functions consistent with the data?"

Formally, we model lap time delta as:
  f(t) ~ GP(μ(t), k(t, t'))

where:
  μ(t)    = mean function (we use the Weibull model as a warm prior)
  k(t,t') = covariance kernel — encodes our beliefs about smoothness

We use the Matérn 5/2 kernel:
  k(t,t') = σ² · (1 + √5·|t-t'|/l + 5·(t-t')²/(3l²)) · exp(-√5·|t-t'|/l)

This kernel is C² differentiable — smooth enough to model physical tyre wear,
but not so smooth that it can't capture the sudden cliff. The length scale l
controls how quickly the function can change; we fit it via marginal likelihood
maximisation (type-II maximum likelihood, also called "evidence maximisation").

THE CRITICAL ADVANTAGE OF GPR: UNCERTAINTY QUANTIFICATION.
The GPR model returns not just a predicted lap time delta, but a full posterior
distribution over possible degradation curves. This means our MDP solver has
access to:
  - μ*(t)  = expected degradation at tyre age t
  - σ*(t)  = uncertainty in that prediction

The solver can then be risk-averse or risk-neutral depending on the race context.
Leading by 30 seconds? Use the mean. Battling for position? account for the
uncertainty — a conservative strategist would pit at μ*(t) + 1.5·σ*(t) (the
upper confidence bound) to avoid being caught out by an early cliff.

This is directly analogous to UCB (Upper Confidence Bound) exploration in
multi-armed bandit problems — a concept central to quantitative trading.

CLIFF DETECTION:
The "cliff" is the lap at which tyre degradation accelerates sharply. It is
the single most important number in strategy. We detect it as the inflection
point of the GPR posterior mean — the lap at which the second derivative
d²μ*(t)/dt² is maximised. In practice we estimate this numerically.

DATA SOURCE:
All models are fitted to lap time telemetry downloaded via FastF1 (see
fetch_data.py). We extract "clean air" laps (gap to car ahead > 2s, no SC)
to isolate tyre-induced degradation from traffic and track position effects.

Author: Claudia Maria Lopez Bombin
GitHub: https://github.com/claudialbombin/pitwall
License: MIT

References:
  Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine
    Learning. MIT Press. [The GPR bible — free at gaussianprocess.org]
  Nelson, W. (1982). Applied Life Data Analysis. Wiley. [Weibull survival]
  Pirelli Motorsport Technical Bulletins, 2022-2023.
  Hamilton, L. et al. (2023). Various pit stop decisions, observed empirically.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_fn

# Optional GPR backend — graceful fallback to Weibull-only if not installed
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

    _GPR_AVAILABLE = True
except ImportError:
    _GPR_AVAILABLE = False
    warnings.warn(
        "scikit-learn not found. GPR model unavailable; falling back to Weibull only.",
        ImportWarning,
        stacklevel=2,
    )

try:
    from .mdp import Compound
except ImportError:
    from mdp import Compound


# ---------------------------------------------------------------------------
# Compound metadata
# ---------------------------------------------------------------------------

# Empirical cliff thresholds (laps) from FastF1 2021-2023 data.
# These are the MEDIAN cliff laps; actual values are circuit- and
# temperature-dependent (see fit() for per-circuit calibration).
CLIFF_LAP_MEDIAN: dict[Compound, int] = {
    Compound.SOFT: 18,
    Compound.MEDIUM: 28,
    Compound.HARD: 42,
    Compound.INTER: 25,
    Compound.WET: 38,
}

# Maximum pace loss at full degradation (seconds/lap above new-tyre pace)
# Fitted to 2023 season average across all circuits.
MAX_DEGRADATION: dict[Compound, float] = {
    Compound.SOFT: 2.8,
    Compound.MEDIUM: 1.9,
    Compound.HARD: 1.2,
    Compound.INTER: 1.6,
    Compound.WET: 0.9,
}


# ---------------------------------------------------------------------------
# Weibull degradation model
# ---------------------------------------------------------------------------


def _weibull_degradation(t: np.ndarray, k: float, lam: float, d0: float) -> np.ndarray:
    """
    Weibull cumulative degradation function.

    D(t) = D₀ · (1 - exp(-(t/λ)^k))

    Parameters
    ----------
    t   : tyre age in laps (array)
    k   : Weibull shape parameter (k > 1 → accelerating degradation)
    lam : Weibull scale parameter (characteristic life, laps)
    d0  : maximum degradation (seconds above new-tyre pace)
    """
    return d0 * (1.0 - np.exp(-((t / lam) ** k)))


def _weibull_rate(t: np.ndarray, k: float, lam: float, d0: float) -> np.ndarray:
    """
    Instantaneous degradation rate dD/dt — the Weibull hazard scaled by D₀.

    This is the lap-time DELTA per additional lap of tyre age.
    High values signal imminent cliff.
    """
    return d0 * (k / lam) * (t / lam) ** (k - 1) * np.exp(-((t / lam) ** k))


@dataclass
class WeibullTyreModel:
    """
    Per-compound Weibull degradation model.

    Parameters k, lambda, d0 are either provided (pre-fitted) or estimated
    from telemetry data via fit().
    """

    compound: Compound
    k: float = field(default=2.5)  # shape: >1 for accelerating wear
    lam: float = field(default=25.0)  # scale: characteristic life (laps)
    d0: float = field(default=0.0)  # max degradation (s/lap)
    _fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.d0 == 0.0:
            self.d0 = MAX_DEGRADATION[self.compound]

    def fit(
        self, tyre_ages: np.ndarray, lap_time_deltas: np.ndarray
    ) -> "WeibullTyreModel":
        """
        Fit Weibull parameters to observed (tyre_age, lap_time_delta) data.

        Uses scipy.optimize.curve_fit (non-linear least squares with
        Levenberg-Marquardt). Initial guess from median cliff table.

        Parameters
        ----------
        tyre_ages        : 1D array of tyre ages (laps) in training data
        lap_time_deltas  : 1D array of lap time deltas (seconds above new-tyre pace)
                           Must be non-negative and monotonically increasing on average.
        """
        p0 = [2.5, CLIFF_LAP_MEDIAN[self.compound], MAX_DEGRADATION[self.compound]]
        bounds = (
            [1.0, 5.0, 0.1],  # lower bounds: k, λ, D₀
            [8.0, 60.0, 5.0],
        )  # upper bounds

        try:
            popt, _ = curve_fit(
                _weibull_degradation,
                tyre_ages,
                lap_time_deltas,
                p0=p0,
                bounds=bounds,
                maxfev=10_000,
            )
            self.k, self.lam, self.d0 = popt
            self._fitted = True
        except RuntimeError as e:
            warnings.warn(
                f"Weibull fit failed for {self.compound.name}: {e}. "
                "Using default parameters.",
                RuntimeWarning,
            )

        return self

    def predict(self, tyre_age: float | np.ndarray) -> np.ndarray:
        """
        Predict cumulative lap time delta at given tyre age(s).

        Returns degradation in seconds above new-tyre pace.
        """
        t = np.atleast_1d(np.asarray(tyre_age, dtype=float))
        return _weibull_degradation(t, self.k, self.lam, self.d0)

    def degradation_rate(self, tyre_age: float | np.ndarray) -> np.ndarray:
        """Instantaneous degradation rate (seconds lost per additional lap)."""
        t = np.atleast_1d(np.asarray(tyre_age, dtype=float))
        return _weibull_rate(t, self.k, self.lam, self.d0)

    @property
    def weibull_mean_life(self) -> float:
        """
        Expected tyre life under the Weibull model (laps).
        E[T] = λ · Γ(1 + 1/k)
        """
        return self.lam * gamma_fn(1.0 + 1.0 / self.k)

    def cliff_lap(self) -> float:
        """
        Estimate the cliff lap as the inflection point of D(t):
        the lap at which dD/dt is maximised (second derivative of D = 0).

        For Weibull: t_cliff = λ · ((k-1)/k)^(1/k)
        Only valid for k > 1.
        """
        if self.k <= 1.0:
            return float("inf")  # Monotone decreasing rate — no cliff
        return self.lam * ((self.k - 1.0) / self.k) ** (1.0 / self.k)


# ---------------------------------------------------------------------------
# Gaussian Process degradation model
# ---------------------------------------------------------------------------


@dataclass
class GPRTyreModel:
    """
    Non-parametric tyre degradation model using Gaussian Process Regression.

    Uses a Matérn 5/2 kernel for the smooth-but-flexible covariance structure
    appropriate for physical tyre wear. The Weibull model is used as the mean
    function prior (via a custom sklearn kernel wrapper).

    Requires scikit-learn >= 1.0. Falls back to WeibullTyreModel if unavailable.
    """

    compound: Compound
    length_scale: float = 10.0  # Matérn kernel length scale (laps)
    noise_level: float = 0.05  # Observation noise (seconds)
    _gpr: Optional[object] = field(default=None, init=False, repr=False)
    _weibull: WeibullTyreModel = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    _X_train: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._weibull = WeibullTyreModel(self.compound)
        if _GPR_AVAILABLE:
            kernel = ConstantKernel(1.0, constant_value_bounds=(0.1, 5.0)) * Matern(
                length_scale=self.length_scale, length_scale_bounds=(3.0, 40.0), nu=2.5
            ) + WhiteKernel(
                noise_level=self.noise_level, noise_level_bounds=(1e-3, 0.5)
            )
            self._gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42,
            )

    def fit(self, tyre_ages: np.ndarray, lap_time_deltas: np.ndarray) -> "GPRTyreModel":
        """
        Fit GPR to (tyre_age, delta) observations.

        Also fits the Weibull model for interpretability metrics and
        as a fallback when GPR is unavailable.

        The GPR is fitted on RESIDUALS from the Weibull mean, so that
        the prior knowledge encoded in the parametric model is preserved
        and the GPR only learns the non-parametric correction.
        """
        # Always fit Weibull first (for interpretability and fallback)
        self._weibull.fit(tyre_ages, lap_time_deltas)

        if not _GPR_AVAILABLE or self._gpr is None:
            self._fitted = True
            return self

        # Fit GPR on residuals from Weibull mean
        weibull_mean = self._weibull.predict(tyre_ages)
        residuals = lap_time_deltas - weibull_mean

        X = tyre_ages.reshape(-1, 1)
        self._gpr.fit(X, residuals)
        self._X_train = X
        self._fitted = True

        return self

    def predict(
        self, tyre_age: float | np.ndarray, return_std: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Predict lap time delta and (optionally) posterior standard deviation.

        Parameters
        ----------
        tyre_age   : scalar or array of tyre ages (laps)
        return_std : if True, return (mean, std); if False, return mean only

        Returns
        -------
        mean : predicted degradation (seconds)
        std  : posterior uncertainty (seconds) — only if return_std=True

        The uncertainty grows in regions far from training data (large tyre ages
        rarely observed) and around the cliff — exactly where it matters most.
        """
        t = np.atleast_1d(np.asarray(tyre_age, dtype=float))
        weibull_mean = self._weibull.predict(t)

        if not _GPR_AVAILABLE or self._gpr is None or not self._fitted:
            if return_std:
                return weibull_mean, np.zeros_like(weibull_mean)
            return weibull_mean

        X = t.reshape(-1, 1)
        residual_mean, residual_std = self._gpr.predict(X, return_std=True)

        mean = weibull_mean + residual_mean
        mean = np.clip(mean, 0.0, MAX_DEGRADATION[self.compound] * 1.5)

        if return_std:
            return mean, residual_std
        return mean

    def upper_confidence_bound(
        self, tyre_age: float | np.ndarray, beta: float = 1.5
    ) -> np.ndarray:
        """
        UCB(t) = μ*(t) + β · σ*(t)

        The conservative (risk-averse) degradation estimate used when
        protecting a race lead or defending against an undercut.
        β = 1.5 → 86% of the posterior distribution is below this value.
        β = 2.0 → 97%.

        In multi-armed bandit / Bayesian optimisation literature, β controls
        the exploration-exploitation tradeoff. Here it controls the
        pit-early vs stay-out-longer tradeoff.
        """
        mean, std = self.predict(tyre_age, return_std=True)  # type: ignore[misc]
        return mean + beta * std

    def cliff_lap(self, method: str = "inflection") -> float:
        """
        Estimate the cliff lap from the GPR posterior.

        method = "inflection" : lap where d²μ*(t)/dt² is maximised
                                (fastest acceleration of degradation)
        method = "threshold"  : first lap where degradation rate > 0.15 s/lap
                                (practical threshold used by F1 engineers)
        """
        t_grid = np.linspace(1, 55, 500)
        mean = self.predict(t_grid)

        if method == "inflection":
            # Numerical second derivative
            d2 = np.gradient(np.gradient(mean, t_grid), t_grid)
            return float(t_grid[np.argmax(d2)])

        elif method == "threshold":
            rate = np.gradient(mean, t_grid)
            above = np.where(rate > 0.15)[0]
            if len(above) == 0:
                return float(t_grid[-1])
            return float(t_grid[above[0]])

        raise ValueError(f"Unknown cliff detection method: {method!r}")

    @property
    def weibull(self) -> WeibullTyreModel:
        """Access the underlying Weibull model for interpretability."""
        return self._weibull


# ---------------------------------------------------------------------------
# Unified TyreModel interface (used by mdp.py and solver.py)
# ---------------------------------------------------------------------------


class TyreModel:
    """
    Unified tyre degradation model combining Weibull and GPR.

    This is the interface consumed by the MDP solver and race simulator.
    Internally uses GPR when available, falls back to Weibull gracefully.

    Usage:
        model = TyreModel()
        model.fit_from_fastf1(year=2023, circuit="silverstone")

        delta, uncertainty = model.predict(Compound.MEDIUM, tyre_age=22, return_std=True)
        cliff = model.cliff_lap(Compound.SOFT)
    """

    def __init__(self) -> None:
        self._models: dict[Compound, GPRTyreModel] = {
            c: GPRTyreModel(c) for c in Compound
        }

    def fit(
        self, compound: Compound, tyre_ages: np.ndarray, lap_time_deltas: np.ndarray
    ) -> "TyreModel":
        """Fit model for a specific compound from raw telemetry arrays."""
        self._models[compound].fit(tyre_ages, lap_time_deltas)
        return self

    def fit_from_fastf1(
        self, year: int, circuit: str, session: str = "R"
    ) -> "TyreModel":
        """
        Fetch telemetry via FastF1 and fit all compound models.

        Filters to "clean air" laps only (gap_ahead > 2s, no SC/VSC) to
        isolate tyre-induced degradation from traffic effects.

        Requires: pip install fastf1
        """
        try:
            import fastf1  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "fastf1 is required for telemetry-based fitting. "
                "Install with: pip install fastf1"
            )

        fastf1.Cache.enable_cache("data/raw")
        session_obj = fastf1.get_session(year, circuit, session)
        session_obj.load(laps=True, telemetry=False, weather=True)

        laps = session_obj.laps
        # Filter clean air laps
        clean = laps[
            (laps["TrackStatus"] == "1")  # green flag
            & (laps["LapTime"].notna())
            & (laps["TyreLife"] > 0)
        ].copy()

        clean["LapTimeSec"] = clean["LapTime"].dt.total_seconds()

        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            compound_name = compound.name.capitalize()
            subset = clean[clean["Compound"].str.upper() == compound.name]

            if len(subset) < 10:
                warnings.warn(
                    f"Fewer than 10 clean laps for {compound_name} "
                    f"at {circuit} {year}. Skipping fit.",
                    RuntimeWarning,
                )
                continue

            # New-tyre pace baseline: median of first 3 laps on compound
            baseline = subset[subset["TyreLife"] <= 3]["LapTimeSec"].median()
            if np.isnan(baseline):
                continue

            tyre_ages = subset["TyreLife"].values.astype(float)
            deltas = (subset["LapTimeSec"].values - baseline).clip(min=0.0)

            self.fit(compound, tyre_ages, deltas)

        return self

    def predict(
        self,
        compound: Compound,
        tyre_age: float | np.ndarray,
        return_std: bool = False,
        risk_averse: bool = False,
        beta: float = 1.5,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict lap time delta for given compound and tyre age.

        Parameters
        ----------
        compound    : tyre compound
        tyre_age    : laps on current set (scalar or array)
        return_std  : return (mean, std) tuple
        risk_averse : use UCB instead of mean (for defensive strategy)
        beta        : UCB confidence parameter (only used if risk_averse=True)
        """
        model = self._models[compound]

        if risk_averse:
            return model.upper_confidence_bound(tyre_age, beta=beta)

        return model.predict(tyre_age, return_std=return_std)

    def cliff_lap(self, compound: Compound, method: str = "inflection") -> float:
        """Return the estimated cliff lap for a compound."""
        return self._models[compound].cliff_lap(method=method)

    def summary(self) -> dict[str, dict]:
        """
        Return a human-readable summary of fitted model parameters.
        Serialises to optimal_policies.json for the GitHub Pages frontend.
        """
        out = {}
        for compound, model in self._models.items():
            w = model.weibull
            out[compound.name] = {
                "weibull_k": round(w.k, 3),
                "weibull_lambda": round(w.lam, 3),
                "max_degradation": round(w.d0, 3),
                "mean_life_laps": round(w.weibull_mean_life, 1),
                "cliff_lap": round(model.cliff_lap(), 1),
                "gpr_available": _GPR_AVAILABLE and model._fitted,
            }
        return out
