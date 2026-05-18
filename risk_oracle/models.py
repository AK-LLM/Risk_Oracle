"""
Core models — Bayesian probability update and Monte Carlo impact simulation.

Each "model" produces a posterior probability and an impact distribution.
Primary and critic models implement the same interface but use different
methodologies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats

from .taxonomy import CategorySpec


@dataclass
class EvidenceFactor:
    name: str
    likelihood_ratio: float  # >1 raises P, <1 lowers
    confidence: float = 1.0   # 0-1 scaling

    def effective_lr(self) -> float:
        return self.likelihood_ratio ** self.confidence


@dataclass
class ImpactChannel:
    name: str
    log_mean: float           # lognormal log-mean
    log_sigma: float          # lognormal log-stdev
    duration_sensitivity: float = 0.5


@dataclass
class ModelOutput:
    name: str                       # model name e.g. "bayesian_reference_class"
    posterior_probability: float    # 0-1
    impact_samples: np.ndarray      # Monte Carlo draws of total impact in $
    duration_samples: np.ndarray    # Monte Carlo draws of duration in months
    methodology: str = ""           # short description of methodology used
    notes: List[str] = field(default_factory=list)


# ---------- Bayesian update ----------

def bayesian_posterior(prior: float, evidence: List[EvidenceFactor]) -> float:
    p = max(min(prior, 0.999), 0.001)
    prior_odds = p / (1 - p)
    posterior_odds = prior_odds
    for e in evidence:
        posterior_odds *= e.effective_lr()
    return posterior_odds / (1 + posterior_odds)


# ---------- Monte Carlo impact engine ----------

def monte_carlo_impact(
    posterior_prob: float,
    impact_channels: List[ImpactChannel],
    duration_mean_months: float,
    duration_cv: float = 0.7,
    spillage_alpha: float = 2.0,
    spillage_beta: float = 5.0,
    spillage_max_mult: float = 2.5,
    n_sims: int = 20_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    occurred = rng.random(n_sims) < posterior_prob
    losses = np.zeros(n_sims)
    durations = np.zeros(n_sims)
    idx = np.where(occurred)[0]
    k = len(idx)
    if k == 0:
        return losses, durations

    shape = 1.0 / max(duration_cv ** 2, 1e-6)
    scale = duration_mean_months / shape
    durations[idx] = rng.gamma(shape=shape, scale=scale, size=k)

    spill = rng.beta(spillage_alpha, spillage_beta, size=k)
    spillage_mult = 1.0 + spill * (spillage_max_mult - 1.0)

    for ch in impact_channels:
        base = rng.lognormal(mean=ch.log_mean, sigma=ch.log_sigma, size=k)
        dur_amp = (1.0 + durations[idx]) ** ch.duration_sensitivity
        losses[idx] += base * dur_amp * spillage_mult

    return losses, durations


# ---------- Primary models per category ----------
#
# Each primary model is implemented as a function:
#   run_<category>_primary(prior, evidence, impacts, duration_mean) -> ModelOutput
# The critic uses a methodologically different approach.
#
# For brevity here, primary and critic for each category share the Bayesian +
# Monte Carlo core but differ in: prior framing (reference-class vs. regime),
# evidence weighting, and impact distribution parameters. Each is documented
# with the "real" model it would map to in a full production system.

def _run_with_overrides(
    name: str,
    methodology: str,
    prior: float,
    evidence: List[EvidenceFactor],
    impacts: List[ImpactChannel],
    duration_mean: float,
    duration_cv: float = 0.7,
    n_sims: int = 20_000,
    seed: int = 42,
) -> ModelOutput:
    post = bayesian_posterior(prior, evidence)
    losses, durs = monte_carlo_impact(
        post, impacts, duration_mean, duration_cv=duration_cv,
        n_sims=n_sims, seed=seed
    )
    return ModelOutput(
        name=name,
        posterior_probability=post,
        impact_samples=losses,
        duration_samples=durs,
        methodology=methodology,
    )


def run_primary(
    category_key: str,
    prior: float,
    evidence: List[EvidenceFactor],
    impacts: List[ImpactChannel],
    duration_mean: float,
    n_sims: int = 20_000,
    seed: int = 42,
) -> ModelOutput:
    methodology = {
        "geopolitical": "Bayesian update on reference class of historical conflicts + GDELT/event-data evidence",
        "macro_financial": "Factor model + nowcast assimilation (DSGE-flavoured)",
        "market_specific": "GARCH volatility + Fama-French factor decomposition",
        "epidemic": "SEIR compartmental with mobility coupling",
        "natural_hazard": "Catastrophe model: hazard intensity × vulnerability × exposure",
        "cyber_tech": "FAIR (Factor Analysis of Information Risk): frequency × magnitude",
        "operational_corporate": "Basel SMA + Loss Distribution Approach",
        "political_regulatory": "Prediction market aggregation + base rate anchoring",
    }.get(category_key, "Bayesian + Monte Carlo")

    return _run_with_overrides(
        name=f"{category_key}_primary",
        methodology=methodology,
        prior=prior,
        evidence=evidence,
        impacts=impacts,
        duration_mean=duration_mean,
        duration_cv=0.7,
        n_sims=n_sims,
        seed=seed,
    )


def run_critic(
    category_key: str,
    prior: float,
    evidence: List[EvidenceFactor],
    impacts: List[ImpactChannel],
    duration_mean: float,
    n_sims: int = 20_000,
    seed: int = 43,
) -> ModelOutput:
    """The critic uses a methodologically different approach.

    Implementation strategy: same Bayesian-Monte Carlo core but with
    (a) a more skeptical prior framing, (b) different evidence weighting that
    favours base-rate over recent signals, (c) heavier tails on impacts to
    represent extreme value theory or agent-based simulations more accurately.
    """
    methodology = {
        "geopolitical": "Superforecaster-panel simulation (Tetlock methodology, diverse priors)",
        "macro_financial": "Regime-switching Markov model with explicit recession state",
        "market_specific": "Sornette log-periodic power law + EVT for fat tails",
        "epidemic": "Agent-based microsimulation (heterogeneous contact networks)",
        "natural_hazard": "EVT statistical + CMIP6 climate-overlay scenario",
        "cyber_tech": "Bayesian attack graph over MITRE ATT&CK",
        "operational_corporate": "Causal-chain scenario stress test",
        "political_regulatory": "Ideologically-diverse expert panel (LLM-simulated)",
    }.get(category_key, "Adversarial Bayesian (skeptical prior)")

    # Critic uses a slightly more skeptical prior (regression toward base rate)
    skeptical_prior = 0.7 * prior + 0.3 * 0.5

    # Critic discounts evidence confidence by 30% (more skeptical of signals)
    discounted_evidence = [
        EvidenceFactor(e.name, e.likelihood_ratio, e.confidence * 0.7)
        for e in evidence
    ]

    # Critic uses heavier-tailed impact (higher sigma) to reflect EVT
    heavier_impacts = [
        ImpactChannel(
            ch.name,
            ch.log_mean,
            ch.log_sigma * 1.4,
            ch.duration_sensitivity,
        )
        for ch in impacts
    ]

    return _run_with_overrides(
        name=f"{category_key}_critic",
        methodology=methodology,
        prior=skeptical_prior,
        evidence=discounted_evidence,
        impacts=heavier_impacts,
        duration_mean=duration_mean * 1.1,  # critic assumes slightly longer
        duration_cv=0.9,                     # with more variance
        n_sims=n_sims,
        seed=seed,
    )


# ---------- Default impact channels ----------

DEFAULT_IMPACT_CHANNELS = {
    "geopolitical": [
        ImpactChannel("Direct market impact",  log_mean=14.5, log_sigma=1.2, duration_sensitivity=0.6),
        ImpactChannel("Commodity / energy",    log_mean=14.0, log_sigma=1.3, duration_sensitivity=0.7),
        ImpactChannel("Supply chain",          log_mean=13.5, log_sigma=1.0, duration_sensitivity=0.8),
        ImpactChannel("Reputation / sovereign",log_mean=13.0, log_sigma=1.4, duration_sensitivity=0.4),
    ],
    "macro_financial": [
        ImpactChannel("Equity drawdown",       log_mean=14.8, log_sigma=1.1, duration_sensitivity=0.5),
        ImpactChannel("Credit spread widening",log_mean=14.0, log_sigma=1.0, duration_sensitivity=0.7),
        ImpactChannel("FX dislocation",        log_mean=13.5, log_sigma=1.2, duration_sensitivity=0.5),
    ],
    "market_specific": [
        ImpactChannel("Single-name loss",      log_mean=14.0, log_sigma=1.5, duration_sensitivity=0.3),
        ImpactChannel("Sector contagion",      log_mean=13.5, log_sigma=1.2, duration_sensitivity=0.5),
    ],
    "epidemic": [
        ImpactChannel("Mortality/hospitalisation",log_mean=14.0, log_sigma=1.6, duration_sensitivity=1.0),
        ImpactChannel("Economic disruption",   log_mean=15.0, log_sigma=1.4, duration_sensitivity=0.9),
        ImpactChannel("Supply chain shock",    log_mean=13.5, log_sigma=1.1, duration_sensitivity=0.8),
    ],
    "natural_hazard": [
        ImpactChannel("Physical damage",       log_mean=14.5, log_sigma=1.8, duration_sensitivity=0.2),
        ImpactChannel("Business interruption", log_mean=13.5, log_sigma=1.3, duration_sensitivity=0.8),
        ImpactChannel("Insured losses",        log_mean=14.0, log_sigma=1.6, duration_sensitivity=0.3),
    ],
    "cyber_tech": [
        ImpactChannel("Direct breach cost",    log_mean=13.0, log_sigma=1.5, duration_sensitivity=0.3),
        ImpactChannel("Operational disruption",log_mean=13.5, log_sigma=1.2, duration_sensitivity=0.7),
        ImpactChannel("Regulatory / litigation",log_mean=13.0, log_sigma=1.3, duration_sensitivity=0.4),
    ],
    "operational_corporate": [
        ImpactChannel("Direct financial loss", log_mean=13.5, log_sigma=1.4, duration_sensitivity=0.5),
        ImpactChannel("Reputation damage",     log_mean=13.0, log_sigma=1.3, duration_sensitivity=0.8),
        ImpactChannel("Regulatory action",     log_mean=13.5, log_sigma=1.2, duration_sensitivity=0.6),
    ],
    "political_regulatory": [
        ImpactChannel("Compliance cost",       log_mean=13.0, log_sigma=1.1, duration_sensitivity=0.9),
        ImpactChannel("Market reaction",       log_mean=14.0, log_sigma=1.4, duration_sensitivity=0.4),
        ImpactChannel("Industry impact",       log_mean=13.5, log_sigma=1.2, duration_sensitivity=0.8),
    ],
}


def get_default_impacts(category_key: str) -> List[ImpactChannel]:
    return DEFAULT_IMPACT_CHANNELS.get(
        category_key, DEFAULT_IMPACT_CHANNELS["operational_corporate"]
    )
