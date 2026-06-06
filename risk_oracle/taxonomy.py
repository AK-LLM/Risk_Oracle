"""
Trigger taxonomy — the 8 categories the router classifies into,
each mapped to a primary model and a methodologically-different critic.
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class CategorySpec:
    name: str
    description: str
    primary_model: str
    critic_model: str
    base_rate_default: float
    typical_duration_months: float
    osint_signals: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    reference_class_note: str = ""


TAXONOMY: Dict[str, CategorySpec] = {
    "geopolitical": CategorySpec(
        name="Geopolitical conflict",
        description="Wars, escalations, sanctions, regime change, alliance shifts",
        primary_model="bayesian_reference_class",
        critic_model="superforecaster_panel",
        base_rate_default=0.55,
        typical_duration_months=18.0,
        osint_signals=["gdelt", "acled", "polymarket", "metaculus"],
        keywords=["war", "conflict", "strike", "invasion", "sanctions",
                  "coup", "election", "regime", "ceasefire", "treaty",
                  "military", "diplomatic", "alliance", "nato", "iran",
                  "israel", "russia", "ukraine", "china", "taiwan"],
        reference_class_note="Regional conflicts with direct strikes — historical "
                             "base rate ~50-70% remain active at 6 months.",
    ),
    "macro_financial": CategorySpec(
        name="Macro-financial",
        description="Recessions, currency crises, rate cycles, sovereign debt",
        primary_model="factor_nowcast",
        critic_model="regime_switching",
        base_rate_default=0.30,
        typical_duration_months=12.0,
        osint_signals=["fred", "world_bank", "polymarket", "fed_speeches"],
        keywords=["recession", "inflation", "rates", "fed", "ecb", "boj",
                  "currency", "sovereign", "default", "yield", "gdp",
                  "unemployment", "cpi", "central bank", "monetary"],
        reference_class_note="Recessions follow yield curve inversions 60-70% "
                             "of the time within 18 months.",
    ),
    "market_specific": CategorySpec(
        name="Market-specific",
        description="Asset price moves, sector shocks, liquidity events, single-name",
        primary_model="garch_factor",
        critic_model="evt_tail",
        base_rate_default=0.20,
        typical_duration_months=3.0,
        osint_signals=["market_data", "polymarket"],
        keywords=["stock", "equity", "bond", "crypto", "bitcoin", "etf",
                  "options", "futures", "earnings", "ipo", "merger",
                  "acquisition", "bankruptcy", "delisting", "ticker"],
        reference_class_note="Sector-wide drawdowns >20% occur roughly every 4 years.",
    ),
    "epidemic": CategorySpec(
        name="Epidemic / biosecurity",
        description="Disease outbreaks, pandemics, zoonotic spillover, bioterrorism",
        primary_model="seir_mobility",
        critic_model="agent_based",
        base_rate_default=0.15,
        typical_duration_months=24.0,
        osint_signals=["healthmap", "promed", "world_bank"],
        keywords=["virus", "outbreak", "pandemic", "ebola", "covid",
                  "influenza", "h5n1", "mers", "marburg", "vaccine",
                  "epidemic", "infection", "transmission", "r0",
                  "hantavirus", "zoonotic"],
        reference_class_note="Novel pathogen with airborne potential reaching a "
                             "major city — historically 20-30% become regional outbreaks.",
    ),
    "natural_hazard": CategorySpec(
        name="Natural hazard",
        description="Earthquakes, hurricanes, floods, wildfires, climate-driven",
        primary_model="cat_model",
        critic_model="evt_climate",
        base_rate_default=0.40,
        typical_duration_months=2.0,
        osint_signals=["usgs", "noaa", "world_bank"],
        keywords=["earthquake", "hurricane", "typhoon", "tsunami", "flood",
                  "wildfire", "tornado", "volcano", "drought", "storm",
                  "climate", "category", "magnitude", "richter"],
        reference_class_note="Major Atlantic hurricane (Cat 3+) US landfall — "
                             "historical base rate ~30% per season.",
    ),
    "cyber_tech": CategorySpec(
        name="Cyber / tech",
        description="Breaches, infrastructure attacks, AI incidents, platform collapse",
        primary_model="fair_frequency_magnitude",
        critic_model="bayesian_attack_graph",
        base_rate_default=0.35,
        typical_duration_months=4.0,
        osint_signals=["cisa_kev", "have_i_been_pwned"],
        keywords=["breach", "ransomware", "exploit", "cve", "vulnerability",
                  "ddos", "phishing", "malware", "zero-day", "attack",
                  "hack", "cyber", "ai", "agi", "platform", "outage"],
        reference_class_note="Fortune 500 firms suffer reportable cyber incidents "
                             "at roughly 40% annual rate.",
    ),
    "operational_corporate": CategorySpec(
        name="Operational / corporate",
        description="Reputational events, supply-chain breaks, fraud, key-person, ESG",
        primary_model="basel_loss_distribution",
        critic_model="scenario_causal_chain",
        base_rate_default=0.25,
        typical_duration_months=8.0,
        osint_signals=["edgar", "world_bank"],
        keywords=["fraud", "scandal", "lawsuit", "recall", "investigation",
                  "ceo", "resignation", "bankruptcy", "supply", "chain",
                  "boycott", "reputation", "esg", "whistleblower"],
        reference_class_note="Large-cap firms experience material adverse events "
                             "at ~15% annual rate.",
    ),
    "political_regulatory": CategorySpec(
        name="Political / regulatory",
        description="Elections, legislation, court rulings, regulatory shifts",
        primary_model="prediction_market_aggregation",
        critic_model="ideological_panel",
        base_rate_default=0.50,
        typical_duration_months=6.0,
        osint_signals=["polymarket", "metaculus", "manifold", "politician_trades"],
        keywords=["election", "vote", "polling", "legislation", "bill",
                  "ruling", "court", "supreme", "regulation", "ban",
                  "approval", "antitrust", "ftc", "sec", "doj", "eu"],
        reference_class_note="Incumbent re-elections in stable democracies — "
                             "historical base rate ~65%.",
    ),
}


CATEGORY_KEYS = list(TAXONOMY.keys())


def get_category(key: str) -> CategorySpec:
    if key not in TAXONOMY:
        raise KeyError(f"Unknown category: {key}. Known: {CATEGORY_KEYS}")
    return TAXONOMY[key]


def all_categories() -> List[CategorySpec]:
    return list(TAXONOMY.values())
