# Risk Oracle

A probabilistic risk modeling system that takes a natural-language trigger event and produces a calibrated probability forecast with confidence band, OSINT verification, and decision support.

Architecture (designed across many conversations):

1. **Smart router** — extracts features from the trigger, classifies it into one of 8 categories, and assigns primary + critic models from different methodological traditions.
2. **Parallel ensemble** — primary model + adversarial critic run independently.
3. **Reconciliation** — Bayesian model averaging weighted by historical calibration; disagreement widens the uncertainty band rather than averaging it away.
4. **OSINT verification** — implied-observables check against multi-source open feeds (GDELT, FRED, Polymarket, prediction markets).
5. **Calibration loop** — every prediction logged with resolution date; Brier-scored on resolution; weights feed back to router and reconcile.
6. **Decision layer** — Kelly-sized position recommendations, sensitivity analysis, value-of-information for next research step.

v2 features included as live modules: cross-category contagion, time dynamics (hazard surface), tail-explicit output, black swan detection, decision layer.

v3 hooks present: prediction market integration (live for Polymarket and Manifold), causal DAG structure (stub interface), active learning (VOI calculation).

## Setup — Local install

```bash
git clone <repo>
cd risk_oracle
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml to add any API keys you have
streamlit run app.py
```

Opens at http://localhost:8501. SQLite calibration database is created automatically at `~/.risk_oracle/calibration.db`.

## Setup — Streamlit Community Cloud

1. Push this directory to a GitHub repo (public or private).
2. Go to https://share.streamlit.io and connect the repo.
3. In the Streamlit Cloud app settings, under **Secrets**, paste the contents of `.streamlit/secrets.toml.example` and fill in any keys you have.
4. Deploy.

**Important caveat for Streamlit Community**: the calibration database is **ephemeral** — it resets when the app restarts. For persistent calibration data on Community Cloud, either (a) export the DB regularly using the in-app Download button on the Calibration page, or (b) point at an external database (Supabase free tier works; set `EXTERNAL_DB_URL` in secrets).

## API keys (all optional — system works without them)

The system is designed to **degrade gracefully**. Every external API is optional; missing keys fall back to cached or rule-based behavior.

| Key | Purpose | Where to get | Cost |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | LLM-powered router (smart classification) | https://console.anthropic.com | Pay-as-you-go, fractions of a cent per query |
| `OPENAI_API_KEY` | Alternative LLM for router | https://platform.openai.com | Pay-as-you-go |
| `FRED_API_KEY` | Macro data from Federal Reserve | https://fred.stlouisfed.org/docs/api/api_key.html | Free |
| `ACLED_API_KEY` / `ACLED_EMAIL` | Conflict event data | https://acleddata.com/register | Free for non-commercial |

Without any keys, the router falls back to keyword-based routing, and OSINT uses only free unauthenticated sources (GDELT, World Bank, Polymarket, Manifold, SEC EDGAR, NOAA).

## Project layout

```
risk_oracle/
├── app.py                   Streamlit entrypoint
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example
└── risk_oracle/
    ├── taxonomy.py          8 trigger categories + model assignments
    ├── router.py            Feature extraction + routing logic
    ├── models.py            Bayesian update, Monte Carlo, primary+critic per category
    ├── reconcile.py         Model averaging + disagreement-as-uncertainty
    ├── osint.py             OSINT API integrations (GDELT, FRED, prediction markets, etc.)
    ├── contagion.py         Cross-category contagion engine
    ├── tail_risk.py         EVT / tail-explicit output
    ├── decision.py          Kelly sizing + sensitivity + VOI
    ├── calibration.py       SQLite store, Brier scoring, weight updates
    └── visualization.py     Plotly charts
```

## How to use

1. Open the app. Default tab is **New forecast**.
2. Enter a precise, dated trigger question. Good: *"Will direct Iran-Israel strikes still be occurring on October 1, 2026?"*. Bad: *"Will there be war?"*.
3. Optionally adjust the prior probability and evidence factors.
4. Hit **Run forecast**. The system routes, runs primary + critic, reconciles, queries OSINT, and produces output with confidence band, tail risk, and decision recommendation.
5. The forecast is auto-logged. Set the expected resolution date and the system will prompt you to score it later.
6. **Calibration** tab shows your Brier scores by category, which feeds back into model weighting.

## Honest caveats

- This system is a discipline, not an oracle. On any single novel high-stakes question it will not consistently hit 75–80% accuracy. Nothing does. What it provides is well-calibrated probabilities over hundreds of predictions if you run the loop.
- Calibration only emerges after ~50 resolved predictions per category. Be patient.
- All the underlying models are simplified versions of the real methods. Production-grade SEIR, DSGE, catastrophe modeling, GARCH etc. require dedicated libraries and tuning. The architecture is right; the underlying models are illustrative until you swap in domain-specific implementations.
- This is not financial, legal, or medical advice. It is a tool for thinking.

## License

MIT.
