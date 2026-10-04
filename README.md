# Football Prediction Engine

A hybrid **statistical + LLM** football betting prediction system built from the
architecture discussed in the design conversation. It combines specialized
worker models, supervisory auditors, a master-brain meta-controller, and an
optional DeepSeek LLM expert — then scans the full market universe to output
only the single lowest-risk selection per fixture.

> ⚠️ **Honest expectations (read this first).** No system can guarantee 98–99%
> accuracy on football. Even the best models in the world don't achieve that —
> football has irreducible randomness (red cards, VAR, injuries). This engine is
> built to **maximize precision through abstention**: it skips matches it isn't
> sure about, calibrates its probabilities, and only releases picks above strict
> thresholds. That is the closest a machine can get to "bulletproof" — not a
> guarantee, but a disciplined edge. Anyone promising 100% hit rates is lying.

## Architecture

```
Raw match data (API or synthetic)
        │
        ▼
Feature engineering (Elo, EMA xG, H2H, venue fortress, KPAI, game-state)
        │
        ▼
Worker models (goals -> Dixon-Coles Poisson, corners, cards)
        │
        ▼
Auditors (IsolationForest anomaly, variance, calibration)
        │
        ▼
Master Brain (meta-controller: veto rules + mix statistical & DeepSeek)
        │
        ▼
Universal market scanner -> single best selection per fixture
        │
        ▼
Accumulator builder (independence checks, ALL-OR-NOTHING ticket)
```

## Project layout

```
config/config.yaml            # all thresholds & settings
prompts/system_prompt.md      # DeepSeek expert system prompt
src/
  config.py                   # YAML + env config loader
  data/                       # ingestion (Odds API, API-Football) + synthetic fallback
  features/                   # elo, rolling EMA, h2h, venue, injuries, game_state, builder
  models/                     # workers, poisson (Dixon-Coles), calibration, conformal,
                              # anomaly, master_brain
  llm/                        # DeepSeek client, prompts, expert, fine-tuning
  markets/                    # full market registry, universal scanner, accumulator
  pipeline/                   # training, time-series validation, shadow backtest
  utils/                      # logging, notifications
  main_pipeline.py            # daily entry point
scripts/
  train.py                    # train the engine
  run_daily.py                # run today's prediction
  backtest.py                 # forward-only shadow backtest
  fine_tune_deepseek.py       # submit DeepSeek fine-tuning job
.github/workflows/            # GitHub Actions (heavy lifting runs on runners)
tests/                        # pytest suite
```

## Local vs GitHub Actions

| Concern | Local (your machine) | GitHub Actions (runner) |
|---|---|---|
| Disk | tiny — code only | full deps installed fresh |
| Model backend | auto-falls back to sklearn `HistGradientBoosting` | xgboost/lightgbm |
| Training | small synthetic | full synthetic or real data |
| LLM | optional, needs key | optional, needs key |
| Output | `output/` (gitignored) | uploaded as artifacts |

The model backend is resolved automatically (`src/models/base.py`): xgboost →
lightgbm → sklearn HistGradientBoosting. No heavy packages are installed
locally.

## Quick start (local, no keys needed)

```bash
# 1. Train the engine on synthetic data
python3 scripts/train.py

# 2. Run today's prediction (mock fixtures)
python3 scripts/run_daily.py

# 3. Forward-only backtest (honest precision/abstention metrics)
python3 scripts/backtest.py

# 4. Run the test suite
python3 -m pytest tests/ -q
```

## GitHub Actions

The workflow `.github/workflows/daily_prediction_pipeline.yml` runs daily at
06:00 UTC and on manual dispatch. It installs the full stack on the runner,
trains, predicts, uploads artifacts, and sends notifications.

**Secrets to configure** (Settings → Secrets and variables → Actions):

| Secret | Purpose |
|---|---|
| `ODDS_API_KEY` / `ODDS_API_KEYS` | The Odds API — fallback odds (comma-separate for rotation) |
| `FOOTBALL_API_KEY` / `FOOTBALL_API_KEYS` | API-Football — fixtures, lineups, injuries, history |
| `ODDSPAPI_KEY` / `ODDSPAPI_KEYS` | OddsPapi — PRIMARY odds + free historical odds |
| `DEEPSEEK_API_KEY` | DeepSeek LLM expert + fine-tuning |
| `DEEPSEEK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` (or your fine-tuned id) |
| `DEEPSEEK_BASE_URL` | `https://inference.dahl.global/v1` |
| `DISCORD_WEBHOOK_URL` | optional ticket notifications |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | optional Telegram notifications |

## DeepSeek hybrid (optional)

DeepSeek V4 Flash is used as a **specialist expert, not for every prediction**,
via the OpenAI-compatible inference endpoint `https://inference.dahl.global/v1`
(model `deepseek-ai/DeepSeek-V4-Flash-0731`):

- The statistical workers handle the structured data (goals, xG, corners, cards).
- DeepSeek is only invoked for fixtures that pass the statistical gate.
- Its structured probabilities are blended with the workers by the master
  brain (`llm.mix_weight`, default 0.3 LLM / 0.7 statistical).
- **350k context window**: with `llm.rich_context: true`, the expert receives
  the RAW data — last-8 form for both teams, H2H history, injury/lineup
  availability and odds — not just a compressed summary.
- **300k output budget**: `llm.max_tokens` (default 8000) lets it return a full
  per-market probability breakdown with reasoning.

Verify your key first:

```bash
export DEEPSEEK_API_KEY=sk-...
python3 scripts/test_deepseek.py
```

> **Fine-tuning note:** the `dahl.global` endpoint is inference-only. The
> fine-tuning script (`scripts/fine_tune_deepseek.py`) targets the official
> DeepSeek platform (`https://api.deepseek.com`) — you need a DeepSeek account
> there to fine-tune. If your provider supports fine-tuning on a different
> endpoint, set `llm.fine_tune.base_url` in `config/config.yaml`.

## Key design decisions

- **Leakage-safe features**: every match's features use only data available
  before its kickoff.
- **Expanding-window validation**: never standard K-Fold on sequential data.
- **Dixon-Coles Poisson**: corrects the low-score bias of plain Poisson.
- **Conformal prediction** (implemented from scratch): mathematically bounded
  coverage; wide prediction sets → abstain. The risk gate runs at
  `alpha <= 0.001` (99.9% coverage) — if the prediction set spans more than
  one outcome class, the fixture is skipped.
- **Abstention is success**: the engine is allowed to skip everything.
- **One selection per fixture**: the scanner picks the single lowest-risk line.
- **ALL-OR-NOTHING accumulator**: the ticket is a straight accumulator — if
  ANY leg fails, the whole ticket is a loss. System-bet splits are off by
  default (`pipeline.system_bets: false`) because they hedge against that rule.

### Zero-cut execution (0.985 hard floor)

`pipeline.min_confidence` defaults to **0.985** — a single leg fails only
~1.5% of the time. The engine would rather output a **lower-odds ticket (or
none)** than dilute this floor to force 300 odds. Nothing below the floor
ever ships:

- `markets.thresholds.high_floor: 0.985` — ultra-safe markets (team Over 0.5
  goals, Asian Handicap +2.5, corners Over 6.5) must clear this before EXECUTE.
- `markets.thresholds.primary: 0.90` — other markets clear here, but the
  accumulator still filters them out at 0.985.
- `models.conformal.alpha: 0.001` — the conformal risk gate abstains unless
  the outcome is essentially certain.

### Target-odds optimizer (over-select → trim to ~300)

The builder **over-selects** every qualified independent fixture, then brings
the combined odds down toward the target (default 300) using a safety-first
strategy:

1. **Downgrade** every leg to its leanest variant (e.g. `Home Win` →
   `Double Chance 1X` → `Double Chance 12`), which lowers odds and raises
   confidence.
2. If still above the upper band, **remove** the weakest legs.
3. If that overshoots the lower band, **selectively upgrade** legs back up
   the odds ladder to land as close to 300 as possible.

If the pool can't reach 300 without breaking the 0.985 floor, the engine
keeps everything it cleared safely and reports `target_reached: False` — it
never forces an inaccurate pick just to hit a number.

### Basketball to even the edges

Basketball (moneyline, spreads, totals, team totals, player props) is loaded
alongside football so the engine can reach 300+ odds without forcing
low-confidence football picks. Both sports share the same master-brain
pipeline and the same 0.985 floor.

### Structured Telegram notification

The daily ticket is sent to Telegram (and Discord) as a structured message —
one block per selection with **clubs, kickoff time, country, league, sport,
market, odds and confidence**. Only matches that have **not started** are
included (the pipeline filters out any kickoff before the run time).

## Real data (when you add keys)

With keys set, the `DataLoader` pulls REAL data instead of synthetic:

- **API-Football** → historical fixtures + per-match statistics (xG, corners,
  cards) for training, plus today's fixtures. Cache-aware (saved to
  `data/raw/`) and budget-aware (free tier = 100 req/day per key).
  The free plan only covers **seasons 2022–2024**, so the pipeline trains on
  **all three** (`data.football_api.seasons`) — ~1,140 EPL-style matches —
  instead of a single season.
- **OddsPapi** (primary odds) → current + **free historical odds** with
  line-movement analysis, 300+ bookmakers. This is what trains the edge model.
  Because API-Football's free plan can't access the **current season**, today's
  fixtures fall back to OddsPapi (`build_oddspapi_today_fixtures`), which has
  the current season's fixtures + odds.
- **The Odds API** (fallback odds) → current odds if OddsPapi is unavailable.

### Multi-key rotation

Every provider supports **multiple keys** — comma-separate them and the client
rotates automatically when one hits its daily budget:

```bash
export FOOTBALL_API_KEYS="key1,key2,key3,key4,key5"
export ODDSPAPI_KEYS="key1,key2,key3,key4,key5"
export ODDS_API_KEYS="key1,key2,key3,key4,key5"
```

5 keys ≈ 5x the free-tier quota. Per-key usage is tracked in `data/raw/`.

Verify all keys:

```bash
export FOOTBALL_API_KEY=... ODDS_API_KEY=... ODDSPAPI_KEY=...
python3 scripts/test_apis.py
```

Build the historical-odds edge dataset (OddsPapi free historical odds):

```bash
python3 scripts/build_historical_odds.py --limit 20
```

## Limitations

- The 300+ odds target requires many independent legs at the 0.985 confidence
  floor; on most days the engine will report `target_reached: False` and that
  is correct — it refuses to force low-confidence picks.
- No system can guarantee 100% hit rate — anyone claiming otherwise is lying.

