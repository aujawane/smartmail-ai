# SmartMail AI — Team Agent Playbook

> **Authoritative instructions for all six teammates and their coding agents.**
> Project: Intro to AI | Delivery window: **14 calendar days** | Version: **1.0 (2026-11-16)**
> Read this entire file **before creating, changing, or deleting code**.

## 0. Mission and non-negotiable scope

Build a **local-demo-ready email spam classifier** that compares (1) a **fine-tuned DistilBERT** model against (2) a **Multinomial Naive Bayes baseline**, using the **same dataset split** and a React interface connected to a FastAPI backend. Report held-out accuracy, precision, recall, F1, false-positive rate, and inference time. A working, reproducible experiment is more important than extra features.

**Must ship by Day 14**
1. A reproducible email dataset pipeline with a fixed train / validation / test split.
2. A fine-tuned DistilBERT binary classifier with saved inference artifacts.
3. A Naive Bayes + TF-IDF baseline with saved inference artifacts.
4. A single prediction API serving both models, including graceful unavailable states.
5. A simple React demo with pasted-email input, side-by-side predictions, and a metric comparison view.
6. A test-set evaluation report, confusion matrices, setup instructions, and an honest presentation of limitations.

**Not in the 14-day scope:** Gmail integration, live inbox access, email deletion, user accounts, database, continuous retraining, URL reputation lookups, browser extension, deploying to production, complex explainability dashboard. If ahead of schedule, Srishti can add a **separate** token-masking explanation experiment, not a blocker to release.

**Academic integrity:** We fine-tune pretrained weights; we do not claim to have invented or pretrained the transformer. Confirm that pretrained models are allowed for the course. If not permitted, immediately escalate to the team before investing in DistilBERT.

## 1. Who owns what (strict file ownership)

| Person | Workstream | Only owner of these paths |
| --- | --- | --- |
| **Aditya** | Tech lead, DistilBERT, architecture, integration decisions | `src/smartmail/models/distilbert/`, `tests/distilbert/`, `configs/distilbert.yaml`, `src/smartmail/contracts.py`, `pyproject.toml`, `AGENTS.md`, root `README.md`, `.gitignore` |
| **Srishti** | Evaluation, error analysis, optional explainability | `src/smartmail/evaluation/`, `tests/evaluation/`, `reports/`, `scripts/evaluation/` |
| **Eduardo Valdovinos** | Dataset, preprocessing, stable splits | `src/smartmail/data/`, `tests/data/`, `scripts/data/`, `configs/dataset.yaml` |
| **Dion** | Naive Bayes + TF-IDF baseline | `src/smartmail/models/naive_bayes/`, `tests/naive_bayes/`, `configs/naive_bayes.yaml` |
| **Luis Meza** | FastAPI, adapters, end-to-end integration | `src/smartmail/api/`, `tests/api/`, `scripts/api/` |
| **Madeline Cohen** | React frontend, charts and user experience | `frontend/` (including its tests, package files, and assets) |

**Shared files are not a free-for-all.** Only Aditya edits the root contract, packaging, and instructions. To request a change to a shared contract, open an issue or leave a comment in the team channel; Aditya updates this document and the contract before any dependent agent changes code. Luis coordinates API integration, but does not rewrite model code. Everyone owns tests for their own area.

**Aditya scaffolds the repository on Day 1**, including empty package `__init__.py` files, base environment, `src/smartmail/contracts.py`, and the basic API contract. This is the only cross-area bootstrap pass. After that, follow ownership strictly. When a missing shared file blocks you, request it; do not independently create it.

## 2. Architecture and directory contract

```text
smartmail-ai/
├── AGENTS.md                         # this file: source of truth (Aditya)
├── README.md                         # setup, run, demo (Aditya)
├── pyproject.toml                     # Python dependencies/tool config (Aditya)
├── .gitignore                         # ignore data, weights, secrets (Aditya)
├── configs/
│   ├── dataset.yaml                  # Eduardo
│   ├── distilbert.yaml               # Aditya
│   └── naive_bayes.yaml              # Dion
├── src/smartmail/
│   ├── __init__.py
│   ├── contracts.py                   # interfaces + shared labels (Aditya)
│   ├── data/                          # Eduardo
│   ├── models/
│   │   ├── distilbert/                # Aditya
│   │   └── naive_bayes/              # Dion
│   ├── evaluation/                    # Srishti
│   └── api/                           # Luis
├── scripts/
│   ├── data/                          # Eduardo
│   ├── evaluation/                    # Srishti
│   └── api/                           # Luis
├── tests/
│   ├── data/                          # Eduardo
│   ├── distilbert/                    # Aditya
│   ├── naive_bayes/                   # Dion
│   ├── evaluation/                    # Srishti
│   └── api/                           # Luis
├── frontend/                          # Madeline (Vite, React, TypeScript)
├── reports/                           # Srishti: small, derived reports only
├── data/                               # LOCAL ONLY — gitignored
└── artifacts/                          # LOCAL ONLY — gitignored
```

**Language and stack:** Python 3.11; PyTorch + Transformers + Datasets for DistilBERT; scikit-learn for TF-IDF and Naive Bayes; FastAPI + Uvicorn for backend; React + TypeScript + Vite for frontend; pytest + Ruff for Python tests/lint. Keep dependencies in **one** `pyproject.toml` controlled by Aditya. No Docker or cloud deployment is required for the demo.

**Integration principle:** Train and evaluate with local scripts, persist artifacts outside Git, and load them into a locally running API. The UI must work with mock response fixtures before real weights are available. The entire project must remain demonstrable even if the transformer checkpoint is temporarily unavailable.

## 3. Frozen data contract — used by every workstream

**Canonical labels:** `0 = ham` (legitimate), `1 = spam`. For JSON, spell labels as lowercase strings: `"ham"` and `"spam"`. Never swap this mapping.

**Input:** One string field called `text`, containing human-readable email content. The frontend has one paste-email textarea; do **not** add separate subject/body inputs during this sprint. Preserve any pasted subject line as part of the text.

**Processed dataset files:** `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`. Each has exactly these required columns:

| Column | Type | Contract |
| --- | --- | --- |
| `sample_id` | string | Stable, unique deterministic hash/ID for each email after deduplication |
| `text` | string | Readable, nonempty plain text; not MIME JSON; consistent with API input |
| `label` | int | `0` for ham, `1` for spam |

Optional extra metadata can live in a **separate** sidecar file, not these three training CSVs. The split should be 80/10/10, stratified, fixed `seed=42`, with **exact duplicate text removed before splitting**. Check for near duplicates if time permits. Do **not** tune against test data. Eduardo exports a small `data/processed/manifest.json` containing source identifier, dataset revision/commit if known, split seed, label counts and text-cleaning version.

**Source:** Start with `talby/spamassassin` (Hugging Face; its published view has `label`, `group`, `text`, and only a `train` split). The published card notes that the `text` configuration may encode MIME content structurally; **inspect actual rows first**, then extract human-readable plain text instead of assuming `text` is already clean body text. Confirm label mapping from dataset metadata, not from guesswork. If that path causes difficulties, Eduardo proposes a documented alternative immediately; do not silently replace the corpus.

**Privacy and provenance:** Do not commit raw emails, processed corpora, checkpoints, tokens, credentials, or real inbox messages. The public corpus has ambiguous licensing information: check usage and attribution before redistribution. For screenshots, tests and demos use **synthetic** sample emails. Do not log full pasted email text in API requests.

**Preprocessing rule:** Preserve text meaning. Normalization may strip surrounding whitespace and normalize line endings, but **do not** globally remove stopwords or stem words before DistilBERT. Naive Bayes owns its own vectorizer preprocessing. Both models train and infer from the same canonical readable `text` field.

## 4. Frozen Python model interface

Aditya defines `src/smartmail/contracts.py` on Day 1. Both model teams must implement the exact callable contract below. Do not introduce a new naming convention or import path without approval.

```python
from pathlib import Path
from typing import Protocol

HAM = 0
SPAM = 1

class SpamPredictor(Protocol):
    def predict_spam_probability(self, text: str) -> float:
        """Return P(spam) as a Python float in [0.0, 1.0]."""
```

**Required entry points for both model packages:**

```python
# src/smartmail/models/distilbert/__init__.py
# src/smartmail/models/naive_bayes/__init__.py
from pathlib import Path
from smartmail.contracts import SpamPredictor

def load_predictor(artifact_dir: Path) -> SpamPredictor:
    """Load LOCAL saved artifacts and return a ready inference object."""
```

Each `load_predictor` must throw a **clear, typed exception** if artifacts are missing or incompatible, and must not download large files on import. Each `predict_spam_probability` must validate nonempty input, avoid training, and return a finite value in `[0, 1]`. Thresholding is **centralized in the API** as `spam if p >= 0.5 else ham`; no team independently changes thresholds this sprint. The returned model score is a probability-like output; do not call it a calibrated confidence unless calibration is actually evaluated.

**Artifacts:**
- DistilBERT saves a Transformers-compatible model/tokenizer under `artifacts/distilbert/` plus minimal metadata.
- Naive Bayes saves a single versioned vectorizer/classifier bundle under `artifacts/naive_bayes/` plus metadata.
- Neither directory is committed to Git. Share checkpoints through a team-approved private drive and keep a SHA256 checksum + instructions in `README.md`; never embed machine-specific absolute paths.

**Training scripts:** Within each model-owned directory, create `train.py` with a CLI that reads the fixed CSVs, produces local artifacts, and prints validation metrics. Do not build interactive notebooks as the only training path. Data loading must not mutate another team's CSV files.

## 5. Frozen HTTP API contract — for Luis and Madeline

Base URL during local development: `http://localhost:8000`; all routes below are under `/api/v1`. **No authentication, database, user profiles, or email persistence** in the MVP.

### GET `/api/v1/health`
Response `200`:

```json
{"status":"ok","models":{"distilbert":"ready","naive_bayes":"unavailable"}}
```

Allowed model states: `"ready"` or `"unavailable"`. The example is illustrative; the health route must reflect actual loaded artifacts.

### POST `/api/v1/predict`
Request:

```json
{"text":"Subject: Verify your account\nPlease click here immediately."}
```

Response `200` even if one or both models are unavailable (as long as request is valid):

```json
{
  "models": {
    "distilbert": {"status":"ready","label":"spam","spam_probability":0.932},
    "naive_bayes": {"status":"unavailable","label":null,"spam_probability":null}
  }
}
```

**Always include both model keys.** `status=unavailable` requires `label=null` and `spam_probability=null`. `status=ready` requires `label` one of `"ham"`/`"spam"`, and `spam_probability` a number from `0` to `1`. Numbers shown above are mock values, not trained results. Validation errors (missing/empty text, over-limit input) return `422` with FastAPI validation format. Cap input length at **20,000 characters** at the API boundary, and use explicit truncation to the model token limit internally. Avoid logging raw email content.

### GET `/api/v1/metrics`
Response `200`:

```json
{
  "dataset_version": "unavailable",
  "models": {
    "distilbert": {"status":"unavailable","accuracy":null,"precision":null,"recall":null,"f1":null,"false_positive_rate":null,"avg_inference_ms":null},
    "naive_bayes": {"status":"unavailable","accuracy":null,"precision":null,"recall":null,"f1":null,"false_positive_rate":null,"avg_inference_ms":null}
  }
}
```

Srishti generates small, committed `reports/metrics.json` following this shape after final test evaluation. Luis's endpoint reads that file; if missing, serve the documented unavailable defaults. Madeline must build against the shapes above using **static frontend fixtures**. Do not block UI development waiting on the backend.

**CORS:** Allow only the localhost frontend origin configured for Vite during development. Avoid wildcard CORS with credentials.

## 6. Evaluation protocol — Srishti owns the results

- Both models use **the same exact fixed test.csv**; never compare different splits.
- Accuracy, precision, recall, F1, confusion matrix and false-positive rate are required.
- Define **spam as the positive class (1)** and `FPR = FP / (FP + TN)`; make class counts explicit.
- Report inference latency on the **same local machine**, after a warm-up, using the same set of samples. Label it as environment-specific.
- Include at least 10 investigated misclassifications (can be summarized without publishing raw messages) and discuss dataset age/domain bias and truncation.
- Do not make up performance numbers, force a target accuracy, cherry-pick examples, or use test-set results to select hyperparameters.
- Prefer deterministic runs and log seed, dataset revision, model/checkpoint IDs, training settings and runtime environment.
- Optional: token masking/occlusion analysis can provide an approximate explanation; it is **not** ground-truth model reasoning.

**Srishti owns generated `reports/metrics.json`** and small markdown/PNG reports. Larger run logs are local only. Luis consumes that JSON unchanged, and Madeline consumes the HTTP endpoint unchanged.

## 7. Per-person acceptance criteria

**Aditya — DistilBERT and technical lead**
- [ ] Bootstrap root repo and freeze contracts by end of Day 1.
- [ ] Fine-tune `distilbert/distilbert-base-uncased` as binary classification with `id2label={0:"ham",1:"spam"}`.
- [ ] Use `AutoTokenizer`, `AutoModelForSequenceClassification`, and `Trainer` (or an equivalent explicit PyTorch loop); document setup.
- [ ] Start with `max_length=256`, LR `2e-5`, batch 8, 2 epochs, seed 42; adjust on **validation** only.
- [ ] Save loadable local artifacts and implement `load_predictor` / `predict_spam_probability`.
- [ ] Smoke-test CPU inference on one sample and document GPU training steps.
- [ ] Own integration decisions, dependency changes, release checklist and final README.

**Srishti — Evaluation and interpretability**
- [ ] Implement pure evaluation functions that consume labels + probabilities.
- [ ] Verify F1/FPR/confusion matrix with unit tests on known toy examples.
- [ ] Evaluate both models on test split after training is frozen.
- [ ] Publish `reports/metrics.json`, a concise `reports/findings.md`, and confusion matrix images.
- [ ] Document failure examples, limitations, and a fair comparison.
- [ ] If time remains, add a contained token-masking experiment in her own folder.

**Eduardo — Data pipeline**
- [ ] Inspect real dataset structure and label mapping before parsing.
- [ ] Produce canonical readable `text` values and consistent `sample_id`.
- [ ] Deduplicate and create deterministic 80/10/10 stratified splits.
- [ ] Commit data preparation code/config/tests, **not** dataset CSVs.
- [ ] Provide a manifest and exact reproducibility command in the team channel.
- [ ] Deliver a **synthetic tiny fixture** for offline tests and UI demos.

**Dion — Naive Bayes baseline**
- [ ] Implement TF-IDF + `MultinomialNB` using scikit-learn.
- [ ] Train on Eduardo's train split, select settings on validation, never test.
- [ ] Implement the same `load_predictor` and probability interface.
- [ ] Save a versioned local bundle; make loading fail cleanly without it.
- [ ] Unit-test probabilities, model reload equivalence and empty input behavior.
- [ ] Document the model's assumptions and training command.

**Luis — Backend and integration**
- [ ] Create FastAPI routes exactly as in Section 5.
- [ ] Load both models at startup, independently, without failing app startup if one is absent.
- [ ] Validate max length and empty text, no raw email logging.
- [ ] Add mock predictors and API tests so the backend works before real models.
- [ ] Implement metrics JSON reader and sensible missing-file behavior.
- [ ] Provide a single documented local start command.

**Madeline — Frontend and UX**
- [ ] React + TypeScript + Vite under `frontend/` only.
- [ ] Single input textarea, Analyze action, loading/error states, accessible labels.
- [ ] Side-by-side model statuses/results (explicit unavailable state).
- [ ] Metrics comparison using `/api/v1/metrics`; clearly label missing metrics.
- [ ] Build using typed mock fixtures with the exact Section 5 response contract.
- [ ] Demo gracefully if backend or a model is not yet available.

## 8. Git rules to prevent conflicts

1. **One branch per person/workstream**, created from latest `main`:
   - `feat/aditya-distilbert`
   - `feat/srishti-evaluation`
   - `feat/eduardo-data`
   - `feat/dion-naive-bayes`
   - `feat/luis-api`
   - `feat/madeline-frontend`
2. Never work directly on `main`. Protect `main` with PR reviews if the GitHub plan permits; no force-pushes to main.
3. **Do not edit another owner's paths**, even for a quick fix. Open an issue/PR comment and tag the owner. Agent must stop rather than cross boundaries.
4. **No agent may change contracts**, folder paths, labels, HTTP JSON shapes, dependency pins, API URLs or dataset split policy without explicit agreement and a versioned update here by Aditya.
5. PRs should be focused and small. Prefer one PR per milestone or roughly <300 changed lines excluding generated/test fixtures; large unavoidable training changes are acceptable with review.
6. Before PR: update from latest `main`, run tests for your code, check `git diff --name-only`, and verify no files outside your owned paths are modified. If conflicts appear, **do not auto-resolve by deleting others' code**; coordinate with owner.
7. **Never autoformat the entire repository** and never commit generated data, huge checkpoints, caches, environment files, `.env`, or notebook output.
8. Do not automatically merge PRs, rewrite history, alter GitHub branch protections, or deploy. Have a teammate review; Aditya merges after required checks.
9. Every PR description includes: purpose, files changed, commands run/results, dependencies, and blockers. Add a short demo screenshot only for UI PRs.
10. Asynchronous integration: Luis merges against **fixed interfaces**, not by editing models. Madeline merges against fixed HTTP contract, not by editing the API.

**Daily sync (10 minutes):** Every person posts: `Yesterday | Today | Blocker | Contract change requested?`. A blocker impacting a frozen interface must be raised the **same day**. Aditya updates this file before work diverges.

## 9. Exact 14-day integration schedule

| Day | Definition of done | Main owners |
| --- | --- | --- |
| **1** | Repo skeleton, contracts, setup commands, role branches and synthetic fixture specification frozen | Aditya + all |
| **2** | Dataset parsing proof, NB training proof, DistilBERT tokenizer/model smoke test; frontend/API mocks working | Eduardo, Dion, Aditya, Luis, Madeline |
| **3–4** | Fixed train/val/test CSV pipeline, baseline train/eval, first DistilBERT fine-tuning run; UI and API independently functional | Eduardo, Dion, Aditya, Luis, Madeline |
| **5** | **Integration checkpoint A:** API can load baseline and respond to `/predict`; front end consumes mock HTTP contract | Luis, Dion, Madeline |
| **6–7** | DistilBERT checkpoint loadable; API serves both models; evaluation functions and tests passing | Aditya, Luis, Srishti |
| **8** | **MVP freeze:** real end-to-end demo for two models; any missing feature gets cut, not expanded | All |
| **9–10** | Final validation-selected model settings, reproducible training commands, first full evaluation and charts | Aditya, Dion, Srishti, Eduardo |
| **11** | **Feature freeze:** no new features or architecture changes; bug fixes only | All |
| **12** | Complete test suite, README reproduction instructions, failure analysis and slide screenshots | All |
| **13** | Full rehearsal on one designated demo laptop; rehearse offline recovery/fallback | All |
| **14** | Final deliverable and presentation; no risky changes | All |

**If behind by Day 8:** Prioritize (1) model comparison with reproducible metrics, (2) working API, (3) simple UI. Cut explainability extras, complex charts, deployment and polish. A defensible research result with a functional plain UI wins over an unfinished fancy product.

## 10. Commands and testing gates

After Aditya scaffolds `pyproject.toml`, the standard root workflow should work:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tests
```

**Data prep:** Eduardo documents a runnable entry command such as `python -m smartmail.data.prepare` that writes local `data/processed/*.csv`. **Train:** Aditya/Dion document model-specific `python -m smartmail.models.<name>.train` commands. **Serve:** Luis documents `uvicorn smartmail.api.main:app --reload` from the repository root. **Frontend:** Madeline documents `cd frontend && npm install && npm run dev` and `npm run build`. These module/command names are the desired targets; owners implement them and coordinate any deviation through Section 8.

**Minimum test gates before merge:**
- A data unit test proves no `sample_id` overlap among splits and only valid labels.
- A test for each model proves `predict_spam_probability` returns a float in `[0,1]`, and missing artifact loading fails explicitly.
- API tests cover a valid request, empty input, too long input, no models available, and one model available.
- Evaluation tests prove accuracy, F1 and FPR on a toy confusion matrix.
- Frontend typecheck/build succeeds; UI handles loading, unavailable, and API error states.
- Integration smoke test pastes a synthetic email into the website and receives two structured model results.

**CI:** Nice to have after first integration. Do not spend Day 1 building elaborate CI. Basic Python tests/lint plus frontend build are enough.

## 11. Agent operating protocol — paste into every agent session

> You are coding on the SmartMail AI team. **Read the repository root `AGENTS.md` entirely first; it is authoritative.** Identify the current teammate and assigned owned paths from Section 1. Before writing code, state: (a) my assigned scope, (b) the frozen interfaces I must respect, (c) exactly which files I plan to modify, and (d) the tests I will run. Work **only** inside my owned paths; do not modify shared contracts, another member's files, architecture, dependencies, labels, response schemas or Git workflow. If blocked by a missing interface or dependency, create a clear request for the owning teammate instead of implementing a conflicting workaround. Use mock fixtures to work independently until integration. Do not commit data, checkpoint files, tokens or secrets. Before finishing, run relevant tests, inspect the diff for out-of-scope files, and summarize changes, remaining issues, and handoff requirements. **Do not push, merge, or deploy without explicit human approval.**

**Teammate-specific startup prompt:** `I am [name]. My ownership is [paths from Section 1]. Help me implement [one clear acceptance criterion from Section 7]. Read AGENTS.md before proposing edits.`

**Ownership check before any change:**
1. Which teammate am I working for?
2. Is every file I want to edit in that teammate's assigned paths?
3. Does the change preserve dataset columns, labels, Python interface and HTTP contract?
4. Can I use a fixture/mock instead of changing someone else's component?
5. Have I communicated a blocking contract issue to Aditya?

If any answer is uncertain, **stop and ask the teammate**. Do not silently guess or refactor across the repo.

## 12. Definition of done (Day 14)

A human can clone the repo, install dependencies, prepare the public dataset, reproduce saved-model training (GPU as available), run backend and frontend locally, paste a synthetic email, see both predictions, and inspect evaluation metrics generated from a held-out test set. All six contributions are visible, relevant automated tests pass, and the presentation accurately describes transfer learning, the Naive Bayes baseline, results, limitations and what was actually implemented.

**No invented metrics. No hidden test leakage. No raw-email or checkpoint commits. No surprise interface changes.**

## 13. Reference links (for maintainers)

- DistilBERT model: https://huggingface.co/distilbert/distilbert-base-uncased
- Hugging Face text classification: https://huggingface.co/docs/transformers/tasks/sequence_classification
- SpamAssassin dataset and schema: https://huggingface.co/datasets/talby/spamassassin
- SpamAssassin dataset card: https://huggingface.co/datasets/talby/spamassassin/blob/main/README.md

These links are reference documentation; this file's contracts take precedence for repository integration.
