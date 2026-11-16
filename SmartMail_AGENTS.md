# SmartMail AI | Instructions for every coding agent

**Read this file and `MASTER_PLAN.md` before editing anything.** `MASTER_PLAN.md` v1.0 (November 16, 2026) is the authoritative specification for scope, responsibilities, contracts, milestones, and success criteria. These instructions implement that plan; they do not replace it. If the two documents disagree, report the discrepancy and follow `MASTER_PLAN.md` until the team approves a change.

## 1. Mission and scope

Build a **14-day Introduction to AI research project** comparing a fine-tuned **DistilBERT** email spam classifier with a **Naive Bayes** baseline on the **same frozen held-out test set**.

**Final output:** reproducible Python training and analysis scripts, two loadable models, honestly measured metrics, charts, and `notebooks/final_demo.ipynb` for a class presentation.

**Out of scope:** React/frontend, FastAPI/backend, databases, authentication, Gmail integration, cloud deployment, browser extensions, production services. Do not revive the earlier web-app plan or add these without explicit team approval. The final demo is a **Colab notebook**, not a website.

## 2. Before starting a coding session (required)

1. Identify the teammate you serve: **Aditya, Srishti, Eduardo, Dion, Luis, or Madeline**. If unknown, ask which teammate before changing files.
2. Read `MASTER_PLAN.md` Sections **3–5** (ownership, contracts, steps), **6–8** (handoffs, dates, gates), and **9** (Git workflow). Read the entire document at the start of the project.
3. Confirm the current branch is **not `main`**, inspect working-tree changes, and check the latest agreed contracts. Do not overwrite or discard another contributor's uncommitted work.
4. State a short plan naming: owned files to change, intended output, tests, and any upstream artifact required. Keep changes strictly within your assignment.
5. If a necessary handoff has not arrived, implement with **clearly labeled synthetic fixtures or mocks** that obey the exact contract. Never represent mock outputs as measured results.
6. After making changes, run relevant tests/smoke checks. Summarize the files changed, commands run, proof of success, artifact location, blockers, and the next downstream handoff.

**Never silently change** data schema, label mapping, prediction CSV format, metrics semantics, frozen splits, project scope, or another teammate's public functions. Propose such changes in a pull request and obtain agreement from the affected owners first.

## 3. Locked technical contracts

### Dataset: Eduardo owns its production

Three immutable CSVs: `data/processed/train.csv`, `data/processed/val.csv`, and `data/processed/test.csv`.

Required columns: `email_id` (stable unique string), `text` (subject + body or body alone), `label` (integer **1 = spam, 0 = legitimate**). Do not write the ground-truth label into `text`.

Use seed **42**, about **80/10/10 stratified** train/validation/test, deduplication and near-duplicate checks **before** splitting, and a frozen `data/processed/manifest.json` with source/provenance, counts, class balance, seed, and SHA256 hashes. No test-label leakage. Once frozen, changes require an explicit team-wide experimental restart.

### Model prediction interfaces: Aditya and Dion implement

Each loadable model must provide:

```python
def predict_spam_probability(texts: list[str]) -> list[float]:
    """One finite probability in [0, 1] per input email."""
```

Every exported prediction CSV must use these **exact** columns in order:

```text
email_id,label,spam_probability,predicted_label
```

`predicted_label` uses the validation-selected threshold (default `0.5`, record any alternative). `spam_probability` is a probability estimate, **not automatically calibrated confidence**. Both final prediction files must cover **identical test email IDs**, with no duplicates, missing rows, or extras.

Expected locations:

```text
reports/predictions/naive_bayes_test.csv
reports/predictions/distilbert_test.csv
reports/metrics/naive_bayes.json
reports/metrics/distilbert.json
reports/figures/*.png
```

### Experiment integrity: all agents enforce

- **Train** fits parameters; **validation** selects settings, checkpoints, and thresholds; **test** is used only for the final locked comparison.
- Never inspect test labels to tune, select examples for optimization, or revise model settings.
- Use identical frozen test IDs and evaluation definitions for both models. Report class counts and denominators.
- Keep software versions, random seed, run configurations, dataset hash, and checkpoint reference with outputs. Document potential GPU nondeterminism.
- For inference time, compare on the **same device**, with declared batch size, sample count, warmup, repetitions, and summary statistic. Clearly label any unmatched hardware comparison.
- Do not fabricate test results, improvement percentages, charts, accuracy claims, explanations, or training logs. Synthetic examples must say **SYNTHETIC / MOCK**.
- Do not upload personal emails, credentials, secrets, large raw corpora, or bulky model checkpoints to Git.

## 4. Six owners: exact boundaries and handoffs

| Owner | Work only in these primary paths | Can start before dependencies? | Required handoff |
|---|---|---|---|
| **Eduardo** | `src/data/prepare.py`, `data/processed/manifest.json` | **Yes**, immediately | Schema sample **Day 2**; frozen splits + hashes **Day 3**; data summary **Day 5** |
| **Aditya** | `src/models/distilbert_train.py`, `configs/distilbert_base.json` | **Yes**, use synthetic rows | Initial loadable checkpoint + validation outputs **Day 5**; chosen candidate **Day 7** |
| **Dion** | `src/models/naive_bayes.py`, `src/experiments/baseline_experiments.py` | **Yes**, use sample rows | Loadable baseline + validation results **Day 5**; final prediction CSV **Day 8** |
| **Srishti** | `src/experiments/distilbert_ablation.py`, `src/analysis/explainability.py` | **Yes**, use a toy predictor | Controlled ablation + explained examples / error cases **Day 9** |
| **Luis** | `src/evaluation/metrics.py`, `src/evaluation/benchmark.py`, shared metric tests under `tests/` | **Yes**, use mock predictions | Example metrics JSON **Day 2**; final held-out evaluation and benchmarks **Day 10** |
| **Madeline** | `src/visualization/plots.py`, `notebooks/final_demo.ipynb` | **Yes**, use mock JSON | Real plots and clean-run notebook **Day 12** |

**Everyone contributes tests for their own module** (coordinate ownership of `tests/` paths with Luis), reviews at least one other person's PR, and explains their component in the presentation. No one is automatically the default fixer for other people's work. If work is unbalanced, explicitly reassign a bounded task.

### Who actually needs to wait for whom?

- **Aditya and Dion** wait for **Eduardo's Day 3 frozen CSVs** only to run their final full-data training. Implement and smoke-test before then.
- **Srishti** waits for **Aditya's Day 5 checkpoint** only for real model-dependent perturbation experiments. Build the scripts with a toy predictor now.
- **Luis** waits for the **saved models/prediction outputs and Day 8 configuration freeze** only for final test metrics. Build all validators and benchmark harnesses with mocks first.
- **Madeline** waits for **Luis's Day 10 real metrics** and **Srishti's Day 9 analysis** only to finalize charts and notebook discussion. Build the demo shell immediately.
- **No one waits for a frontend, backend, or full team meeting** to begin an independent owned module.

If a handoff is **over one day late**, report the *specific artifact*, owner, impact, and fallback; keep working on a documented mock. Do not invent missing artifacts or quietly alter another person's code.

## 5. Nonnegotiable quality gates per owner

- **Eduardo:** all labels 0/1; nonempty text; unique IDs; no known cross-split duplicates/near duplicates; deterministic splits and hashes; documented public source/terms.
- **Aditya:** reproducible fine-tuning command; saved model + tokenizer reload; sample inference produces valid probabilities; loss history; first checkpoint Day 5 and candidate Day 7. Start with `distilbert/distilbert-base-uncased`, max length 256, batch 8/16, learning rate 2e-5, 2–3 epochs; treat these as *initial* settings, not guaranteed optimum.
- **Dion:** Naive Bayes fitted on train only; at least two **validation** configurations; serialized model and vectorizer reload; exact predictions contract; code identifies where libraries vs self-written algorithms are used.
- **Srishti:** at least two controlled transformer settings with validation metrics; at least five relevant explained examples; model sensitivity is labeled as **perturbation analysis**, not a causal proof; document one limitation.
- **Luis:** tested confusion matrix, accuracy, spam precision/recall/F1, macro F1, false positive rate; strict prediction ID validation; metrics JSON with sample counts, seed, dataset identity, environment; hardware-matched latency benchmark.
- **Madeline:** at least three legible plots derived from genuine stored results; one clean-run demo notebook; demonstrates both model predictions on two previously unseen sample emails; no hidden manual notebook state.

**Global acceptance:** both loadable classifiers, frozen/leakage-checked data, same test IDs, honest comparable metrics, controlled transformer ablation, clear limitations, working notebook, all six contributions represented. **Spam F1 >= 0.90 is a stretch target, not a condition for claiming success.** Never invent performance to satisfy it.

## 6. Timeline and integration checkpoints

| Day | Checkpoint / responsible handoff |
|---|---|
| **1** | Confirm professor permits pretrained models; agree contracts; initialize repo and each feature branch. |
| **2** | Eduardo sample CSV; Luis metrics JSON example; all modules can smoke-test with mocks. |
| **3** | Eduardo publishes final immutable splits and manifest. |
| **5** | Dion baseline; Aditya initial checkpoint; share validation outputs. |
| **7** | Aditya candidate model; early integration validation. |
| **8** | Aditya, Dion, and Srishti freeze all model/threshold choices. |
| **9–10** | Srishti analysis; Luis runs the one final held-out comparison and exports real results. |
| **12** | Madeline verifies notebook runs from clean Colab session; demo frozen. |
| **13–14** | Whole team rehearses, verifies artifacts, prepares report/slides, submits. |

**Do not move test evaluation earlier** just because an upstream artifact appears early. Data and model settings must be frozen first.

## 7. Git, file ownership, and artifacts

Branch names:

```text
feat/data-eduardo
feat/distilbert-aditya
feat/tuning-srishti
feat/baseline-dion
feat/evaluation-luis
feat/demo-madeline
```

- Work on your own branch; never directly edit `main`. Open small PRs, ideally every two days. Ask another teammate to review; do not force-push shared branches.
- Restrict changes to the owned paths above. A shared config, `requirements.txt`, `README.md`, `MASTER_PLAN.md`, `.gitignore`, or model contract change requires a coordinated PR, not an unannounced edit.
- Save large CSVs and model checkpoints in team-controlled shared storage, with versioned names, SHA256 checksums, and retrieval instructions; commit only lightweight manifests, code, reports, and notebook.
- Ignore `data/raw/`, `data/processed/*.csv`, `models/`, `.env`, `.ipynb_checkpoints/`, `__pycache__/`, and local temporary outputs.
- One person owns final notebook edits (**Madeline**). Others contribute modules, experiments, results, or separate PRs for agreed changes.
- If shared-state issues or merge conflicts arise, stop and coordinate instead of resolving by dropping another person's changes.

### PR and end-of-session report (required)

```text
Owner:
Branch:
Files changed:
Purpose / contract impact:
Commands run and results:
Tests passed / not run (why):
Artifacts produced (path, hash/version if external):
Handoff to (person, deadline):
Blockers / missing inputs:
Next action:
```

## 8. Resilience and fallback choices

- **Professor disallows pretrained models:** Stop DistilBERT work; the team must approve the alternative (Naive Bayes plus a simple neural classifier) and update both master files together.
- **Dataset unavailable by Day 3:** Eduardo proposes a documented public alternative; avoid any accidental test leakage or undisclosed changes.
- **GPU limited:** Reduce sequence length/batch size/dataset subset or train for one epoch first, document compromises, preserve the baseline.
- **Transformer run fails by Day 7:** Maintain a complete Naive Bayes comparison/demo where feasible, show genuine partial transformer experiments and disclose what failed.
- **Demo unavailable during presentation:** Use a saved, genuine execution and exported charts; say clearly if the demonstration is prerecorded.

**Final instruction:** Optimize for reproducibility, scientific integrity, limited scope, and independent ownership. Read the relevant section of `MASTER_PLAN.md` before each task; do not build anything outside the agreed research plan.
