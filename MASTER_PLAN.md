# SmartMail AI

## 14-Day Master Execution Plan

**Course:** Introduction to Artificial Intelligence  
**Team:** Aditya, Srishti, Eduardo Valdovinos, Dion, Luis Meza, Madeline Cohen  
**Version:** 1.0 | November 16, 2026  
**Status:** Agreed implementation blueprint; actual results must be measured

> **Project decision:** We are conducting a controlled experiment comparing a fine-tuned DistilBERT classifier against a Naive Bayes baseline for email spam detection. We will deliver reproducible Python code, results, and one Google Colab demo notebook. **No frontend, backend, live Gmail access, or deployment is in scope.** This plan supersedes the previous web application plan and its older `AGENTS.md` instructions.

## 1. Objective and research question

**Research question:** On the same held-out email test set, does fine-tuned DistilBERT improve spam detection compared with a traditional Naive Bayes classifier, and is the improvement worth the added computational cost?

The presentation must demonstrate a fair comparison rather than claim that DistilBERT is automatically better. We will report precision, recall, F1, accuracy, false positives, inference time, and representative mistakes. We will also show the effect of at least one DistilBERT training decision, and discuss limitations such as outdated datasets and long emails.

**Final deliverables:**

1. A frozen, documented dataset split and preprocessing script.
2. A trained Naive Bayes baseline and saved model.
3. A fine-tuned DistilBERT checkpoint and reproducible training script.
4. Shared metrics, prediction files, runtime benchmarks, and error analysis.
5. Clear figures and one runnable `notebooks/final_demo.ipynb`.
6. A short report/presentation with honest conclusions, contributions, and limitations.

## 2. Operating principles

- **One source of truth:** This `MASTER_PLAN.md` governs responsibilities and contracts. Change it only with team agreement through a pull request. If an older `AGENTS.md` conflicts, update that file to point here before coding.
- **Everyone codes:** Each person owns a meaningful technical deliverable and explains it in the presentation.
- **Work in parallel; wait only for final inputs:** Build against the shared schema and synthetic examples before upstream files arrive. Do not block an entire workstream waiting for a teammate.
- **One fixed experiment:** The same frozen dataset splits, label mapping, and evaluation code apply to both models.
- **No result fabrication:** Every number in slides must come from versioned evaluation outputs; use no invented accuracy or benchmark claims.
- **Reproducibility first:** Record random seeds, library versions, training config, dataset provenance, and artifact locations.
- **No raw private emails:** Use a public dataset with an appropriate license; never upload personal inbox exports or credentials.

## 3. Team ownership and workload

Each role is planned for approximately **12 to 15 focused hours over two weeks**, plus a similar share of rehearsal and presentation. If a role runs long, swap a bounded task rather than silently transferring all work to Aditya.

| Person | Primary ownership | Owned code / assets | Handoff |
|:--|:--|:--|:--|
| **Aditya** | Core DistilBERT fine-tuning pipeline, one controlled training run | `src/models/distilbert_train.py`, `configs/distilbert_base.json` | Checkpoint and training history to Srishti and Luis |
| **Srishti** | DistilBERT experiments, validation-based tuning, token masking and error cases | `src/analysis/explainability.py`, `src/experiments/distilbert_ablation.py` | Ablation summary and example explanations to Madeline |
| **Eduardo** | Public dataset ingestion, cleaning, deduplication, splitting and manifest | `src/data/prepare.py`, `data/processed/manifest.json` | Frozen CSV splits to Aditya, Dion, Luis |
| **Dion** | Naive Bayes baseline, vectorizer experiments and model serialization | `src/models/naive_bayes.py`, `src/experiments/baseline_experiments.py` | Baseline model and validation/test predictions to Luis |
| **Luis** | Shared evaluation, test-set integrity, runtime benchmark, metrics exports | `src/evaluation/metrics.py`, `src/evaluation/benchmark.py`, `tests/` | Metrics and prediction reports to Madeline |
| **Madeline** | Charts, reproducible final demo notebook, visual QA of results | `src/visualization/plots.py`, `notebooks/final_demo.ipynb` | Final figures, runnable notebook, demo sequence |

**Coordination is shared:** Each member reviews at least one teammate's pull request. The person merging a pull request checks interfaces; Aditya is **not** the sole reviewer, integration engineer, slide author, or troubleshooter. Rotate a 10-minute integration check between Luis, Madeline, and Eduardo on Days 4, 7, and 10.

## 4. Day 1 decisions and locked contracts

These decisions should be agreed in a 30-minute kickoff, then committed before feature branches diverge.

### 4.1 Dataset schema

Eduardo owns generating the following **three CSVs**, which are copied to shared storage rather than committed if large:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

Each CSV contains exactly these required columns:

| Column | Type | Meaning |
|:--|:--|:--|
| `email_id` | string | Stable unique identifier; one example per ID |
| `text` | string | Email subject and body text, combined consistently |
| `label` | integer | `1 = spam`, `0 = legitimate (ham)` |

**Text construction:** If available, use `Subject: <subject>\nBody: <body>`. If subject is missing, use the body alone. Do not include the gold label in text. Strip or mask sender addresses and personal identifiers when practical; apply the same rule to all splits. No mandatory stemming or stop-word removal for DistilBERT.

**Split policy:** Start with an approximately 80/10/10 stratified split with seed `42`. Deduplicate and group near-duplicate messages *before* splitting to avoid leakage. Record the counts, label balance, source, license/provenance, seed and SHA256 hashes in `data/processed/manifest.json`. Do not regenerate splits once frozen unless the whole team explicitly restarts the experiment.

**Dataset freeze:** Eduardo publishes a 20-row synthetic/example schema sample by **Day 2**, and the final validated split plus manifest by **end of Day 3**. All model training uses the frozen files.

### 4.2 Prediction interface

Both models must emit a CSV at a shared path with identical columns:

`email_id,label,spam_probability,predicted_label`

`spam_probability` must be a finite float between 0 and 1. `predicted_label` is determined using the chosen validation-set threshold (default `0.5`; record any change). Rows must correspond exactly to the frozen test `email_id` values. Do not call raw class scores calibrated confidence if calibration was not evaluated.

Suggested outputs:

- `reports/predictions/naive_bayes_test.csv`
- `reports/predictions/distilbert_test.csv`
- `reports/metrics/naive_bayes.json`
- `reports/metrics/distilbert.json`
- `reports/figures/*.png`

The baseline and DistilBERT owners expose a simple `predict_spam_probability(texts: list[str]) -> list[float]` interface from their loadable model wrappers. Luis validates that both implement the contract.

### 4.3 Shared experiment and evaluation rules

- The **training split** fits model parameters; the **validation split** selects hyperparameters, checkpoints and thresholds; the **test split** is reserved for final comparison.
- No one manually examines test labels to design features or tune the model. A final held-out evaluation is run only after choices are frozen.
- Use fixed seed `42` where supported, with software versions recorded. GPU operations may still have some nondeterminism; document it.
- Start DistilBERT with `distilbert/distilbert-base-uncased`, sequence length 256, batch size 8 or 16, learning rate 2e-5, 2 to 3 epochs. These are **initial settings**, not promised best values.
- Evaluate Naive Bayes and DistilBERT on exactly the same IDs. Measure runtime on the same device, batch size and sample count with warmup; clearly label CPU versus GPU timings.

## 5. Exact instructions by team member

### Aditya | DistilBERT core training

**Start immediately (Day 1):**

1. Confirm the professor allows fine-tuning a pretrained model. Set up Python, PyTorch, Transformers, datasets and a Colab GPU fallback.
2. Create `src/models/distilbert_train.py` with CLI arguments for `--train`, `--val`, `--output`, `--seed`, `--max-length`, `--epochs`, and `--batch-size`.
3. Build a **smoke test using synthetic rows**; confirm tokenization, forward pass, one optimizer step, save and reload. Do not wait for Eduardo's full dataset.
4. After Eduardo freezes data (Day 3), run a short small-subset experiment, then the primary training run. Save tokenizer, model, label mapping, training config and loss history.
5. Give Srishti an initial checkpoint and validation outputs by **Day 5**, then a selected candidate checkpoint to Luis by **Day 7**.

**Wait only for:** Eduardo's frozen splits before final training. **Do not wait for:** baseline model, charting, or demo notebook.

**Success gate:** From a clean environment, a documented command fine-tunes and saves a loadable checkpoint; reloading it produces one valid spam probability for a sample email; training and validation loss are recorded. Initial checkpoint is shared by Day 5, final candidate by Day 7.

### Srishti | Optimization and explainability

**Start immediately (Day 1):**

1. Define a controlled experiment table with at least **two training settings** (for example 1 versus 2 epochs, or max length 128 versus 256) and the *same validation metric*.
2. Write `src/experiments/distilbert_ablation.py` to compare saved run metadata and validation results. Define an output CSV before checkpoints exist.
3. Write `src/analysis/explainability.py` to mask individual words or short token spans and measure changes in prediction. Label this as **perturbation sensitivity**, not proof of causal reasoning.
4. With Aditya's first checkpoint (target Day 5), run at least one controlled validation experiment, 5 to 10 explanation examples, and a short failure analysis.
5. Export `reports/analysis/ablation.csv` and `reports/analysis/error_cases.csv` for Madeline by **Day 9**.

**Wait only for:** a checkpoint to run model-dependent experiments. **Before then:** implement the ablation loader and masking logic with a toy predictor.

**Success gate:** At least two documented settings with validation metrics, 5 interesting explained cases, and one honest limitation. Work does not require changing Aditya's training script; request a separate run rather than editing his file.

### Eduardo | Dataset engineering

**Start immediately (Day 1):**

1. Select and document a public labeled email spam corpus, its provenance, license/terms, and spam/ham definitions.
2. Implement a deterministic `src/data/prepare.py`: parse emails, combine subject and body, normalize text minimally, remove empty examples, deduplicate, and generate IDs.
3. Publish a 20-row **schema-valid synthetic/sample file by Day 2**. Confirm with Aditya and Dion that they can read it.
4. Generate and validate final stratified train/validation/test splits with no duplicate IDs or duplicate text across splits; examine near duplicates, not just exact hashes.
5. Publish the final CSVs and `manifest.json` by **end of Day 3**, with SHA256 hashes and download instructions. Export `reports/data_summary.json` by Day 5.

**Wait for:** nobody. **Handoff dependency:** everyone else needs your stable schema immediately and frozen files by Day 3.

**Success gate:** 0 missing labels; labels only 0/1; no empty texts; unique IDs; no known duplicate or near-duplicate leakage; split reproducibility under seed 42; class counts and data source documented; all files readable using `pandas.read_csv`.

### Dion | Naive Bayes baseline

**Start immediately (Day 1):**

1. Implement text feature extraction using TF-IDF or count vectors, and Multinomial Naive Bayes in `src/models/naive_bayes.py` (record which operations use libraries versus your own code).
2. Build training and prediction functions against Eduardo's schema sample; add a simple test with several labeled example sentences.
3. After the Day 3 dataset freeze, train the baseline and compare at least two vectorizer/model settings **on validation only**.
4. Save model + vectorizer with a load function and export validation predictions by **Day 5**.
5. After choosing settings, export frozen test predictions to Luis by **Day 8**; include dependency versions and reproducible commands.

**Wait only for:** frozen splits for final model fitting. **Before then:** implement, test and smoke-run the baseline on synthetic data.

**Success gate:** A saved/reloaded baseline outputs valid probabilities; at least two validation settings compared; test predictions follow the common CSV contract; no test-set tuning or leakage.

### Luis | Evaluation and benchmarking

**Start immediately (Day 1):**

1. Implement `src/evaluation/metrics.py` to validate prediction schema and calculate confusion matrix, accuracy, spam precision, spam recall, spam F1, macro F1 and false positive rate.
2. Unit test metrics using a tiny manually checkable confusion matrix and deliberately broken prediction files.
3. Define a JSON schema for metrics and a clear runtime methodology: same machine/device, warmup, batch size, number of emails, repeated measurements and median latency.
4. Run the evaluation pipeline against synthetic predictions first; share an **example metrics JSON by Day 2** so Madeline can build charts.
5. When candidate models arrive, run integration checks; run final held-out evaluation **only after settings are frozen around Days 9 to 10**. Export both metrics JSONs, paired prediction files, and benchmarks.

**Wait only for:** final saved model/prediction files for true results. **Before then:** finish metric code, tests, JSON schema and speed benchmark harness.

**Success gate:** Both prediction files have exactly the same test IDs and no missing/extra rows; metric tests pass; generated JSON includes denominators, confusion matrix, evaluation timestamp, seed/data version, and environment; runtime comparison is explicitly hardware-matched.

### Madeline | Figures and final demo

**Start immediately (Day 1):**

1. Create `src/visualization/plots.py` to read Luis's JSON and render clear confusion matrices, a metric comparison and a training/ablation figure.
2. Develop `notebooks/final_demo.ipynb` using **mock JSON and synthetic example emails**. Never edit the training files or calculate fake final results.
3. Create a simple demo flow: dataset summary -> training methodology -> model comparison -> 2 or 3 live email predictions -> mistakes and limitations.
4. Once Luis delivers real metrics and Srishti delivers experiments (Days 9 to 10), swap mocks for actual outputs and label every chart with the test dataset version.
5. Execute the notebook top to bottom from a clean Colab runtime by **Day 12**. Own presentation visuals; ask every teammate for one explanation slide, not the entire presentation.

**Wait only for:** final metrics to publish real charts. **Before then:** use mock schema and confirm the notebook runs with fake clearly-labeled values.

**Success gate:** At least 3 legible, correctly labeled plots; the final notebook runs top to bottom without manual hidden state; demo accepts a new email and displays both model outputs; all plots point to real artifacts.

## 6. Dependency map: who should wait for whom?

| Handoff and owner | Due | Unblocks | What recipients do while waiting |
|:--|:--|:--|:--|
| Data **schema sample**, Eduardo | Day 2 | Aditya, Dion, Luis | Write loaders, algorithms and metrics using synthetic rows |
| Metrics **JSON example**, Luis | Day 2 | Madeline | Build chart functions using mock metrics |
| **Frozen train/val/test**, Eduardo | Day 3 | Aditya and Dion final training | Finish smoke tests, CLI and unit tests |
| DistilBERT **initial checkpoint**, Aditya | Day 5 | Srishti's model-dependent tests | Implement ablation and masking on toy predictor |
| Naive Bayes **baseline artifact**, Dion | Day 5 | Luis's early integration | Verify metric contracts with synthetic predictions |
| DistilBERT **candidate checkpoint**, Aditya | Day 7 | Luis integration and Srishti experiments | Continue tool and report scaffolding |
| **Frozen model configurations**, Aditya + Dion + Srishti | Day 8 | Luis's final evaluation | Finish experiment notes and validation checks |
| **Final test metrics**, Luis | Day 10 | Madeline final figures and demo | Render demo with clearly marked mock values |
| **Final explanation and error cases**, Srishti | Day 9 | Madeline final interpretation slides | Build plotting and notebook shells |
| **Validated demo notebook**, Madeline | Day 12 | Whole team rehearsal | Each member prepares concise role explanation |

**Escalation:** If a handoff is more than **one day late**, post a blocker with the exact missing artifact, proposed fallback and latest safe deadline. The dependent teammate continues on their mock/test harness; the team revises the milestone together. No one should silently wait.

## 7. Fourteen-day execution calendar

| Day | Team checkpoint | Required evidence |
|:--|:--|:--|
| 1 | Kickoff; professor permission; repo; frozen schema, roles and label convention | `MASTER_PLAN.md`, owner assignments, folder scaffold |
| 2 | API/schema mocks ready | 20-row sample, sample metrics JSON, training and plot smoke tests |
| 3 | Dataset freeze | Validated three CSVs, source and hash manifest |
| 4 | Both model pipelines run | Baseline training proof; DistilBERT training smoke proof |
| 5 | First model handoffs | Baseline artifact; first DistilBERT checkpoint; validation outputs |
| 6 | Validate and tune | Training curves, initial benchmark, experiment logs |
| 7 | DistilBERT candidate delivered | Loadable checkpoint and documented configuration |
| 8 | Model settings frozen | Chosen validation thresholds and configs, no test tuning |
| 9 | Controlled analysis and final test starts | Srishti's ablations; Luis's integrity checks |
| 10 | Final evaluation delivered | Both models' held-out metrics and speed results |
| 11 | Real results replace mock results | Charts and error-case narrative sourced to real artifacts |
| 12 | Demo freeze | Clean notebook execution and saved demo evidence |
| 13 | Rehearsal and QA | Presentation run-through, all contributors speaking |
| 14 | Submission and backup | Clean repo, artifacts accessible, notebook + slides + report |

## 8. Measurable success parameters

### 8.1 Minimum viable project: all must pass

- [ ] A professor-approved model strategy (or documented alternative approved before the team commits).
- [ ] Public source, class mapping, license/provenance and frozen data hashes documented.
- [ ] Train, validation and test splits are disjoint; no known cross-split duplicate leakage.
- [ ] Naive Bayes and DistilBERT both produce loadable models and test predictions.
- [ ] Both models are evaluated on the **same exact held-out test IDs** with identical metrics.
- [ ] Reports include accuracy, precision, recall, F1, macro F1, false positive rate and confusion matrices.
- [ ] At least **two** controlled DistilBERT validation settings, with an explanation of the chosen one.
- [ ] Runtime measurement states device, batch size, sample count and timing method.
- [ ] One notebook runs top to bottom in a clean environment and classifies **two previously unseen sample emails**.
- [ ] All final figures are generated from real saved results, and each person can explain their contribution.

### 8.2 Quality targets: aspirational, not fabricated pass/fail grades

- **Model quality target:** Aim for spam F1 >= 0.90 on the held-out test set, **but report the actual number even if lower**. The project succeeds academically if the comparison is rigorous and limitations are understood.
- **Reliability:** Zero crashes on the final prepared demo sequence; save a local backup of outputs in case Colab GPU is unavailable during the presentation.
- **Reproducibility:** A teammate other than the file owner can execute the documented steps and obtain predictions/metrics from the stored artifacts.
- **Scope discipline:** No extra web application work until all minimum viable gates pass.
- **Presentation:** Every member has a technical contribution to demonstrate and can answer one methodology question.

### 8.3 Scientific integrity rules

Do not judge success only by accuracy, and do not claim a model is better if its gain is within plausible evaluation noise. Include the number of test emails in each class, and discuss at least three real errors (if there are three). If no errors occur in a small set, acknowledge the sample size. Do not compare GPU DistilBERT timing with CPU Naive Bayes timing without labeling it as non-equivalent.

## 9. GitHub collaboration without merge conflicts

**Repository layout** (only code, manifests, lightweight results and notebook go into Git):

```text
smartmail-ai/
  MASTER_PLAN.md
  README.md
  requirements.txt
  .gitignore
  configs/distilbert_base.json
  src/data/prepare.py                       # Eduardo
  src/models/distilbert_train.py             # Aditya
  src/models/naive_bayes.py                  # Dion
  src/experiments/distilbert_ablation.py     # Srishti
  src/experiments/baseline_experiments.py    # Dion
  src/analysis/explainability.py             # Srishti
  src/evaluation/metrics.py                  # Luis
  src/evaluation/benchmark.py                # Luis
  src/visualization/plots.py                 # Madeline
  notebooks/final_demo.ipynb                 # Madeline only
  tests/                                     # Luis leads, owners add tests
  reports/                                   # small JSON, CSV, PNG only
  data/processed/manifest.json              # Eduardo
```

**Never commit** large raw email corpora, personal emails, `.env` files, access tokens, or large model checkpoints. Store datasets/checkpoints in a shared Drive folder with clear paths, version names and SHA256 checksums. Provide a README on how to retrieve them. Add `data/raw/`, `data/processed/*.csv`, `models/`, `.env`, `.ipynb_checkpoints/` and `__pycache__/` to `.gitignore`.

**Branch conventions:** `feat/data-eduardo`, `feat/distilbert-aditya`, `feat/tuning-srishti`, `feat/baseline-dion`, `feat/evaluation-luis`, `feat/demo-madeline`. Nobody writes directly to `main`. Open small pull requests, keep edits inside owned paths, and ask for approval before changing another owner's contract. Do not force push shared branches. Integrate at least every two days.

**Pull request checklist:** What changed? Which files? How tested? What command runs it? What artifact is produced? Is a downstream handoff affected? Does it alter data hashes or the frozen test set? Link to evidence and request a reviewer other than the author.

**Daily asynchronous update format:** `Done | Next | Blocked by | Artifact / PR link`. A quick daily check is sufficient; only hold a meeting for a blocked handoff or a major decision.

## 10. Blockers and fallback plans

| Risk or missed milestone | Trigger | Response |
|:--|:--|:--|
| Pretrained models not allowed | Professor rejects approach on Day 1 | Switch to Naive Bayes plus a simple neural classifier trained by the team; revise research question immediately |
| Dataset unavailable or messy | Validated split absent at Day 3 | Use a vetted public alternative; freeze one clean version; reduce data scope rather than bypass leakage checks |
| Colab GPU unavailable | DistilBERT smoke test fails or GPU quota blocks training | Reduce max length to 128, batch size to 4, subset size, and start with one epoch; keep truthful compute notes |
| Fine tuning fails by Day 7 | No working loadable checkpoint | Protect the baseline and demo; show a smaller successful transformer experiment if possible, otherwise state explicitly what failed |
| Integration delay | Outputs fail schema checks on Day 8 | Luis provides a validation script and failing row examples; owners repair their own exports rather than rewriting someone else's module |
| Demo environment unavailable | Colab cannot run during presentation | Show previously saved genuine plots, measured outputs and a recorded local execution; do not pretend a recording is live |

## 11. Submission package and final sign-off

At the end of Day 14, the GitHub repository and shared artifact folder must contain:

- A concise `README.md` with setup, data access, model loading, training and evaluation commands.
- Source code owned by all six people with passing smoke/unit tests and clear dependency versions.
- Dataset manifest and accessible frozen data files (outside Git if large).
- Both trained model artifacts with versioned configuration and download instructions.
- JSON/CSV metrics and predictions with the same test IDs and dataset hash.
- At least three correctly titled figures, plus analysis of failures and model tradeoffs.
- `notebooks/final_demo.ipynb`, executed from clean state, with a backup demonstration.
- Presentation/report covering motivation, methods, experimental design, results, limitations, individual contributions and next steps.

**Final acceptance:** Each owner signs off on their own handoff. Luis confirms the evaluation contract and test-set integrity. Madeline confirms the notebook/figures. One teammate other than Aditya runs the final demo. The whole team agrees that conclusions follow from measured results.

## 12. First action: team kickoff checklist

1. Confirm grading rubric and whether pretrained models are allowed.
2. Approve the six roles and the dates for Days 2, 3, 5, 7, 8, 10 and 12.
3. Freeze CSV fields, label mapping, predictions schema and shared metrics schema.
4. Create one private GitHub repository and shared storage for datasets/checkpoints.
5. Have each person open their first small PR or issue by end of Day 1.
6. Confirm the Day 2 outputs: sample CSV, sample metrics JSON, model smoke tests and demo plotting scaffold.

**Team rule:** Nobody should need to ask, "What do I do next?" Start the next step in your section; if its final input is not ready, use the specified mock/sample and report only the concrete dependency.
