# Responsible Medical LLMs for Mental Healthcare (LCS2 Round 2)

A small, reproducible benchmark that evaluates **Llama-3.2-3B-Instruct** and **Qwen2.5-3B-Instruct** on the
fixed MedMisBench mental-health multiple-choice task across four dimensions: hallucination and uncertainty,
interpretability and explainability, bias and fairness, and privacy and security. It also includes a
cross-dimension analysis and one prompt-level intervention.

The report is `YashvardhanGoel_LCS2_Round2_Report.pdf`.

## Repository contents

```
.
├── main.ipynb                               # the whole pipeline: every model run, metric, table and figure
├── main_results/                            # every saved output from the reported run (see below)
├── YashvardhanGoel_LCS2_Round2_Report.pdf   # final report (ACL format)
├── AI_USAGE.md                              # AI tools used, what for, and how outputs were checked
└── README.md
```

Everything runs from the single notebook `main.ipynb`. It contains the prompts, configuration, seed,
model and dataset revisions, all experiments and the code that produces every table and figure.

## Fixed setup

| Setting | Value |
| --- | --- |
| Llama | `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Qwen | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Dataset | `Johndfm/MedMisBench-MentalHealth` @ `9757095598b6555c3e935679fdb7384aed4eb651`, file `medmisbench_mentalhealth.jsonl` |
| Dataset check | SHA-256 `0c7db2295ca5bf220933bc95d8f613c11e3f8c0d3d5f0235dbf56595d5434c21` (asserted before every run) |
| Subsets | zero-based `row_index` in stored order, no shuffling: Core 0–199, Stress 0–59, Interpretability 0–39 |
| Inference | local Hugging Face `transformers`, float16, no quantization, one model in memory at a time |
| Decoding | greedy (`do_sample=False`), `max_new_tokens=384` |
| Seed | 42 for `random`, `numpy`, `torch` and every bootstrap/random control |
| Hardware | free Kaggle notebook, GPU T4 ×2 accelerator (one GPU used) |
| Environment | Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, pydantic 2.13.5 (saved in `main_results/environment.json`); Kaggle docker image version 28755 |

Llama's chat template writes the current date into the system prompt, so the notebook fixes it to
`26 Jul 2024` to keep prompts identical across runs.

## Reproduce from a fresh Kaggle notebook

1. **Hugging Face access.** Accept the Llama 3.2 licence on its Hugging Face page and create a read token.
2. **Create the notebook.** On Kaggle, create a new notebook and import `main.ipynb`
   (File → Import Notebook). In the settings, choose the **GPU T4 ×2** accelerator and turn **Internet on**.
3. **Add the token.** Add-ons → Secrets → add a secret named `HF_TOKEN` with your token and attach it to the
   notebook. The notebook logs in with it.
4. **Dataset (automatic).** The configuration cell (second code cell) downloads
   `medmisbench_mentalhealth.jsonl` from `Johndfm/MedMisBench-MentalHealth` at revision
   `9757095598b6555c3e935679fdb7384aed4eb651` with `hf_hub_download`, so nothing needs to be uploaded. The
   SHA-256 check stops the run if the file is not the expected revision.
5. **Run everything.** Keep `REUSE_SAVED = False` and run all cells (Run All). The reported run took about
   3.8 hours. Every output is written to `/kaggle/working/results/`.
6. **Compare.** The new files should match `main_results/`. Greedy decoding is deterministic on the same
   GPU and library versions; a different GPU or library version can change a few generations.

**Reusing saved runs.** Set `REUSE_SAVED = True` and point `BASELINE_DIR` at a Kaggle dataset that holds
the saved `.jsonl` run files (for example the ones in `main_results/`). The notebook then loads those runs
instead of generating them again, and all analysis cells still run.

## Notebook layout

The notebook follows the brief, with one markdown heading per part:

| Notebook section | What it does |
| --- | --- |
| Setup and configuration | imports, revisions, seed, dataset hash check, prompts, Pydantic output parser, model loading |
| Part A: baseline | runs both models on the 200 Core questions; summary and calibration checks |
| Two falsifiable hypotheses | H1 (misleading context) and H2 (security), fixed before the runs |
| Experiment 5.1 | clean vs. misleading context on the Stress set; note-source labels; H1 test |
| Experiment 5.2 | attribution patching and mean ablation (top/random/bottom $k$), type-matched control, cases |
| Experiment 5.3 | six metadata variants, paired flips, justified vs. unjustified flips, cases |
| Experiment 5.4 | targeted injection, canary leakage, no-prohibition control, H2 test, cases |
| Part C | cross-dimension analysis (Fisher, CMH, Holm-corrected exploratory checks) |
| Part D | prompt hardening before/after on four conditions; extra input-delimiting check |

All prompts are defined in the notebook: the base system prompt (setup), the misleading-context format
(5.1), the metadata lines (5.3), the injection and canary texts (5.4), and the hardened and delimited
prompts (Part D). The report's Appendix C lists them too.

## Saved outputs (`main_results/`)

**Raw model outputs (`.jsonl`).** One record per model and question, including failed generations. Each
record holds `row_index`, `dataset_id`, `model`, `model_revision`, `dataset_revision`, `precision`,
`generation_settings`, `gold_answer`, `raw_output`, `parsed_answer`, `confidence`, `evidence`, `correct`
and `parse_status`. File names are `<condition>__<model>.jsonl`, for example
`baseline_core__Llama-3.2-3B-Instruct.jsonl` or `security_injection__Qwen2.5-3B-Instruct.jsonl`.
The 5.2 files (`interp_items__*`, `interp_ablation__*`, `interp_typed__*`) store attributions and
ablation results per question; `interp_skipped__*` lists questions that could not be scored.

**Where each table and figure in the report comes from:**

| Report | Source file(s) | Notebook section |
| --- | --- | --- |
| Table 1 (setup) | `environment.json` | Setup |
| Table 2 (clean baseline) | `baseline_core_summary.csv`, `baseline_core_checks.csv` | Part A |
| Table 3 (misleading context) | `stress_5_1_results.csv`, `stress_5_1_by_label.csv`, `stress_5_1_paired.csv` | 5.1 |
| Table 4 (faithfulness test) | `interp_5_2_results.csv`, `interp_5_2_typed_results.csv`, `interp_5_2_cases.csv` | 5.2 |
| Figures 1–2 | `interp_5_2_ablation_curve.png`, `interp_5_2_typed_curve.png` | 5.2 |
| Table 5 (bias) | `bias_5_3_results.csv`, `bias_5_3_paired.csv`, `bias_question_types.csv` | 5.3 |
| Table 6 (security) | `security_5_4_injection_results.csv`, `security_5_4_canary_results.csv`, `security_5_4_h2.csv`, `security_5_4_items.csv` | 5.4 |
| Table 7 (cross-dimension) | `cross_c_primary.csv`, `cross_c_secondary.csv`, `cross_c_items.csv` | Part C |
| Table 8 (intervention) | `harden_d_results.csv`, `harden_d_by_label.csv`, `delimit_d_results.csv`, `delimit_d_by_label.csv` | Part D |

Two statistics in the report are printed by the notebook rather than saved to a CSV: the H1 Fisher odds
ratios and p-values (the cell that writes `stress_5_1_by_label.csv`) and the justified/unjustified flip counts with their Fisher test
(5.3, "justified vs unjustified flips"). Both are computed from the saved paired files above.

## Data and safety

The dataset is derived from MedMisBench (Zhou et al., 2026, arXiv:2606.12291). It **contains false and
misleading medical statements on purpose**, and the saved model outputs include confident but wrong
medical advice. Everything here is for robustness research only and is not clinical guidance. Only the
synthetic canaries required by the brief are used; no real patient data or secrets are placed in model
context.

## AI usage

See `AI_USAGE.md` and Appendix D of the report.
