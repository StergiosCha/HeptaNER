#!/usr/bin/env python3
"""
Evaluate all model × strategy results against ground truth from the Excel file.
Uses deterministic semantic matching (no API needed).
Scores core layer, extension layer, and hybrid tags separately.
Produces a final xlsx with full breakdown + 95% Wilson confidence intervals.

Supports multi-run variance analysis: evaluate 3 runs and report mean ± std.

Usage:
    cd heptanesian_ner
    python scripts/evaluate_all_results.py --runs 1           # single run
    python scripts/evaluate_all_results.py --runs 1 2 3       # 3 runs with variance
"""

import json
import os
import sys
import re
import math
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────
EXCEL_PATH = "data/testing_acts.xlsx"
EXCEL_SHEET = "Additional acts"
BENCHMARK_ACTS = "data/benchmark_40_acts.json"
OUTPUT_BASE = "outputs/v3_evaluation"

MODELS = ["claude", "claude4", "claude4o", "gpt4o", "gpt41", "o3mini", "gemini", "llama", "llama8b", "deepseek", "deepseekr", "qwen", "gemini25", "mistral", "gpt52", "llama4"]
MODEL_LABELS = {
    "claude":    "Claude 3.7 Sonnet",
    "claude4":   "Claude Sonnet 4",
    "claude4o":  "Claude Opus 4",
    "gpt4o":     "GPT-4o",
    "gpt41":     "GPT-4.1",
    "o3mini":    "o3-mini",
    "gemini":    "Gemini 2.0 Flash",
    "llama":     "Llama 3.3 70B",
    "llama8b":   "Llama 3.1 8B",
    "deepseek":  "DeepSeek V3",
    "deepseekr": "DeepSeek R1",
    "qwen":      "Qwen 2.5 72B",
    "gemini25":  "Gemini 2.5 Flash",
    "mistral":   "Mistral Large 3",
    "gpt52":     "GPT-5.2 Chat",
    "llama4":    "Llama 4 Maverick",
}
STRATEGIES = ["zero_shot", "few_shot", "full_context", "nesy"]

# ── Semantic Equivalence Groups for CORE LAYER ─────────────────────────────
# Each group = one ground-truth core category and its acceptable surface forms.
# IMPORTANT: categories that are distinct in the taxonomy get their own group.
CORE_EQUIVALENCE = [
    # 1. Venditio
    {"venditio", "sale", "πωληση", "πώληση", "emptio venditio", "emptio-venditio"},
    # 2. Donatio
    {"donatio", "donation", "δωρεα", "δωρεά", "donatio inter vivos",
     "donatio propter animam"},
    # 3. Permutatio
    {"permutatio", "exchange", "ανταλλαγη", "ανταλλαγή"},
    # 4. Emphyteusis (distinct from Locatio!)
    {"emphyteusis", "εμφυτευση", "εμφύτευση", "emphyteusis perpetua",
     "emphyteusis ecclesiastica"},
    # 5. Locatio conductio rei (Lease)
    {"locatio conductio rei", "locatio conductio", "lease", "μισθωση", "μίσθωση",
     "εκμισθωση", "εκμίσθωση", "locatio agricola", "locatio vineae"},
    # 6. Mutuum (Loan)
    {"mutuum", "loan", "δανειο", "δάνειο", "foenus", "mutuum cum pignore"},
    # 7. Praevenditio (Forward sale)
    {"praevenditio", "forward sale", "προπωληση", "προπώληση",
     "venditio futurae rei", "venditio futura"},
    # 8. Quietantia (Receipt)
    {"quietantia", "receipt", "αποδειξη πληρωμης", "απόδειξη πληρωμής",
     "confessio solutionis", "apodixis"},
    # 9. Instrumentum obligationis (Debt acknowledgement)
    {"instrumentum obligationis", "debt acknowledgement", "ομολογια χρεους",
     "ομολογία χρέους", "recognitio debiti"},
    # 10. Testamentum (Will)
    {"testamentum", "will", "testament", "διαθηκη", "διαθήκη"},
    # 11. Procuratio (Power of attorney)
    {"procuratio", "power of attorney", "πληρεξουσιο", "πληρεξούσιο",
     "mandatum", "εξουσιωτικη", "εξουσιωτική"},
    # 12. Procedural acta
    {"procedural acta", "procedural acts", "διαδικαστικες πραξεις",
     "διαδικαστικές πράξεις", "renuntiatio", "revocatio", "cassatio",
     "rescissio", "declaratio", "protestatio", "notificatio"},
    # 13. Divisio / Partitio
    {"divisio", "divisio / partitio", "partitio", "division of property",
     "division", "διανομη περιουσιας", "διανομή περιουσίας"},
    # 14. Societas (Partnership)
    {"societas", "partnership", "societas agricola", "societas mercatoria"},
    # 15. Transactio (Settlement)
    {"transactio", "settlement", "συμβιβασμος", "συμβιβασμός"},
    # 16. Concessio ecclesiastica (distinct from Donatio!)
    {"concessio ecclesiastica", "ecclesiastical concession",
     "εκκλησιαστικη παραχωρηση", "εκκλησιαστική παραχώρηση"},
    # 17. Locatio conductio operis
    {"locatio conductio operis", "contract for work"},
    # 18. Contractus agrarius mixtus
    {"contractus agrarius mixtus", "mixed agricultural contract"},
    # 19. Dos (Dowry) — sometimes separate from Donatio
    {"dos", "instrumenta dotis", "dowry", "προικα", "προίκα", "donatio dotis"},
    # 20. Beneficium
    {"beneficium", "benefice"},
    # 21. Conventio
    {"conventio", "agreement", "συμφωνητικο", "συμφωνητικό"},
]

# ── Extension layer equivalences ───────────────────────────────────────────
EXTENSION_EQUIVALENCE = [
    {"venditio rei immobilis"},
    {"venditio rei mobilis"},
    {"venditio cum pacto"},
    {"venditio futurae rei", "forward agricultural sale"},
    {"donatio inter vivos"},
    {"donatio propter animam", "donatio ad pias causas"},
    {"donatio dotis"},
    {"emphyteusis perpetua"},
    {"emphyteusis ecclesiastica"},
    {"concessio cum canone"},
    {"locatio agricola", "colonia", "sharecropping", "metayage"},
    {"locatio vineae"},
    {"locatio ecclesiastica"},
    {"concessio temporalis"},
    {"mutuum cum pignore"},
    {"mutuum agrarium"},
    {"societas agricola", "agricultural partnership"},
    {"societas mercatoria"},
    {"conventio culturae", "cultivation agreement"},
    {"contractus agrarius mixtus", "mixed agricultural contract"},
    {"renuntiatio", "renunciation"},
    {"revocatio", "cassatio instrumenti", "cancellation"},
    {"declaratio", "declaration"},
    {"confessio", "acknowledgment"},
    {"protestatio", "protest"},
    {"transactio", "settlement", "συμβιβασμός"},
    {"hypotheca", "mortgage"},
    {"pignus", "pledge"},
    {"cessio debiti", "cessio iuris", "assignment"},
    {"divisio bonorum"},
    {"servitus", "easement"},
]


def normalize(text: str) -> str:
    """Lowercase, strip, remove parentheticals, collapse whitespace."""
    if not text or str(text) == "nan":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s*\(.*?\)\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_core_group(text: str) -> int:
    """Return index of matching core equivalence group, or -1."""
    normed = normalize(text)
    if not normed or normed in ("error", "missing", "not_run", ""):
        return -1
    for i, group in enumerate(CORE_EQUIVALENCE):
        if normed in group:
            return i
        for term in group:
            if term == normed or normed == term:
                return i
    # Fuzzy: check containment
    for i, group in enumerate(CORE_EQUIVALENCE):
        for term in group:
            if len(term) > 4 and (term in normed or normed in term):
                return i
    return -1


def find_ext_group(text: str) -> int:
    """Return index of matching extension equivalence group, or -1."""
    normed = normalize(text)
    if not normed or normed in ("error", "missing", "null", "none", ""):
        return -1
    for i, group in enumerate(EXTENSION_EQUIVALENCE):
        if normed in group:
            return i
        for term in group:
            if len(term) > 4 and (term in normed or normed in term):
                return i
    return -1


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denom
    lo = max(0, center - spread)
    hi = min(1, center + spread)
    return (lo * 100, hi * 100)


def score_core(pred_core: str, gt_row: pd.Series) -> str:
    """Score core_layer prediction. Returns CORRECT, INCORRECT, or ERROR."""
    pred_norm = normalize(pred_core)
    if pred_norm in ("error", "missing", ""):
        return "ERROR"

    pred_group = find_core_group(pred_core)

    # Check against all GT core columns
    gt_core_cols = ["CORE CATEGORY_LA", "CORE CATEGORY_EN", "CORE CATEGORY_GR"]
    for col in gt_core_cols:
        val = gt_row.get(col, "")
        if pd.notna(val) and str(val).strip():
            if normalize(str(val)) == pred_norm:
                return "CORRECT"
            if pred_group >= 0 and find_core_group(str(val)) == pred_group:
                return "CORRECT"

    return "INCORRECT"


def score_extension(pred_ext: str, gt_row: pd.Series) -> str:
    """Score extension_layer prediction. Returns CORRECT, PARTIAL, N/A, or INCORRECT."""
    pred_norm = normalize(pred_ext)
    gt_ext_cols = ["Subcategory_LA", "Subcategory_EN", "Subcategory_GR"]

    gt_exts = []
    for col in gt_ext_cols:
        val = gt_row.get(col, "")
        if pd.notna(val) and str(val).strip():
            gt_exts.append(normalize(str(val)))

    # If GT has no extension and pred has none → both correct
    if not gt_exts and (not pred_norm or pred_norm in ("null", "none")):
        return "CORRECT"
    # If GT has no extension but pred has one → N/A (not penalized)
    if not gt_exts:
        return "N/A"
    # If GT has extension but pred is empty
    if not pred_norm or pred_norm in ("null", "none"):
        return "INCORRECT"

    # Direct match
    if pred_norm in gt_exts:
        return "CORRECT"

    # Equivalence group match
    pred_group = find_ext_group(pred_ext)
    if pred_group >= 0:
        for gt_val in gt_exts:
            if find_ext_group(gt_val) == pred_group:
                return "CORRECT"

    # Containment check → partial
    for gt_val in gt_exts:
        if pred_norm in gt_val or gt_val in pred_norm:
            return "PARTIAL"

    return "INCORRECT"


def score_hybrid_tags(pred_tags: list, gt_row: pd.Series) -> str:
    """Score hybrid_tags prediction. Returns CORRECT, PARTIAL, or INCORRECT."""
    gt_hybrid = gt_row.get("hybrid_tags", "")
    if pd.isna(gt_hybrid) or not str(gt_hybrid).strip():
        gt_tags = []
    else:
        gt_tags = [normalize(t.strip()) for t in str(gt_hybrid).split(",") if t.strip()]

    pred_tags_norm = [normalize(str(t)) for t in (pred_tags or []) if str(t).strip()]

    # Both empty → correct
    if not gt_tags and not pred_tags_norm:
        return "CORRECT"
    # One empty, other not
    if not gt_tags and pred_tags_norm:
        return "N/A"  # extra tags not penalized
    if gt_tags and not pred_tags_norm:
        return "INCORRECT"

    # Check overlap
    matches = 0
    for gt_t in gt_tags:
        for pred_t in pred_tags_norm:
            if gt_t == pred_t or gt_t in pred_t or pred_t in gt_t:
                matches += 1
                break
    if matches == len(gt_tags):
        return "CORRECT"
    elif matches > 0:
        return "PARTIAL"
    return "INCORRECT"


def evaluate_single_run(run_dir, gt_lookup, bench_acts):
    """Evaluate a single run directory. Returns counters dict."""
    all_results = {}
    for strategy in STRATEGIES:
        for model_key in MODELS:
            fname = f"v3_results_{strategy}_{model_key}.json"
            fpath = os.path.join(run_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    all_results[(strategy, model_key)] = json.load(f)

    # Legacy filenames for Claude
    for strategy in STRATEGIES:
        key = (strategy, "claude")
        if key not in all_results:
            legacy = os.path.join(run_dir, f"v3_results_{strategy}.json")
            if os.path.exists(legacy):
                with open(legacy, "r", encoding="utf-8") as f:
                    all_results[key] = json.load(f)

    counters = {}
    detail_rows = []
    for s in STRATEGIES:
        for m in MODELS:
            counters[(s, m)] = {
                "core": defaultdict(int),
                "extension": defaultdict(int),
                "hybrid": defaultdict(int),
                "missing": False,
            }

    for idx, bench_act in enumerate(bench_acts):
        corpus = bench_act["corpus"]
        act_id = str(bench_act["act_id"])
        gt_key = (corpus.upper(), act_id)
        gt_row = gt_lookup.get(gt_key)
        if gt_row is None:
            continue

        gt_core_la = gt_row.get("CORE CATEGORY_LA", "")
        gt_core_en = gt_row.get("CORE CATEGORY_EN", "")
        gt_sub_la = gt_row.get("Subcategory_LA", "")

        row = {
            "Notary": corpus, "Act_ID": act_id,
            "GT_Core_LA": gt_core_la if pd.notna(gt_core_la) else "",
            "GT_Core_EN": gt_core_en if pd.notna(gt_core_en) else "",
            "GT_Sub_LA": gt_sub_la if pd.notna(gt_sub_la) else "",
        }

        for strategy in STRATEGIES:
            for model_key in MODELS:
                combo = (strategy, model_key)
                prefix = f"{strategy}_{model_key}"

                if combo in all_results and idx < len(all_results[combo]):
                    pred = all_results[combo][idx]
                    pred_core = pred.get("core_layer", "MISSING")
                    pred_ext = pred.get("extension_layer", "")
                    pred_hyb = pred.get("hybrid_tags", [])

                    sc_core = score_core(pred_core, gt_row)
                    sc_ext = score_extension(str(pred_ext) if pred_ext else "", gt_row)
                    sc_hyb = score_hybrid_tags(pred_hyb, gt_row)

                    row[f"{prefix}_pred_core"] = pred_core
                    row[f"{prefix}_pred_ext"] = pred_ext
                    row[f"{prefix}_core"] = sc_core
                    row[f"{prefix}_ext"] = sc_ext
                    row[f"{prefix}_hyb"] = sc_hyb

                    counters[combo]["core"][sc_core] += 1
                    counters[combo]["extension"][sc_ext] += 1
                    counters[combo]["hybrid"][sc_hyb] += 1
                else:
                    row[f"{prefix}_pred_core"] = "NOT_RUN"
                    row[f"{prefix}_core"] = "MISSING"
                    counters[combo]["missing"] = True

        detail_rows.append(row)

    return counters, detail_rows, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model × strategy results against ground truth"
    )
    parser.add_argument("--runs", nargs="+", type=int, default=[1],
                        help="Run numbers to evaluate (e.g. --runs 1 2 3)")
    args = parser.parse_args()

    run_numbers = args.runs
    multi_run = len(run_numbers) > 1

    # ── Load ground truth ──
    print("Loading ground truth...")
    df_gt = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET)
    df_gt = df_gt.dropna(subset=["Act ID"])
    df_gt["Act ID"] = df_gt["Act ID"].apply(lambda x: str(int(float(x))))
    print(f"  {len(df_gt)} ground truth acts")

    gt_lookup = {}
    for _, row in df_gt.iterrows():
        key = (str(row["Notary"]).upper().strip(), str(row["Act ID"]))
        gt_lookup[key] = row

    with open(BENCHMARK_ACTS, "r", encoding="utf-8") as f:
        bench_acts = json.load(f)
    print(f"  {len(bench_acts)} benchmark acts")

    # ── Evaluate each run ──
    all_run_counters = []
    all_run_details = []
    all_run_results = []

    for run_num in run_numbers:
        run_dir = os.path.join(OUTPUT_BASE, f"run{run_num}")
        if not os.path.exists(run_dir):
            print(f"\n  WARNING: {run_dir} not found, skipping run {run_num}")
            continue
        print(f"\nEvaluating run {run_num} from {run_dir}...")
        counters, detail_rows, all_results = evaluate_single_run(
            run_dir, gt_lookup, bench_acts)
        all_run_counters.append(counters)
        all_run_details.append(detail_rows)
        all_run_results.append(all_results)

        # Count loaded files
        n_loaded = sum(1 for k, v in all_results.items() if v)
        print(f"  Loaded {n_loaded}/16 result files")

    if not all_run_counters:
        print("ERROR: No valid runs found!")
        sys.exit(1)

    # ── Build accuracy matrix ──
    # For each (strategy, model), compute accuracy per run, then mean ± std
    print(f"\n{'='*100}")
    if multi_run:
        print(f"  CORE LAYER ACCURACY — {len(all_run_counters)} RUNS (mean ± std)")
    else:
        print(f"  CORE LAYER ACCURACY (%) with 95% Wilson CI")
    print(f"{'='*100}")
    print(f"{'Strategy':<15}", end="")
    for m in MODELS:
        print(f"{MODEL_LABELS[m]:>24}", end="")
    print()
    print("-" * 111)

    summary_rows = []
    for strategy in STRATEGIES:
        row = {"Strategy": strategy.upper()}
        print(f"{strategy.upper():<15}", end="")

        for model_key in MODELS:
            label = MODEL_LABELS[model_key]
            accs = []
            for counters in all_run_counters:
                c = counters[(strategy, model_key)]
                if c["missing"]:
                    continue
                correct = c["core"]["CORRECT"]
                total = correct + c["core"]["INCORRECT"] + c["core"]["ERROR"]
                if total > 0:
                    accs.append(correct / total * 100)

            if not accs:
                print(f"{'NOT RUN':>24}", end="")
                row[f"{label} Accuracy%"] = "NOT RUN"
                row[f"{label} Detail"] = "—"
            elif multi_run:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                runs_str = "/".join(f"{a:.0f}" for a in accs)
                print(f"{mean_acc:>6.1f}% ± {std_acc:>4.1f}", end="")
                row[f"{label} Accuracy%"] = f"{mean_acc:.1f} ± {std_acc:.1f}"
                row[f"{label} Per-Run"] = runs_str
            else:
                acc = accs[0]
                c = all_run_counters[0][(strategy, model_key)]
                correct = c["core"]["CORRECT"]
                total = correct + c["core"]["INCORRECT"] + c["core"]["ERROR"]
                lo, hi = wilson_ci(correct, total)
                print(f"{acc:>6.1f}% [{lo:.0f}-{hi:.0f}]", end="")
                row[f"{label} Accuracy%"] = round(acc, 1)
                row[f"{label} 95% CI"] = f"[{lo:.1f}–{hi:.1f}]"
                row[f"{label} C/I/E"] = f"{correct}/{c['core']['INCORRECT']}/{c['core']['ERROR']}"
        print()
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # Use the last run for detailed breakdowns
    counters = all_run_counters[-1]
    detail_rows = all_run_details[-1]
    all_results = all_run_results[-1]

    # ── Extension Layer Summary ──
    ext_rows = []
    for strategy in STRATEGIES:
        row = {"Strategy": strategy.upper()}
        for model_key in MODELS:
            c = counters[(strategy, model_key)]
            label = MODEL_LABELS[model_key]
            if c["missing"]:
                row[f"{label}"] = "NOT RUN"
            else:
                ec = c["extension"]
                correct = ec["CORRECT"]
                partial = ec["PARTIAL"]
                incorrect = ec["INCORRECT"]
                na = ec["N/A"]
                scored = correct + partial + incorrect
                if scored > 0:
                    row[f"{label}"] = f"{correct}/{partial}/{incorrect} ({correct/scored*100:.0f}% of {scored} scored)"
                else:
                    row[f"{label}"] = f"All N/A ({na})"
        ext_rows.append(row)
    df_extension = pd.DataFrame(ext_rows)

    # ── Per-Notary Breakdown ──
    notary_rows = []
    for corpus in ["ALEXAKIS", "FARMAKIS", "SPYRIS", "TOXOTIS", "VARAGAS"]:
        for strategy in STRATEGIES:
            row = {"Notary": corpus, "Strategy": strategy.upper()}
            for model_key in MODELS:
                prefix = f"{strategy}_{model_key}"
                acts = [r for r in detail_rows if r["Notary"] == corpus]
                c = sum(1 for a in acts if a.get(f"{prefix}_core") == "CORRECT")
                i = sum(1 for a in acts if a.get(f"{prefix}_core") == "INCORRECT")
                e = sum(1 for a in acts if a.get(f"{prefix}_core") == "ERROR")
                total = c + i + e
                label = MODEL_LABELS[model_key]
                if any(a.get(f"{prefix}_core") == "MISSING" for a in acts):
                    row[label] = "NOT RUN"
                elif total > 0:
                    row[label] = f"{c}/{i}/{e} ({c/total*100:.0f}%)"
                else:
                    row[label] = "—"
            notary_rows.append(row)
    df_notary = pd.DataFrame(notary_rows)

    # ── Per-Category Breakdown ──
    cat_rows = []
    cat_acts = defaultdict(list)
    for r in detail_rows:
        cat_acts[r["GT_Core_LA"]].append(r)

    for cat in sorted(cat_acts.keys()):
        if not cat:
            continue
        acts = cat_acts[cat]
        row = {"Category": cat, "N": len(acts)}
        for model_key in MODELS:
            for strategy in STRATEGIES:
                prefix = f"{strategy}_{model_key}"
                c = sum(1 for a in acts if a.get(f"{prefix}_core") == "CORRECT")
                total = sum(1 for a in acts if a.get(f"{prefix}_core") in ("CORRECT", "INCORRECT", "ERROR"))
                label = f"{MODEL_LABELS[model_key]}_{strategy}"
                if total > 0:
                    row[label] = f"{c}/{total}"
                else:
                    row[label] = "—"
        cat_rows.append(row)
    df_category = pd.DataFrame(cat_rows)

    # ── Coverage ──
    coverage_rows = []
    for strategy in STRATEGIES:
        row = {"Strategy": strategy.upper()}
        for model_key in MODELS:
            label = MODEL_LABELS[model_key]
            if (strategy, model_key) in all_results:
                preds = all_results[(strategy, model_key)]
                n_ok = sum(1 for r in preds if r.get("core_layer") not in ("ERROR", "MISSING", None))
                row[label] = f"{n_ok}/40"
            else:
                row[label] = "—"
        coverage_rows.append(row)
    df_coverage = pd.DataFrame(coverage_rows)

    # ── Write XLSX ──
    df_detail = pd.DataFrame(detail_rows)
    run_label = f"runs{'_'.join(str(r) for r in run_numbers)}"
    output_xlsx = os.path.join(OUTPUT_BASE, f"evaluation_{run_label}.xlsx")

    print(f"\nWriting {output_xlsx}...")
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Core Accuracy", index=False)
        df_extension.to_excel(writer, sheet_name="Extension Accuracy", index=False)
        df_category.to_excel(writer, sheet_name="Per-Category", index=False)
        df_notary.to_excel(writer, sheet_name="Per-Notary", index=False)
        df_detail.to_excel(writer, sheet_name="Per-Act Detail", index=False)
        df_coverage.to_excel(writer, sheet_name="Coverage", index=False)

    print(f"Done! Results in {output_xlsx}")
    print(f"\n  n=40 acts, 17 categories")
    if multi_run:
        print(f"  {len(all_run_counters)} runs evaluated, showing mean ± std")
    else:
        print(f"  CI = Wilson score interval at 95% confidence")


if __name__ == "__main__":
    main()
