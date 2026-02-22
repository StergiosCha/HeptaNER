#!/usr/bin/env python3
"""
Error analysis and full classification dump.
Produces two outputs:
  1. error_analysis.txt   – misclassified acts, confusion patterns, hardest acts
  2. full_predictions.txt – every model's prediction vs gold for every act (for auditing the evaluator)

Usage:
    cd github_release
    python scripts/error_analysis.py
"""

import json, os, re
import pandas as pd
from collections import defaultdict

# ── Config ──
EXCEL_PATH = "data/testing_acts.xlsx"
EXCEL_SHEET = "Additional acts"
BENCHMARK_ACTS = "data/benchmark_40_acts.json"
RESULTS_DIR = "outputs/v3_evaluation/run1"
OUT_DIR = "outputs/v3_evaluation"

MODELS = ["claude", "claude4", "claude4o", "gpt4o", "gpt52", "llama", "llama8b",
          "llama4", "deepseek", "deepseekr", "qwen", "gemini25"]
MODEL_LABELS = {
    "claude": "Cl.3.7Son", "claude4": "Cl.Son4", "claude4o": "Cl.Opus4",
    "gpt4o": "GPT-4o", "gpt52": "GPT-5.2", "llama": "Ll.70B",
    "llama8b": "Ll.8B", "llama4": "Ll.Mav", "deepseek": "DSv3",
    "deepseekr": "DSr1", "qwen": "Qwen72", "gemini25": "Gem2.5",
}
MODEL_LABELS_LONG = {
    "claude": "Claude 3.7 Sonnet", "claude4": "Claude Sonnet 4",
    "claude4o": "Claude Opus 4", "gpt4o": "GPT-4o", "gpt52": "GPT-5.2 Chat",
    "llama": "Llama 3.3 70B", "llama8b": "Llama 3.1 8B",
    "llama4": "Llama 4 Maverick", "deepseek": "DeepSeek V3",
    "deepseekr": "DeepSeek R1", "qwen": "Qwen 2.5 72B",
    "gemini25": "Gemini 2.5 Flash",
}
STRATEGIES = ["zero_shot", "few_shot", "full_context", "nesy"]

# ── Core equivalence groups ──
CORE_EQUIVALENCE = [
    {"venditio", "sale", "πωληση", "πώληση", "emptio venditio", "emptio-venditio"},
    {"donatio", "donation", "δωρεα", "δωρεά", "donatio inter vivos", "donatio propter animam"},
    {"permutatio", "exchange", "ανταλλαγη", "ανταλλαγή"},
    {"emphyteusis", "εμφυτευση", "εμφύτευση", "emphyteusis perpetua", "emphyteusis ecclesiastica"},
    {"locatio conductio rei", "locatio conductio", "lease", "μισθωση", "μίσθωση",
     "εκμισθωση", "εκμίσθωση", "locatio agricola", "locatio vineae"},
    {"mutuum", "loan", "δανειο", "δάνειο", "foenus", "mutuum cum pignore"},
    {"praevenditio", "forward sale", "προπωληση", "προπώληση",
     "venditio futurae rei", "venditio futura"},
    {"quietantia", "receipt", "αποδειξη πληρωμης", "απόδειξη πληρωμής",
     "confessio solutionis", "apodixis"},
    {"instrumentum obligationis", "debt acknowledgement", "ομολογια χρεους",
     "ομολογία χρέους", "recognitio debiti"},
    {"testamentum", "will", "testament", "διαθηκη", "διαθήκη"},
    {"procuratio", "power of attorney", "πληρεξουσιο", "πληρεξούσιο",
     "mandatum", "εξουσιωτικη", "εξουσιωτική"},
    {"procedural acta", "procedural acts", "διαδικαστικες πραξεις",
     "διαδικαστικές πράξεις", "renuntiatio", "revocatio", "cassatio",
     "rescissio", "declaratio", "protestatio", "notificatio"},
    {"divisio", "divisio / partitio", "partitio", "division of property",
     "division", "διανομη περιουσιας", "διανομή περιουσίας"},
    {"societas", "partnership", "societas agricola", "societas mercatoria"},
    {"transactio", "settlement", "συμβιβασμος", "συμβιβασμός"},
    {"concessio ecclesiastica", "ecclesiastical concession",
     "εκκλησιαστικη παραχωρηση", "εκκλησιαστική παραχώρηση"},
    {"locatio conductio operis", "contract for work"},
    {"contractus agrarius mixtus", "mixed agricultural contract"},
    {"dos", "instrumenta dotis", "dowry", "προικα", "προίκα", "donatio dotis"},
    {"beneficium", "benefice"},
    {"conventio", "agreement", "συμφωνητικο", "συμφωνητικό"},
]


def normalize(text):
    if not text or str(text) == "nan":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s*\(.*?\)\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_core_group(text):
    normed = normalize(text)
    if not normed or normed in ("error", "missing", "not_run", ""):
        return -1
    for i, group in enumerate(CORE_EQUIVALENCE):
        if normed in group:
            return i
    for i, group in enumerate(CORE_EQUIVALENCE):
        for term in group:
            if len(term) > 4 and (term in normed or normed in term):
                return i
    return -1


def score_core(pred_core, gt_row):
    pred_norm = normalize(pred_core)
    if pred_norm in ("error", "missing", ""):
        return "ERROR"
    pred_group = find_core_group(pred_core)
    for col in ["CORE CATEGORY_LA", "CORE CATEGORY_EN", "CORE CATEGORY_GR"]:
        val = gt_row.get(col, "")
        if pd.notna(val) and str(val).strip():
            if normalize(str(val)) == pred_norm:
                return "CORRECT"
            if pred_group >= 0 and find_core_group(str(val)) == pred_group:
                return "CORRECT"
    return "INCORRECT"


def main():
    # Load data
    with open(BENCHMARK_ACTS) as f:
        benchmark = json.load(f)

    df_gt = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET)
    gt_lookup = {}
    for _, row in df_gt.iterrows():
        notary = str(row.get("Notary", "")).upper().strip()
        act_id = str(int(row["Act ID"])) if pd.notna(row.get("Act ID")) else ""
        if notary and act_id:
            gt_lookup[(notary, act_id)] = row

    # Load all predictions
    all_preds = {}
    for model in MODELS:
        for strat in STRATEGIES:
            fpath = os.path.join(RESULTS_DIR, f"v3_results_{strat}_{model}.json")
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                all_preds[(model, strat)] = json.load(f)

    # ════════════════════════════════════════════════════════════════════
    # FILE 1: full_predictions.txt – every prediction for every act
    # ════════════════════════════════════════════════════════════════════
    lines1 = []
    lines1.append("=" * 130)
    lines1.append("FULL PREDICTIONS: All 12 models x 4 strategies vs Gold (Core Layer)")
    lines1.append("For each act: + = correct, - = wrong, ! = API error")
    lines1.append("=" * 130)

    header = f"  {'Model':<10}" + "".join(f"{s:<30}" for s in STRATEGIES)

    for act in benchmark:
        corpus = act["corpus"].upper().strip()
        act_id = str(act["act_id"]).strip()
        gt_row = gt_lookup.get((corpus, act_id))
        gold_core = ""
        gold_ext = ""
        if gt_row is not None:
            gold_core = str(gt_row.get("CORE CATEGORY_LA", "")).strip()
            ext_val = gt_row.get("Subcategory_LA", "")
            gold_ext = "" if pd.isna(ext_val) else str(ext_val).strip()

        lines1.append("")
        lines1.append(f"--- Act {act_id} ({corpus})  |  GOLD: {gold_core}" +
                      (f"  /  {gold_ext}" if gold_ext else "") + " ---")
        lines1.append(header)

        for model in MODELS:
            parts = [f"  {MODEL_LABELS[model]:<10}"]
            for strat in STRATEGIES:
                key = (model, strat)
                if key not in all_preds:
                    parts.append(f"{'[NOT RUN]':<30}")
                    continue
                pred_core = "???"
                for r in all_preds[key]:
                    if (str(r.get("act_id", "")).strip() == act_id and
                            r.get("corpus", "").upper().strip() == corpus):
                        pred_core = r.get("core_layer", "???")
                        break
                if gt_row is not None:
                    verdict = score_core(pred_core, gt_row)
                else:
                    verdict = "??"
                m = "+" if verdict == "CORRECT" else ("-" if verdict == "INCORRECT" else "!")
                cell = f"{m} {pred_core}"
                if len(cell) > 28:
                    cell = cell[:27] + "~"
                parts.append(f"{cell:<30}")
            lines1.append("".join(parts))

    out1 = os.path.join(OUT_DIR, "full_predictions.txt")
    with open(out1, "w", encoding="utf-8") as f:
        f.write("\n".join(lines1))
    print(f"Wrote {out1} ({len(lines1)} lines)")

    # ════════════════════════════════════════════════════════════════════
    # FILE 2: error_analysis.txt
    # ════════════════════════════════════════════════════════════════════
    lines2 = []

    # ── Section A: Misclassified acts per strategy ──
    lines2.append("=" * 130)
    lines2.append("SECTION A: MISCLASSIFIED ACTS PER MODEL AND STRATEGY (Core Layer)")
    lines2.append("=" * 130)

    for strat in STRATEGIES:
        lines2.append(f"\n{'━' * 60} {strat.upper()} {'━' * 60}")
        for model in MODELS:
            key = (model, strat)
            if key not in all_preds:
                continue
            errors = []
            correct = api_errors = 0
            for r in all_preds[key]:
                corpus = r.get("corpus", "").upper().strip()
                act_id = str(r.get("act_id", "")).strip()
                gt_row = gt_lookup.get((corpus, act_id))
                if gt_row is None:
                    continue
                pred_core = r.get("core_layer", "")
                verdict = score_core(pred_core, gt_row)
                gold = str(gt_row.get("CORE CATEGORY_LA", "")).strip()
                if verdict == "CORRECT":
                    correct += 1
                elif verdict == "ERROR":
                    api_errors += 1
                else:
                    errors.append({"act_id": act_id, "corpus": corpus,
                                   "gold": gold, "predicted": pred_core})
            total = correct + len(errors) + api_errors
            pct = 100 * correct / total if total else 0
            label = MODEL_LABELS_LONG.get(model, model)
            lines2.append(f"\n  {label}: {correct}/{total} correct ({pct:.1f}%), "
                         f"{len(errors)} wrong, {api_errors} errors")
            for e in errors:
                lines2.append(f"    Act {e['act_id']:>3} ({e['corpus']:<10}): "
                             f"gold={e['gold']:<32} pred={e['predicted']}")

    # ── Section B: Confusion patterns (NeSy) ──
    lines2.append("\n\n" + "=" * 130)
    lines2.append("SECTION B: CONFUSION PATTERNS (NeSy strategy, aggregated across all 12 models)")
    lines2.append("=" * 130)

    confusion = defaultdict(int)
    for model in MODELS:
        key = (model, "nesy")
        if key not in all_preds:
            continue
        for r in all_preds[key]:
            corpus = r.get("corpus", "").upper().strip()
            act_id = str(r.get("act_id", "")).strip()
            gt_row = gt_lookup.get((corpus, act_id))
            if gt_row is None:
                continue
            verdict = score_core(r.get("core_layer", ""), gt_row)
            if verdict == "INCORRECT":
                gold = str(gt_row.get("CORE CATEGORY_LA", "")).strip()
                confusion[(gold, r.get("core_layer", "???"))] += 1

    lines2.append(f"\n  {'Gold':<35} {'Predicted as':<40} {'N':>3}")
    lines2.append(f"  {'─' * 35} {'─' * 40} {'─' * 3}")
    for (gold, pred), count in sorted(confusion.items(), key=lambda x: -x[1]):
        lines2.append(f"  {gold:<35} {pred:<40} {count:>3}")

    # ── Section C: Confusion patterns (all strategies) ──
    lines2.append("\n\n" + "=" * 130)
    lines2.append("SECTION C: CONFUSION PATTERNS (ALL strategies, aggregated across all 12 models)")
    lines2.append("=" * 130)

    confusion_all = defaultdict(int)
    for model in MODELS:
        for strat in STRATEGIES:
            key = (model, strat)
            if key not in all_preds:
                continue
            for r in all_preds[key]:
                corpus = r.get("corpus", "").upper().strip()
                act_id = str(r.get("act_id", "")).strip()
                gt_row = gt_lookup.get((corpus, act_id))
                if gt_row is None:
                    continue
                verdict = score_core(r.get("core_layer", ""), gt_row)
                if verdict == "INCORRECT":
                    gold = str(gt_row.get("CORE CATEGORY_LA", "")).strip()
                    confusion_all[(gold, r.get("core_layer", "???"))] += 1

    lines2.append(f"\n  {'Gold':<35} {'Predicted as':<40} {'N':>4}")
    lines2.append(f"  {'─' * 35} {'─' * 40} {'─' * 4}")
    for (gold, pred), count in sorted(confusion_all.items(), key=lambda x: -x[1]):
        lines2.append(f"  {gold:<35} {pred:<40} {count:>4}")

    # ── Section D: Hardest acts ──
    lines2.append("\n\n" + "=" * 130)
    lines2.append("SECTION D: HARDEST ACTS (NeSy: how many of the 12 models got each act wrong)")
    lines2.append("=" * 130)

    act_diff = defaultdict(lambda: {"correct": 0, "wrong": 0, "error": 0, "preds": []})
    for model in MODELS:
        key = (model, "nesy")
        if key not in all_preds:
            continue
        for r in all_preds[key]:
            corpus = r.get("corpus", "").upper().strip()
            act_id = str(r.get("act_id", "")).strip()
            gt_row = gt_lookup.get((corpus, act_id))
            if gt_row is None:
                continue
            verdict = score_core(r.get("core_layer", ""), gt_row)
            akey = (corpus, act_id)
            if verdict == "CORRECT":
                act_diff[akey]["correct"] += 1
            elif verdict == "ERROR":
                act_diff[akey]["error"] += 1
            else:
                act_diff[akey]["wrong"] += 1
                act_diff[akey]["preds"].append(
                    f"{MODEL_LABELS[model]}:{r.get('core_layer', '?')}")

    lines2.append(f"\n  {'Act':<20} {'Gold':<32} {'OK':>3} {'Bad':>4} {'Err':>4}  "
                  f"Wrong predictions")
    lines2.append(f"  {'─' * 20} {'─' * 32} {'─' * 3} {'─' * 4} {'─' * 4}  {'─' * 60}")
    for (corpus, act_id), info in sorted(act_diff.items(), key=lambda x: -x[1]["wrong"]):
        if info["wrong"] == 0:
            continue
        gt_row = gt_lookup.get((corpus, act_id))
        gt_row = gt_lookup.get((corpus, act_id))
        gold = str(gt_row.get("CORE CATEGORY_LA", "")).strip() if gt_row is not None else "???"
        preds_str = "; ".join(info["preds"][:8])
        if len(info["preds"]) > 8:
            preds_str += f" (+{len(info['preds']) - 8} more)"
        lines2.append(f"  {act_id:>3} ({corpus:<10})    {gold:<32} {info['correct']:>3} "
                     f"{info['wrong']:>4} {info['error']:>4}  {preds_str}")

    # ── Section E: Hardest acts across ALL strategies ──
    lines2.append("\n\n" + "=" * 130)
    lines2.append("SECTION E: HARDEST ACTS (ALL strategies x 12 models = 48 attempts per act)")
    lines2.append("=" * 130)

    act_diff_all = defaultdict(lambda: {"correct": 0, "wrong": 0, "error": 0})
    for model in MODELS:
        for strat in STRATEGIES:
            key = (model, strat)
            if key not in all_preds:
                continue
            for r in all_preds[key]:
                corpus = r.get("corpus", "").upper().strip()
                act_id = str(r.get("act_id", "")).strip()
                gt_row = gt_lookup.get((corpus, act_id))
                if gt_row is None:
                    continue
                verdict = score_core(r.get("core_layer", ""), gt_row)
                akey = (corpus, act_id)
                if verdict == "CORRECT":
                    act_diff_all[akey]["correct"] += 1
                elif verdict == "ERROR":
                    act_diff_all[akey]["error"] += 1
                else:
                    act_diff_all[akey]["wrong"] += 1

    lines2.append(f"\n  {'Act':<20} {'Gold':<32} {'OK':>3} {'Bad':>4} {'Err':>4}  "
                  f"{'%Wrong':>7}")
    lines2.append(f"  {'─' * 20} {'─' * 32} {'─' * 3} {'─' * 4} {'─' * 4}  {'─' * 7}")
    for (corpus, act_id), info in sorted(act_diff_all.items(),
                                          key=lambda x: -x[1]["wrong"]):
        gt_row = gt_lookup.get((corpus, act_id))
        gold = str(gt_row.get("CORE CATEGORY_LA", "")).strip() if gt_row is not None else "???"
        total = info["correct"] + info["wrong"] + info["error"]
        pct_wrong = 100 * info["wrong"] / total if total else 0
        lines2.append(f"  {act_id:>3} ({corpus:<10})    {gold:<32} {info['correct']:>3} "
                     f"{info['wrong']:>4} {info['error']:>4}  {pct_wrong:>6.1f}%")

    out2 = os.path.join(OUT_DIR, "error_analysis.txt")
    with open(out2, "w", encoding="utf-8") as f:
        f.write("\n".join(lines2))
    print(f"Wrote {out2} ({len(lines2)} lines)")


if __name__ == "__main__":
    main()
