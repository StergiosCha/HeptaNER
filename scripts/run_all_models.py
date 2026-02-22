#!/usr/bin/env python3
"""
Run all 4 models × 4 strategies on the 40 benchmark acts.
Produces 16 JSON result files per run in outputs/v3_evaluation/runN/.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    cd heptanesian_ner
    python scripts/run_all_models.py --run 1               # first run
    python scripts/run_all_models.py --run 2               # second run
    python scripts/run_all_models.py --run 3               # third run
    python scripts/run_all_models.py --run 1 --force       # re-run everything
    python scripts/run_all_models.py --run 1 --models claude gpt4o
    python scripts/run_all_models.py --run 1 --strategies zero_shot few_shot
    python scripts/run_all_models.py --run 1 --temp 0.3    # custom temperature

Notes:
    - All models are called via OpenRouter (set OPENROUTER_API_KEY)
    - Default temperature is 0.3 (slight randomness for variance across runs)
    - Use --temp 0.0 for deterministic single-run results
    - Each run saves to outputs/v3_evaluation/run{N}/ to keep them separate
"""

import json
import os
import sys
import re
import argparse
import time
from typing import List, Dict

# Add scripts dir to path so we can import siblings
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from openrouter_client import OpenRouterClient
from symbolic_engine import run_symbolic_engine, format_evidence_for_prompt

# ── Configuration ──────────────────────────────────────────────────────────
INPUT_ACTS = "data/benchmark_40_acts.json"
INPUT_TAXONOMY = "data/katigoriopoiisi.txt"
OUTPUT_BASE = "outputs/v3_evaluation"

# Models to evaluate (short_key -> OpenRouter model string)
MODELS = {
    "claude":    "anthropic/claude-3.7-sonnet",
    "claude4":   "anthropic/claude-sonnet-4",
    "claude4o":  "anthropic/claude-opus-4",
    "gpt4o":     "openai/gpt-4o",
    "gpt41":     "openai/gpt-4.1",
    "o3mini":    "openai/o3-mini",
    "gemini":    "google/gemini-2.0-flash-lite-001",
    "llama":     "meta-llama/llama-3.3-70b-instruct",
    "llama8b":   "meta-llama/llama-3.1-8b-instruct",
    "deepseek":  "deepseek/deepseek-chat",
    "deepseekr": "deepseek/deepseek-r1",
    "qwen":      "qwen/qwen-2.5-72b-instruct",
    "gemini25":  "google/gemini-2.5-flash",
    "mistral":   "mistralai/mistral-large-2512",
    "gpt52":     "openai/gpt-5.2-chat",
    "llama4":    "meta-llama/llama-4-maverick",
}

STRATEGIES = ["zero_shot", "few_shot", "full_context", "nesy"]

# Delay between API calls (seconds) – adjust if you hit rate limits
API_DELAY = 1.0
DEFAULT_TEMP = 0.0  # Deterministic by default; use --temp 0.3 for variance runs


# ── Prompt Builders ────────────────────────────────────────────────────────

def create_zero_shot_prompt(act_text: str) -> str:
    return f"""You are an expert archivist in 16th-century Heptanesian Diplomatics.
Classify the following notarial act into exactly ONE of these legal genres:

VALID CORE CATEGORIES:
1. Venditio (Sale)
2. Donatio (Donation)
3. Permutatio (Exchange)
4. Emphyteusis
5. Locatio conductio rei (Lease)
6. Mutuum (Loan)
7. Praevenditio (Forward sale)
8. Quietantia (Receipt of payment)
9. Instrumentum obligationis (Debt acknowledgement)
10. Testamentum (Will)
11. Procuratio (Power of attorney)
12. Procedural acta (Procedural/declaratory acts)
13. Divisio / Partitio (Division of property)
14. Societas (Partnership)
15. Transactio (Settlement)
16. Concessio ecclesiastica (Ecclesiastical concession)
17. Locatio conductio operis (Contract for work)

Act Text:
"{act_text}"

You must output STRICT JSON with exactly three keys:
{{
  "core_layer": "[One of the Latin terms above]",
  "extension_layer": "A more specific subcategory if applicable, or null",
  "hybrid_tags": ["Any secondary legal actions present, or empty list"]
}}
"""


def create_few_shot_prompt(act_text: str) -> str:
    return f"""You are an expert archivist in 16th-century Heptanesian Diplomatics.
Classify notarial acts into their legal genre. Use the Latin term for the core category.

VALID CORE CATEGORIES:
Venditio, Donatio, Permutatio, Emphyteusis, Locatio conductio rei, Mutuum,
Praevenditio, Quietantia, Instrumentum obligationis, Testamentum, Procuratio,
Procedural acta, Divisio / Partitio, Societas, Transactio, Concessio ecclesiastica,
Locatio conductio operis.

--- EXAMPLES ---

Example 1 (Sale):
Text: "...επούλησεν και επαράδοσεν εις τον αγοραστήν το χωράφι του..."
Output:
{{
  "core_layer": "Venditio",
  "extension_layer": "venditio rei immobilis",
  "hybrid_tags": []
}}

Example 2 (Forward sale / Pre-sale of agricultural produce):
Text: "...έλαβε δουκάτα τρία και υπόσχεται να δώσει λάδι βαρέλια δύο εις τον ερχόμενον καιρόν..."
Output:
{{
  "core_layer": "Praevenditio",
  "extension_layer": "venditio futurae rei",
  "hybrid_tags": ["agricultural credit"]
}}

Example 3 (Emphyteusis / long-term land grant):
Text: "...του εμφύτευσε τον αμπελώνα εις τον αιώνα με κανόνιν δουκάτα δύο τον χρόνον..."
Output:
{{
  "core_layer": "Emphyteusis",
  "extension_layer": "emphyteusis perpetua",
  "hybrid_tags": []
}}

Example 4 (Procedural act — renunciation of rights):
Text: "...παραιτείται και αφίνει κάθε δίκαιον και αξίωσιν όπερ είχε..."
Output:
{{
  "core_layer": "Procedural acta",
  "extension_layer": "Renuntiatio",
  "hybrid_tags": []
}}

Example 5 (Ecclesiastical concession):
Text: "...οι επίτροποι της εκκλησίας αγίου Νικολάου παρεχώρησαν τον ελαιώνα εις αυτόν δια χρόνους δέκα με κανόνιν..."
Output:
{{
  "core_layer": "Concessio ecclesiastica",
  "extension_layer": null,
  "hybrid_tags": ["emphyteusis-like"]
}}

--- END EXAMPLES ---

Now classify the following act. Output STRICT JSON only:

Act Text:
"{act_text}"

Output JSON:
"""


def create_full_context_prompt(act_text: str, taxonomy: str) -> str:
    return f"""You are an expert archivist in 16th-century Heptanesian Diplomatics.
Below is the definitive 3-Layer Taxonomy for classifying notarial documents.

VALID CORE CATEGORIES (use the Latin term):
Venditio, Donatio, Permutatio, Emphyteusis, Locatio conductio rei, Mutuum,
Praevenditio, Quietantia, Instrumentum obligationis, Testamentum, Procuratio,
Procedural acta, Divisio / Partitio, Societas, Transactio, Concessio ecclesiastica,
Locatio conductio operis, Contractus agrarius mixtus.

--- FULL TAXONOMY MANUAL (with extensions and hybrid layer) ---
{taxonomy}
--- END TAXONOMY MANUAL ---

Task: Read the following notarial document carefully. Identify the CORE legal nature
of the transaction, select the most precise extension/subcategory if applicable,
and note any hybrid features.

Act Text:
"{act_text}"

You must output STRICT JSON:
{{
  "core_layer": "[One Latin term from the core categories above]",
  "extension_layer": "[Exact subtype from the taxonomy, or null]",
  "hybrid_tags": ["Any hybrid features, or empty list"]
}}
"""


def create_nesy_prompt(act_text: str, taxonomy: str, symbolic_evidence: str) -> str:
    return f"""You are an expert archivist in 16th-century Heptanesian Diplomatics.
You have been provided with the definitive 3-Layer Taxonomy manual for classifying these documents.

--- TAXONOMY MANUAL ---
{taxonomy}
--- END TAXONOMY MANUAL ---

A deterministic symbolic engine has scanned the document for genre-diagnostic formulaic
phrases from Heptanesian notarial tradition. Its analysis is below.

--- SYMBOLIC ENGINE ANALYSIS ---
{symbolic_evidence}
--- END SYMBOLIC ENGINE ANALYSIS ---

INSTRUCTIONS:
- If the symbolic engine reports HIGH confidence, strongly prefer its candidate unless
  your reading of the full text clearly contradicts it.
- If MEDIUM confidence, treat the symbolic evidence as a strong prior but verify it
  against the full text and taxonomy.
- If LOW or NONE confidence, classify primarily based on your own reading of the text
  and the taxonomy manual.
- Always check: does the symbolic candidate match what the document actually describes?
  The engine can be misled by shared vocabulary.

Act Text:
"{act_text}"

You must output your classification in STRICT JSON format:
{{
  "core_layer": "[Select the exact Latin term from the taxonomy]",
  "extension_layer": "[Select the exact subtype if applicable, or null]",
  "hybrid_tags": ["List any hybrid features identified, or empty list"]
}}
"""


# ── Output Filename Convention ─────────────────────────────────────────────

def result_filename(strategy: str, model_key: str) -> str:
    """e.g. v3_results_zero_shot_claude.json"""
    return f"v3_results_{strategy}_{model_key}.json"


def result_path(strategy: str, model_key: str, output_dir: str) -> str:
    return os.path.join(output_dir, result_filename(strategy, model_key))


# ── Main Runner ────────────────────────────────────────────────────────────

def run_combination(client: OpenRouterClient, acts: list, taxonomy: str,
                    strategy: str, model_key: str,
                    model_id: str, api_delay: float = 1.0,
                    temperature: float = 0.3) -> list:
    """Run one strategy × model combination on all 40 acts."""
    results = []
    errors = 0

    for i, act in enumerate(acts):
        doc_id = str(act["act_id"])
        corpus = act["corpus"]
        text = act.get("content", act.get("text", ""))

        # Build prompt
        if strategy == "zero_shot":
            prompt = create_zero_shot_prompt(text)
        elif strategy == "few_shot":
            prompt = create_few_shot_prompt(text)
        elif strategy == "full_context":
            prompt = create_full_context_prompt(text, taxonomy)
        elif strategy == "nesy":
            sym_result = run_symbolic_engine(text)
            evidence_text = format_evidence_for_prompt(sym_result)
            prompt = create_nesy_prompt(text, taxonomy, evidence_text)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                temperature=temperature
            )

            if response is None:
                raise ValueError("API returned None")

            # Parse JSON from response — try multiple extraction strategies
            data = None
            json_str = None

            # Strategy 1: ```json ... ``` block
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            # Strategy 2: ``` ... ``` block
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            # Try parsing the extracted block
            if json_str:
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            # Strategy 3: Find first { ... } in the response
            if data is None:
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
                if match:
                    try:
                        data = json.loads(match.group())
                    except json.JSONDecodeError:
                        pass

            # Strategy 4: Try the whole response stripped
            if data is None:
                try:
                    data = json.loads(response.strip())
                except json.JSONDecodeError:
                    raise ValueError(f"Cannot parse JSON from response: {response[:200]}")
            data["act_id"] = doc_id
            data["corpus"] = corpus
            results.append(data)
            core = data.get("core_layer", "?")
            print(f"    [{i+1}/40] {corpus}/{doc_id} → {core}")

        except Exception as e:
            errors += 1
            print(f"    [{i+1}/40] {corpus}/{doc_id} → ERROR: {e}")
            results.append({
                "act_id": doc_id,
                "corpus": corpus,
                "core_layer": "ERROR",
                "extension_layer": "ERROR",
                "hybrid_tags": [],
                "error": str(e)
            })

        time.sleep(api_delay)

    print(f"    Done: {40 - errors}/40 successful, {errors} errors")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run all models × strategies on 40 benchmark acts"
    )
    parser.add_argument("--run", type=int, default=1,
                        help="Run number (1, 2, 3...) — results go to run{N}/ subdir")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results file already exists")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Which models to run (default: all)")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        choices=STRATEGIES,
                        help="Which strategies to run (default: all)")
    parser.add_argument("--delay", type=float, default=API_DELAY,
                        help="Delay between API calls in seconds")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                        help="Temperature for LLM calls (default: 0.0)")
    args = parser.parse_args()

    api_delay = args.delay
    temperature = args.temp
    output_dir = os.path.join(OUTPUT_BASE, f"run{args.run}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Load shared data ──
    print("Loading benchmark acts...")
    with open(INPUT_ACTS, "r", encoding="utf-8") as f:
        acts = json.load(f)
    print(f"  {len(acts)} acts loaded")

    print("Loading taxonomy...")
    with open(INPUT_TAXONOMY, "r", encoding="utf-8") as f:
        taxonomy = f.read()
    print(f"  {len(taxonomy)} chars")

    if "nesy" in args.strategies:
        print("NeSy symbolic engine loaded (genre-discriminative formula matching)")

    # ── Init client ──
    client = OpenRouterClient()
    print(f"\nOpenRouter client initialized. API key: ...{client.api_key[-6:]}")

    # ── Run all combos ──
    combos = [(s, m) for s in args.strategies for m in args.models]
    total = len(combos)
    done = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"  RUN {args.run}: {total} COMBINATIONS "
          f"({len(args.strategies)} strategies × {len(args.models)} models)")
    print(f"  Temperature: {temperature}  |  Output: {output_dir}")
    print(f"{'='*60}\n")

    for strategy, model_key in combos:
        model_id = MODELS[model_key]
        out_path = result_path(strategy, model_key, output_dir)

        # Skip if already done
        if os.path.exists(out_path) and not args.force:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            n_ok = sum(1 for r in existing if r.get("core_layer") != "ERROR")
            print(f"[{done+1}/{total}] SKIP {strategy} × {model_key} "
                  f"(already exists: {n_ok}/40 OK) — use --force to re-run")
            skipped += 1
            done += 1
            continue

        print(f"\n[{done+1}/{total}] RUNNING: {strategy} × {model_key} ({model_id})")
        print(f"  Output: {out_path}")

        results = run_combination(
            client, acts, taxonomy,
            strategy, model_key, model_id,
            api_delay=api_delay, temperature=temperature
        )

        # Save results
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        done += 1
        print(f"  Saved to {out_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  RUN {args.run} COMPLETE: {done - skipped} ran, {skipped} skipped")
    print(f"{'='*60}")
    print(f"\nResult files in {output_dir}/:")
    for strategy in args.strategies:
        for model_key in args.models:
            p = result_path(strategy, model_key, output_dir)
            status = "✓" if os.path.exists(p) else "✗"
            print(f"  {status} {result_filename(strategy, model_key)}")

    print(f"\nNext step: run evaluation with:")
    print(f"  python scripts/evaluate_all_results.py --runs 1 2 3")


if __name__ == "__main__":
    main()
