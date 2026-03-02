# HeptaTAX: Neuro-Symbolic Genre Classification of 16th-Century Heptanesian Notarial Acts

Code, data, and reproduction materials for the paper submitted to the [DialRes 2026 Workshop](https://sites.google.com/view/dialres2026/) (LREC-COLING 2026, Palma de Mallorca).

## Overview

This repository provides a benchmark and evaluation pipeline for classifying early modern Greek notarial acts into legal genres (e.g. sale, will, dowry contract) using LLMs. The core contribution is a **neuro-symbolic (NeSy) prompting strategy** that pairs a deterministic symbolic engine (~70 genre-discriminative rules over formulaic legal phrases) with LLM classification. We evaluate 12 LLMs across 4 prompting strategies on a 40-act benchmark annotated at three levels: core genre, extension subcategory, and hybrid cross-cutting tags.

## Repository Structure

```
.
├── scripts/
│   ├── run_all_models.py          # Main runner: queries LLMs via OpenRouter
│   ├── evaluate_all_results.py    # Evaluation: scores all 3 annotation tiers
│   ├── symbolic_engine.py         # Deterministic rule engine (~70 rules)
│   ├── openrouter_client.py       # OpenRouter API client
│   └── __init__.py
├── data/
│   ├── benchmark_40_acts.json     # 40 annotated benchmark acts (8 per notary)
│   ├── katigoriopoiisi.txt        # Genre taxonomy (192 lines)
│   ├── testing_acts.xlsx          # Ground truth spreadsheet
│   └── new_corpus_raw/            # Raw transcriptions (907 acts, 5 notaries)
├── outputs/
│   └── v3_evaluation/
│       ├── run1/                  # Result JSONs (model × strategy)
│       └── evaluation_runs1_2_3.xlsx  # Compiled evaluation results
├── paper/
│   └── dialres2026_draft.tex      # Workshop paper (LREC 2026 format)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenRouter API key:

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=sk-or-v1-your-actual-key
```

## Reproducing Results

### 1. Run all models (requires OpenRouter API key and credits)

```bash
# Run all 12 models × 4 strategies (run 1)
python scripts/run_all_models.py --run 1

# Run specific models only
python scripts/run_all_models.py --run 1 --models gpt4o claude4 llama

# Force re-run even if results exist
python scripts/run_all_models.py --run 1 --models gpt4o --force
```

### 2. Evaluate

```bash
# Evaluate run 1
python scripts/evaluate_all_results.py --runs 1

# Evaluate multiple runs (for variance analysis)
python scripts/evaluate_all_results.py --runs 1 2 3
```

Results are written to `outputs/v3_evaluation/evaluation_runs1_2_3.xlsx`.

## Models

The pipeline supports 12 LLMs via OpenRouter:

| Key | Model | Provider |
|-----|-------|----------|
| `claude` | Claude 3.7 Sonnet | Anthropic |
| `claude4` | Claude Sonnet 4 | Anthropic |
| `claude4o` | Claude Opus 4 | Anthropic |
| `gpt4o` | GPT-4o | OpenAI |
| `gpt52` | GPT-5.2 Chat | OpenAI |
| `llama` | Llama 3.3 70B | Meta |
| `llama8b` | Llama 3.1 8B | Meta |
| `llama4` | Llama 4 Maverick | Meta |
| `deepseek` | DeepSeek V3 | DeepSeek |
| `deepseekr` | DeepSeek R1 | DeepSeek |
| `qwen` | Qwen 2.5 72B | Alibaba |
| `gemini25` | Gemini 2.5 Flash | Google |

## Prompting Strategies

1. **zero_shot** -- genre taxonomy only, no examples
2. **few_shot** -- taxonomy + 5 worked examples
3. **full_context** -- taxonomy + all 40 benchmark acts with gold labels
4. **nesy** -- taxonomy + symbolic engine evidence + 5 examples

## Annotation Tiers

- **Core layer**: 17 legal genre categories (e.g. Πώληση, Διαθήκη, Προικοσύμφωνο)
- **Extension layer**: subcategories within core genres (24/40 acts annotated)
- **Hybrid tags**: cross-cutting features that co-occur with any genre

## Data

- **benchmark_40_acts.json**: 40 acts sampled from 5 notaries (Alexakis, Farmakis, Spyris, Toxotis, Varagas), 8 acts each. Fields: `corpus`, `act_id`, `header`, `content`, `diagnostic_formulas`, `gold_genre`, `gold_genre_raw`.
- **katigoriopoiisi.txt**: The complete genre taxonomy used for classification (Greek/Latin labels).
- **testing_acts.xlsx**: Ground truth annotations at all three tiers.
- **new_corpus_raw/**: Raw transcription files from 5 notarial registers (907 acts total), used for the symbolic engine scalability test (Section 6 in the paper).

## License

[TBD]

## Citation

[TBD -- add after acceptance]
