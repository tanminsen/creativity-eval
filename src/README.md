# `src/`

Source modules for the automated creativity evaluation framework. Grouped
by role:

**Data**
- `data.py` — downloads the MacGyver problem–solution corpus and formats
  the step-wise prompt (paper §5).
- `read_data.py` — small JSON loader helper.

**Generation backends**
- `llama_funcs.py` — HuggingFace Transformers wrappers for Llama, Vicuna
  and Mistral-family models; returns per-token logprobs and optional
  hidden states.
- `openai_funcs.py` — OpenAI client wrappers for entailment, factuality
  and logprob-carrying candidate sampling.

**Divergent creativity (Semantic Entropy)**
- `dabertaMNLI.py` — bidirectional entailment via
  `tasksource/deberta-base-long-nli` (paper Appendix C.3.3).
- `helper_funcs.py` — semantic-entropy variants, length-normalised
  log-prob aggregation, class-probability reduction, scoring utilities
  (paper §3, Eqs. 1–5; Appendix C.1).

**Convergent creativity (Multi-agent judge)**
- `LLMevalframeworks.py` — retrieval-based multi-agent judge
  (Problem / Solution / Criterion analysts, ChromaDB fragment store,
  confidence-based early exit at T=0.5; paper §4 and Appendix D).
  Also contains the single-agent and ChatEval baselines used in Table 2.

**Per-model runners**
- `GPT_run_benchmark.py`, `Llama_run_benchmark.py`,
  `Mixtral_run_benchmark.py`, `vicuna_run_benchmark.py` — step-wise
  benchmark driver for each model family (paper §5).

**Post-processing**
- `process_data.py` — aggregates per-step semantic-entropy values and runs
  the convergent-creativity judge. Imported for side-effects by
  `export_data.py`.
