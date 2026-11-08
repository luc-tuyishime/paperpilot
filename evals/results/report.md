# PaperPilot evaluation

| Metric | Score | Threshold | |
|---|---|---|---|
| retrieval_hit | 1.00 | 0.80 | ✅ |
| citation_hit | 1.00 | 0.70 | ✅ |
| answer_correct | 1.00 | 0.70 | ✅ |
| refusal_correct | 1.00 | 0.80 | ✅ |

16 questions · avg LLM latency 877 ms · 25069 tokens

| Question | Retrieved | Cited | Correct | Refusal | Best score |
|---|---|---|---|---|---|
| `transformer-bleu-en-de` | ✅ | ✅ | ✅ | ✅ | 0.784 |
| `transformer-training-time` | ✅ | ✅ | ✅ | ✅ | 0.731 |
| `transformer-label-smoothing` | ✅ | ✅ | ✅ | ✅ | 0.756 |
| `transformer-heads` | ✅ | ✅ | ✅ | ✅ | 0.807 |
| `bahdanau-model-name` | ✅ | ✅ | ✅ | ✅ | 0.779 |
| `bahdanau-encoder` | ✅ | ✅ | ✅ | ✅ | 0.759 |
| `gan-two-models` | ✅ | ✅ | ✅ | ✅ | 0.836 |
| `gan-game` | ✅ | ✅ | ✅ | ✅ | 0.697 |
| `gat-citation-datasets` | ✅ | ✅ | ✅ | ✅ | 0.797 |
| `gat-inductive` | ✅ | ✅ | ✅ | ✅ | 0.795 |
| `rag-memories` | ✅ | ✅ | ✅ | ✅ | 0.759 |
| `rag-retriever` | ✅ | ✅ | ✅ | ✅ | 0.754 |
| `rag-formulations` | ✅ | ✅ | ✅ | ✅ | 0.730 |
| `unanswerable-capital` | – | – | – | ✅ | 0.548 |
| `unanswerable-gpt4` | – | – | – | ✅ | 0.719 |
| `unanswerable-world-cup` | – | – | – | ✅ | 0.551 |
