# MultiHop-RAG Topic-Partitioned CRAG Evaluation

This repository contains a standalone Topic-Partitioned CRAG evaluation on the MultiHop-RAG dataset. It combines Kiraffe-style BERTopic partitioning with a CRAG-style retrieval evaluator and fallback branch.

## Experiment

Dataset: `yixuantt/MultiHopRAG`

RAG method: Topic-Partitioned CRAG

Local model: Ollama `gemma2:2b`

Topic partitioning:

- BERTopic `nr_topics=auto`
- 7 real topic partitions plus 1 outlier/unclassified bucket
- 8 total topic buckets including outliers

Pipeline:

1. Build a full-corpus vector index from the MultiHop-RAG corpus.
2. Build BERTopic topic partitions using Kiraffe-style automatic topic reduction.
3. Attach topic IDs to all corpus chunks.
4. Extract query knowledge points and route to the closest topic profiles.
5. Retrieve from selected topic partitions.
6. Apply CRAG relevance evaluation to classify retrieval as correct, ambiguous, or incorrect.
7. For ambiguous/incorrect branches, use the full corpus as the no-web-search fallback source.
8. Generate a short answer with local Ollama `gemma2:2b`.
9. Evaluate with EM, Acc, F1, G-Sem, and Tok.

No external web search API is used. The CRAG fallback branch queries the MultiHop-RAG full corpus index.

## Final Result

| Method | EM | Acc | F1 | G-Sem | Tok |
|---|---:|---:|---:|---:|---:|
| TP-CRAG | 60.33 | 64.28 | 61.75 | 78.16 | 1.38 |

Raw metric values are stored in `results/reference_crag_topic_full_corpus/summary.json`.

CRAG branch counts:

| Branch | Count |
|---|---:|
| Correct | 1310 |
| Ambiguous | 947 |
| Incorrect | 299 |

## Files

```text
evaluate_multihop_crag_topic_rag.py      # MultiHop-RAG TP-CRAG evaluator
evaluate_crag_topic_rag.py               # HotpotQA TP-CRAG reference/adaptation source
build_topic_partition.py                 # BERTopic topic partition builder
build_crag_topic_index.py                # attach topic IDs to chunk index
build_multihop_doc_data.py               # normalize MultiHop-RAG corpus into doc_data.pkl
build_reference_index.py                 # build normalized numpy vector index
crag_topic_rag/                          # CRAG branch logic and topic router
naive_rag/                               # shared retrieval, IO, chunking, and LLM utilities
topic_partition_auto/                    # BERTopic auto partition artifacts
topic_index_auto/                        # topic-aware chunk metadata
results/reference_crag_topic_full_corpus/
  summary.json                           # final metrics
  predictions.csv                        # final predictions table
  records.jsonl.gz                       # compressed full per-example records
```

To inspect full records:

```bash
gunzip -c results/reference_crag_topic_full_corpus/records.jsonl.gz | head
```

## Reproduce

Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Prepare MultiHop-RAG under `datasets/multihop_rag/raw/`, then build document data and the full-corpus index:

```bash
python build_multihop_doc_data.py \
  --corpus-file datasets/multihop_rag/raw/corpus.jsonl \
  --out-file datasets/multihop_rag/intermediate/doc_data.pkl

python build_reference_index.py \
  --doc-data datasets/multihop_rag/intermediate/doc_data.pkl \
  --output-dir datasets/multihop_rag/indexes/full_corpus/vector_index \
  --device cuda:0
```

Build topic partitioning and topic-aware index:

```bash
python build_topic_partition.py \
  --doc-data datasets/multihop_rag/intermediate/doc_data.pkl \
  --output-dir topic_partition_auto \
  --nr-topics auto \
  --min-topic-size 5 \
  --hdbscan-min-samples 3 \
  --umap-n-components 5 \
  --backend cpu \
  --device cuda:0

python build_crag_topic_index.py \
  --base-index-dir datasets/multihop_rag/indexes/full_corpus/vector_index \
  --topic-map topic_partition_auto/topic_article_map.jsonl \
  --topic-info topic_partition_auto/topic_info_auto.csv \
  --output-dir topic_index_auto
```

Start Ollama with Gemma:

```bash
ollama serve
ollama pull gemma2:2b
```

Run evaluation:

```bash
python evaluate_multihop_crag_topic_rag.py \
  --data-file datasets/multihop_rag/raw/qa.jsonl \
  --index-dir topic_index_auto \
  --topic-info topic_partition_auto/topic_info_auto.csv \
  --output-dir results/reference_crag_topic_full_corpus \
  --device cuda:0 \
  --model gemma2:2b \
  --resume
```

## Metrics

- `EM`: normalized exact match.
- `Acc`: relaxed answer accuracy with containment matching for non-yes/no answers.
- `F1`: token-level answer F1.
- `G-Sem`: embedding semantic similarity between generated and gold answers.
- `Tok`: average generated answer token count after answer normalization.
