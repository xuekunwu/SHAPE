# representation_RxRx1

Single-cell representation learning on RxRx1 HEPG2 (Cell Painting-style 6-channel crops, 30 siRNA
positive controls). Seven arms each deliver one embedding of every cell: `shape` (SHAPE tuning the
self-supervised fine-tune of `Cell_State_Analyzer_Tool`, DINOv3 ViT-B/16 with a learned 6->3
channel adapter, through its agentic VRR), `dinov3` (the same tool at its defaults) and `frozen`
(the pretrained backbone, zero-shot), `react` (an LLM + the tool on the shared ReAct runner), `llm`
(an LLM writing its own feature code), and two external feature sets, `openphenom` and
`cellprofiler`. Scored by well mAP (same-siRNA retrieval) and a siRNA probe across experiments; no
label enters any arm's decision.

## Dataset

RxRx1 HEPG2 (and optionally HUVEC/RPE/U2OS). `prepare` builds the tool inputs and stages the pinned
DINOv3 assets from the task-116 preps:

```
python benchmarks/representation_RxRx1/run_rxrx1.py prepare --cell-type HEPG2 --data-root DIR --output-root DIR
python benchmarks/representation_RxRx1/run_rxrx1.py assets --output-root DIR   # run on a login node: it downloads
```

## GPU

Needed for `control`, `shape` and `score` (local DINOv3 fine-tuning/inference -- no remote service).

## Running the arms

```
python benchmarks/representation_RxRx1/run_rxrx1.py control --cell-type HEPG2 --output-root DIR
python benchmarks/representation_RxRx1/run_rxrx1.py external --cell-type HEPG2 --output-root DIR --arms openphenom,cellprofiler
python benchmarks/representation_RxRx1/run_rxrx1.py shape --run RUN --cell-type HEPG2 --output-root DIR
python benchmarks/representation_RxRx1/run_rxrx1.py llm --run RUN --cell-type HEPG2 --output-root DIR
python benchmarks/representation_RxRx1/run_rxrx1.py react --run RUN --cell-type HEPG2 --output-root DIR
python benchmarks/representation_RxRx1/run_rxrx1.py score --run RUN --cell-type HEPG2 --output-root DIR \
    --llm-run LLM_RUN --react-run REACT_RUN
```
