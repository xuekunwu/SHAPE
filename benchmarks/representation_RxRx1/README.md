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

[RxRx1](https://www.rxrx.ai/rxrx1) (`metadata.csv` and `rxrx1-images.zip`), HEPG2 by default (also HUVEC, RPE,
U2OS). `build_prep` cuts the nucleus-centred 6-channel crops and splits the wells into a FULL prep and its run prep,
under the names `runners/config.py` (`PREPS`) expects in `--data-root`; `prepare` then builds the tool inputs from
them and `assets` stages the pinned DINOv3 source and weights:

```
mkdir -p RXRX1 && cd RXRX1     # RxRx1 as rxrx.ai distributes it (rxrx1-images.zip is 49 GB)
curl -L -O https://storage.googleapis.com/rxrx/rxrx1/rxrx1-metadata.zip && unzip rxrx1-metadata.zip
curl -L -O https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip && cd ..
python -m benchmarks.representation_RxRx1.runners.build_prep prepare --metadata RXRX1/rxrx1/metadata.csv \
    --images RXRX1/rxrx1-images.zip --cell-type HEPG2 --datasets train,test --n-experiments 11 \
    --output PREPS/prep_hepg2_e11_v3
python -m benchmarks.representation_RxRx1.runners.build_prep subsample --prep PREPS/prep_hepg2_e11_v3 \
    --output PREPS/prep_hepg2_e11_v3_t10 --train-well-fraction 0.1
python benchmarks/representation_RxRx1/run_rxrx1.py prepare --cell-type HEPG2 --data-root PREPS --output-root DIR \
    --dinov3-repo DINOV3 --weights DINOV3_WEIGHTS      # the two DINOv3 inputs of benchmarks/full_CPJUMP1 `cli assets`
python benchmarks/representation_RxRx1/run_rxrx1.py assets --output-root DIR   # downloads: run where there is network
```

The `build_prep` options above (all 11 HEPG2 experiments of the train and test sets, positive-control wells, the
defaults otherwise) follow the prep names; the original task-116 preps were not kept, so they are not checked
against them.

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
