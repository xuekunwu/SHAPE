"""
Canonical configuration for representation_RxRx1.

The settings every stage reads: the cell type, the task-116 preps of each cell type, the output root, the control
seeds, the round budget and the reasoner model. The benchmark scores seven arms on Cell_State_Analyzer_Tool and RxRx1
(SHAPE's agentic SSL VRR -- RepresentationAgenticVerifier through StepVerifier's wave mode -- the frozen and the
default-trained analyzer, ReAct, llm and two external arms) by well mAP (same-siRNA retrieval of well profiles). No
label enters a decision: the siRNA labels are in a private file that only score.py reads. The SHAPE arm bypasses
solve() (the verifier is not registered there): it calls the tool and StepVerifier.resolve_stage directly.

Data: the task-116 v3 preps of one RxRx1 cell type under data_root, made with the agentic_rxrx1
harness: the FULL prep (every cell, a label-free train / val / held-out well split) and its run prep (all cells
of 10 % of every plate's train wells + all val cells). Outputs: <output_root>/<cell type>/. The full-set v4 preps
(`prep_set="full_v4"`, built with build_prep.py) are registered in PREP_SETS; their outputs
go to <output_root>/<cell type>_full_v4/; the full-set v5 preps (`prep_set="full_v5"`, a well-level resplit of v4)
alike, to <output_root>/<cell type>_full_v5/.
GPU: control, shape and score.
"""

from dataclasses import dataclass
import os
from pathlib import Path

BENCH_ROOT = os.environ.get("SHAPE_BENCH_DATA_ROOT", "/scratch/groups/your-group/your-user/shape_benchmark")

# the task-116 v3 preps of each cell type under data_root: (FULL prep, run prep)
PREPS = {
    "HEPG2": ("prep_hepg2_e11_v3", "prep_hepg2_e11_v3_t10"),     # a cell type task 116 never used
    # task 116 tuned on these two: the CARD's mechanism text describes what it saw there, so a run on them is
    # not a clean transfer test
    "HUVEC": ("prep_e16_v3", "prep_e16_v3_t10"),
    "RPE": ("prep_rpe_e11_v3", "prep_rpe_e11_v3_t10"),
}

# the full-set v4 preps of each cell type under data_root (build_prep.py): every experiment of
# the train and test datasets, every well type, round(0.8 x n) experiments held out, val wells per plate targeting
# ~200 per cell type; the run prep holds ~5,000 training cells + every val cell
FULL_V4_PREPS = {
    "HEPG2": ("prep_hepg2_full_v4", "prep_hepg2_full_v4_t5k"),
    "HUVEC": ("prep_huvec_full_v4", "prep_huvec_full_v4_t5k"),
    "RPE": ("prep_rpe_full_v4", "prep_rpe_full_v4_t5k"),
    "U2OS": ("prep_u2os_full_v4", "prep_u2os_full_v4_t5k"),
}
# the full-set v5 preps (build_prep.py resplit --split wells of the full_v4 preps): the same cells
# and images.npy, a WELL-level split: per plate (every experiment, every well) 10 % of the wells seen (80 % of them
# train, 20 % val, >= 2 val wells per plate) and 90 % held out; no held-out experiment; the run prep holds every train
# cell and every val cell
FULL_V5_PREPS = {
    "HEPG2": ("prep_hepg2_full_v5", "prep_hepg2_full_v5_run"),
    "HUVEC": ("prep_huvec_full_v5", "prep_huvec_full_v5_run"),
    "RPE": ("prep_rpe_full_v5", "prep_rpe_full_v5_run"),
    "U2OS": ("prep_u2os_full_v5", "prep_u2os_full_v5_run"),
}
PREP_SETS = {"v3": PREPS, "full_v4": FULL_V4_PREPS, "full_v5": FULL_V5_PREPS}


@dataclass
class BenchmarkConfig:
    # === Data ===
    cell_type: str = "HEPG2"
    prep_set: str = "v3"                                  # PREP_SETS: v3 (the task-116 preps), full_v4 or full_v5
    data_root: str = f"{BENCH_ROOT}/RxRx/agentic_ssl"     # the task-116 preps
    # the SHAPE results layout: <results root>/representation_RxRx1/cache holds the reusable intermediates
    output_root: str = (f"{os.environ.get('SHAPE_RESULTS_ROOT', f'{BENCH_ROOT}/results')}"
                        "/representation_RxRx1/cache")
    split_key: str = "split"                              # the metadata key of the train / val split the tool reads

    # === Controls ===
    control_seeds: tuple = (0, 1, 2)                      # dinov3 (seed 0), dinov3_s<k>: plain DINOv3 fine-tune

    # === SHAPE arm ===
    max_rounds: int = 8                                   # rounds at most (each holds one call: concurrency 1)
    model: str = "gpt-5.6-luna"                           # the reasoner (Stanford AI Gateway, LLM_BASE_URL / LLM_API_KEY)

    # === Resource / runtime ===
    requires_gpu: bool = True

    def preps(self) -> tuple[Path, Path]:
        """(FULL prep, run prep) of the cell type in the prep set."""
        full, run = PREP_SETS[self.prep_set][self.cell_type]
        return Path(self.data_root) / full, Path(self.data_root) / run

    def root(self) -> Path:
        """<output_root>/<cell type> (v3; another prep set: <cell type>_<prep set>): prep/, private/, control/,
        external/, external_pilot/."""
        name = self.cell_type if self.prep_set == "v3" else f"{self.cell_type}_{self.prep_set}"
        return Path(self.output_root) / name


DEFAULT = BenchmarkConfig()
