# ReAct arm contract (user decision 2026-10-11)

One ReAct design for every benchmark: **an LLM plus the tool**. One shared runner,
`benchmarks/_shared/react_arm.py`. Each benchmark contributes only a small tool adapter. Every existing ReAct arm
(segmentation_NeurIPS22-CellSeg, cropper_NeurIPS22-CellSeg, batch_correction_JUMP, representation_RxRx1) is replaced
by it, and full_CPJUMP1 gains a `react` arm built the same way. Old ReAct results are superseded and must be re-run.

## What the agent gets

The prompt holds only:
1. The task text, identical to the text SHAPE's agent gets for this task, including the task's own target definition
   (for example CellSeg `Prompt.txt ## Target`).
2. The input description: the same facts SHAPE is given, with nothing SHAPE computes.
   The input images: the same raw inputs SHAPE sees (for example CellSeg's raw image, the file the segmenter gets),
   attached to the first question of each session only, labelled `INPUT IMAGE: <name>`, at the size of the other
   images and inside the 16-image cap (vision models only).
3. The tool name and its parameter table: **name, type and default value only**. The parameter set is exactly the one
   SHAPE's verifier can set for this tool, including per-image and per-cell parameters where SHAPE has them.
4. One factual line: "Tuning the tool's parameters can improve the tool's performance."
5. One factual section "PYTHON": what the python action does, the workspace layout and its limits (below).
6. The budget, the response schema and the time left.

The prompt never holds:
- CARD text: descriptions, ranges, advice, "earlier runs" priors, readout definitions;
- admission or guard rules, thresholds or verdicts;
- SHAPE-rendered overlays or panels, or any SHAPE QC readout;
- hand-written tool documentation with effect directions;
- hints of any kind.

## What each observation returns

- The tool's **own** return value: its returned dict with arrays and in-memory objects dropped and paths made relative.
- Image files the **tool itself** wrote, for example the segmenter's overlay PNG and crop tiles. They are attached when
  the model is a vision model, at most 16 per round, at the same size SHAPE's agent gets. A TIFF is rendered as SHAPE's
  vision reads render it (`shape.utils.image_data.ImageData`: read by its axes tag, a multi-channel image as its merged
  RGB, one channel percentile-stretched to 8 bit); other files are read as they are.
- No score SHAPE computes, not even task-level feedback SHAPE gets (user decision 2026-10-12: batch_correction_JUMP's
  calibration scores are not in the observations). A labelled input the task gives SHAPE is a python input instead
  (batch: the calibration wells with their compound labels), so the agent computes its own signal from it.
- A call's values go to the tool as the runner resolved them. No SHAPE move check (range, metadata, a configuration
  run before) runs first (user decision 2026-10-12); the tool reports what it cannot use. Only a value the adapter
  cannot map to the tool's input is refused, for example an RxRx1 `init_from` that names no finished call.
- A failed tool call returns its error text. It is an observation, never a crash.

## Python (user decision 2026-10-11: the agent designs its own validation signal)

- Action `{"action": "python", "code": str}`: the runner runs the code in a fresh subprocess of the same Python
  environment (`sys.executable`), cwd = the unit's workspace `react/<unit>/workspace/`, with a scrubbed environment
  (no `LLM_*`, `OPENAI*`, `ANTHROPIC*` or any `*KEY*` / `*TOKEN*` / `*SECRET*` / `*PASSWORD*` / `*CREDENTIAL*`
  variable), at most `PY_TIMEOUT` (300) seconds and never past the deadline. The observation: exit code (or timed
  out), seconds, the files it wrote under its own `scratch/py_<n>/` and stdout + stderr cut to `PY_OUTPUT_CHARS`
  (8000) characters (head and tail).
- A python step is **not a tuning round**. At most `PY_PER_ROUND` (2) python steps between two rounds, counted per
  session (a used round resets the count, and after the last round 2 more are allowed before the finish); one more is
  refused with an error observation and uses its round, like any invalid request.
- What the code can read: the adapter's allow-list `python_inputs(unit) -> {workspace path: file}`, label-free
  inputs of the unit (`inputs/<name>`) and the outputs of its finished calls (`calls/<call id>/<name>`), materialised
  before each python step as symlinks (copies on Windows) with `manifest.json` (path, kind, bytes, shape where cheap).
  Ground truth and private labels are never in it: every adapter calls `assert_no_private(paths, private_roots)`,
  and a private path fails the arm.
- Writes go only to `scratch/`. The allow-listed files are hashed (sha256) when materialised and re-hashed after each
  step (source and workspace entry); a change, or a file written outside `scratch/`, is recorded as a violation in
  the observation and the trace. The prompt says the code may compute any label-free quality signal; it names none.
- Allow-lists: segmentation, the raw image and each call's mask and overlay (never `labels/`); crop, the raw image,
  the given mask and each call's crops and metadata json; batch, the uncorrected features, the label-free well
  metadata (`Metadata_Source`, `_Batch`, `_Plate`, `_Well`, `_PlateType`), the calibration wells with their compound
  labels (`inputs/calibration_labels.csv`: the rows and columns of SHAPE's calibration view, the calibration compounds
  and every control well, no evaluation compound) and each call's corrected array; RxRx1, each call's h5ad and the label-free cell
  metadata (cell_id, plate, experiment, well, site, split; no siRNA).

## Budget, identical to SHAPE

- **Tuning rounds**: at most `shape.agent.verification.base.BUDGET` (8) rounds per unit. Import the constant; never
  copy it.
- The default / trial-0 call counts exactly as SHAPE counts it for that tool:
  - segmentation and crop: the default call is round 1;
  - representation and batch: the input / trial 0 is not a round.
- **Calls per round**: at most SHAPE's per-round cap for this tool on this machine. Call the same verifier
  `concurrency()` / cap function SHAPE uses, so the two caps cannot drift.
- A unit is one image for image tools (each image has its own ReAct session) and the dataset for dataset tools.
  Sessions run concurrently up to the arm's workers, like SHAPE's per-image VRR.
- An invalid or over-budget request returns an error observation and still uses its round.

## Delivery

- The agent names the call to deliver with `finish`.
- If a session ends without a valid finish, it delivers that unit's default call and records
  `default_session_unfinished`.
- The harness never chooses among the calls by any score.

## Models and settings

The same model, image size and token caps as SHAPE's agent in that benchmark (gpt-5.6-luna via the Stanford Gateway).
Basic LLM failures are fatal: no engine, authentication or permission errors, model not found, connection failures.
Every other failed call is recorded and the run goes on. There is no rule fallback.

## Records

Per unit:
- `react/<unit>/TRACE.jsonl`: every prompt digest, reply, action and observation; a python step is a line of kind
  `python` with its code, exit, timed_out, truncated output, seconds, files and violations
  (`react/<unit>/python/py_<n>.py` holds its code);
- `CALLS.json`;
- `SELECTED.json`.

Per arm: tokens, rounds, calls, python steps, python violations and seconds. These are scored and reported by the
benchmark's existing scorer with its existing evaluation settings.

## Shared runner interface

```python
@dataclass
class Param:  name: str; type: str; default: Any

class ToolAdapter(Protocol):
    tool_name: str
    unit_kind: Literal["image", "dataset"]
    def parameters(self) -> list[Param]                 # SHAPE's lever set, defaults only
    def units(self) -> list[str]
    def round_cap(self) -> int                          # SHAPE's per-round cap for this tool
    def default_counts_as_round(self) -> bool           # as SHAPE counts it
    def call(self, unit: str, params: dict, call_id: str) -> dict   # {"output": tool dict, "images": [paths], "error": str|None}
    def deliver(self, unit: str, call_id: str) -> None
    trial0_label: str                                   # optional: the default round's note in the history
                                                        # (default "the tool at its defaults, run by the system";
                                                        # e.g. batch: the uncorrected input, RxRx1: the zero-shot input)
    def python_inputs(self, unit: str) -> dict[str, path]   # optional: the python allow-list (label-free; calls
                                                        # assert_no_private); keys inputs/<name>, calls/<id>/<name>

def run_react(adapter, *, task_text, input_text, engine, out_dir, workers, deadline,
              budget=BUDGET,
              unit_input: Callable[[str], str] | None = None,         # the unit's own input facts (text)
              unit_images: Callable[[str], list[str]] | None = None,  # the unit's raw input image paths
              image_px=IMAGE_PX) -> dict                # per-unit SELECTED + arm totals
```
