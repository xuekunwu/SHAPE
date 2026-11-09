---
tool: Cell_State_Analyzer_Tool
category: bioimage_core
stage: cell_level
depends_on: [Single_Cell_Cropper_Tool]
tunable: true
metadata_consumes: [group, plate, experiment, batch, well, site, technical_group]
metadata_none: batch = the image name; the augment_color route; uncorrected states only
metadata_partial: batch = the plate; the field_aware floor eligible; uncorrected states only
metadata_full: as partial (2 or more replicated non-control groups), plus the group silhouette; corrected states re-clustered after a Harmony correction
---
# Tuning card: Cell_State_Analyzer_Tool

**When to use:** self-supervised representation + clustering of single- or multi-channel
cell/organoid crops. The tool inspects every crop, validates a consistent channel count, and
selects a deterministic C-to-3 adapter initialization automatically. **Requires** crops from
Single_Cell_Cropper_Tool. Input: per-cell crops. Output: embeddings, UMAP, clusters.

Self-supervised representation of single cells (embeddings + clustering). The output is a
feature space, NOT an image -- so there is no mask overlay to judge visually. Two loops tune
it. The linear loop (`RepresentationVerifier`) reads "Tunable parameters": the deterministic
representation QC (effective rank, PC1-vs-area correlation, batch/technical-group silhouette,
NN consistency) is the evidence its mid-loop recommend reasons from, `field_aware` /
`augment_color` are its knobs during a run, and it sets `adaptation_depth` once, up front. The
agentic loop (`RepresentationAgenticVerifier`, "Agentic VRR") tunes the training itself per
input, call by call, over the settings in "Levers" -- among them the depth (`trainable_pct`),
the length (`max_epochs`) and the learning rate -- on a label-free val split.

## Tunable parameters

### adaptation_depth  (values: zero_shot | frozen_head | trained; no-engine floor: `trained`)
How much of the backbone adapts. The linear loop sets it once, up front, from the evidence
below. The agentic loop does not offer it as a lever: its trial 0 is `zero_shot`, and its calls
set the depth with `trainable_pct` ("Levers"; 100 or null train the same set as `trained`).
Deliberately NOT decided by a threshold: the obvious rule ("few crops per class -> freeze")
thresholds the wrong quantity, since crops from one source image are not independent samples.
The evidence is produced pre-flight at NO extra cost -- `baseline_embedding` is a zero-shot
forward pass, itself one of the three candidate actions, and `separability_baseline` measures
how good the PRETRAINED features already are on this input, which is what "domain gap" means
operationally. Decide from that.
- **`zero_shot`** -- no training; the pretrained backbone CLS. Right when the baseline
  embedding is already separable enough that fine-tuning only overfits.
- **`frozen_head`** -- train the projector only.
- **`trained`** -- full fine-tune (tool default) when N supports it and the domain gap is wide.
- Effect to expect: it sets the ceiling on embedding quality; it does not fine-tune within one.
- Every depth is admissible for every supported channel count because the unified C-to-3
  adapter has a deterministic initialization (1-channel replicate, 3-channel identity,
  V6.3 biological 6-channel mapping, balanced mapping otherwise).

### field_aware  (default false; values: on | off)
Mask same-source-field crops out of each other's contrastive negatives (label-free -- uses
the per-crop SOURCE-IMAGE id, never the cell-line label).
- **Switch to `on`** when the embedding is BATCH-DRIVEN (`silhouette_by_batch` exceeds
  `silhouette_by_group`) AND the source id is RELIABLE: real well/plate metadata AND
  multiple cells per source field. A filename-guessed source id with mostly-singleton
  fields cannot support field-aware negatives -- masking a field of ~1 crop is close to a
  no-op, so do not spend a retune on it (use `augment_color` instead, below).
- Effect to expect: switching on lowers `silhouette_by_batch`.

### augment_color  (default: on for multi-channel crops, off for single-channel at the up-front recommend; values: on | off)
Per-channel intensity jitter during contrastive training. It is defined on CxHxW tensors
and therefore has identical semantics for single- and multi-channel fluorescence crops.
- On single-channel crops, intensity jitter is still a valid per-image decorrelator.
  **Switch to `on`** (even on single-channel data) when the embedding is BATCH-DRIVEN and
  the source id is NOT reliable (see `field_aware` above) -- it breaks a confound where the
  embedding tracks per-image intensity/exposure rather than cell state.
- Effect to expect: switching on lowers `silhouette_by_batch`.

### plate_mean_weight / plate_adversarial_weight  (defaults: 0.0; protocol controls)
Optional V6.3 nuisance constraints. They default to zero so comparison groups receive the same
trainer unless the experiment explicitly varies these losses; the agentic loop does not offer them
as levers, so its calls keep them at 0. Even when requested, both are forced to zero unless every crop has real plate and
site/technical-group metadata, there are at least two plates, and every plate has at least two
cells from two sites. Biological `group` labels are never used for eligibility or checkpoint
selection. The effective weights and disable reason are recorded in `plate_nuisance_route`.

### learning_rate  (default 3e-5; safe range [1e-5, 1e-4])
AdamW step for fine-tuning the pretrained DINOv3 backbone. Calibrated label-free against
TRAINING DYNAMICS, not cell-line labels -- the target is the largest lr that still trains
stably.
- **lr too high** -- loss DIVERGES or OSCILLATES (rising/jittery loss curve) and/or
  `effective_rank` crashes (collapse). Recommend: lr **down**.
- **lr too low** -- loss barely moves within the budget (under-fit), `effective_rank` and
  augmentation/NN consistency stay low. Recommend: lr **up**.
- Effect to expect: a stable, monotonically-decreasing loss to a plateau; `effective_rank`
  stays high; alignment (augmentation consistency) rises.

Training length in the linear loop is a deterministic rule
(`resolve_training_config`: `round(8 + 6*sqrt(128/N))` clamped to [8,20], then
`AdaptiveEarlyStopper` plateau-stops within it). The stopping signal is the training loss
trajectory, which is observable WITHIN the run, so an early-stopping rule captures it by
construction. The real kwarg is `max_epochs` (reachable via the `representation` stage
`param_map`), a lever of the agentic loop ("Levers"); the former `### epoch_budget`
heading declared a lever name that no `execute()` kwarg ever had, so a recommendation
naming it validated and then silently did nothing.

## Levers

Each lever below has one line `Family | Type | Range | Default`. Family is the lever's group: `objective` (what
the contrastive training pulls together and pushes apart), `depth` (which part of the backbone trains, and from
which state), `optimizer` (the AdamW step, its schedule and the training length) and `input` (how a training view
is made from a crop). Every other kwarg of the tool stays at its default.

### view_pairing
Family: objective | Type: choice | Range: same_crop, same_site, same_well | Default: same_crop
The second view of each training crop: a second augmentation of the crop itself (same_crop), or an augmentation
of a random other training crop of the same plate-qualified site (same_site) or well (same_well), the crop itself
when it is alone there. same_site and same_well train the cells of one site or well toward each other; the val
proxy measures that kind of within-well likeness on val wells, which never train. same_site and same_well need
that metadata on every crop. The pairing changes the level of the training loss and of the val eval loss, and the tool's own epoch
budget (max_epochs null) and its early stop read the training loss.

### contrastive_mask
Family: objective | Type: choice | Range: off, same_site, same_well | Default: off
Drops from each anchor's in-batch negatives the crops whose second view comes from the anchor's own site
(same_site) or well (same_well); the positive pair always stays, so cells sharing that unit are not pushed
apart. How many negatives it removes depends on how many crops of one unit share a batch: very few with random
batches over many wells, a large share with per_plate batches when a plate holds only a few training wells.
same_site and same_well need that metadata on every crop.

### batch_sampling
Family: objective | Type: choice | Range: random, per_plate, per_experiment | Default: random
Draws each training batch from all training crops (random), or from the crops of a single plate (per_plate) or
experiment (per_experiment), in chunks of at most the batch size, so the in-batch negatives share that plate's or
experiment's technical effects. When a plate holds only a few training wells, a per_plate batch contains crops
of only those wells. per_plate needs plate and per_experiment needs experiment on every crop.

### trainable_pct
Family: depth | Type: integer or null | Range: [0, 100] | Default: null
The share of the backbone's transformer blocks that trains, counted from the last block and resolved to a whole
number of blocks (logged); at 100 the patch embedding and tokens train too; 0 trains only the adapter and
projector, which always train; null is today's full fine-tune, the same trainable set as 100. The adapter sits
before the backbone, so every call back-propagates through all blocks at any value: more trainable blocks add
their parameter gradients and optimizer moments (about 1 GB at full depth), not activation memory, and save
little backward time. The automatic batch estimate does not depend on this value; an out-of-memory error halves
the batch and retries, so a deep call can still run with a smaller batch.

### init_from
Family: depth | Type: choice | Range: pretrained, parent | Default: pretrained
pretrained starts the call from the pretrained backbone with a fresh adapter and projector; parent resumes the
parent call's full training state at its best val-proxy evaluation (weights, optimizer moments, gradient scaler,
random streams and counters) and continues under this call's configuration, with empty optimizer moments for
newly trainable parameters and no warmup unless warmup_fraction is set. A move from a finished call resumes
that call (parent) unless the move sets init_from: pretrained must be set to restart from the pretrained
backbone with a fresh adapter and projector, keeping the parent's other levers. A move from input starts
pretrained. A resumed move with no other change continues training the parent's state.

### learning_rate
Family: optimizer | Type: number | Range: [1e-6, 1e-4] | Default: 3e-5
The AdamW step of the adapter and the unfrozen backbone blocks, and of the projection head unless
head_learning_rate is set; reached after the warmup (warmup_fraction), then constant.

### max_epochs
Family: optimizer | Type: integer or null | Range: [1, 60] | Default: null
null lets the tool set its own budget from the number of training crops, round(8 + 6 sqrt(128 / N)) epochs
within [8, 20], extend it in steps of x1.2 toward min(60, 4 x budget) while the training loss still improves,
and end the call earlier on a training-loss plateau; an integer caps the call at exactly that many epochs. With
a val split the call keeps its val evaluation with the highest proxy wherever the stop falls. A change of
pairing, mask or crop scale shifts the training-loss level that the null budget's extension and stop read.

### head_learning_rate
Family: optimizer | Type: number or null | Range: [1e-6, 1e-2] | Default: null
The AdamW step of the projection head, separate from learning_rate, which keeps setting the adapter and the backbone; null uses learning_rate for the
head as well.

### weight_decay
Family: optimizer | Type: number | Range: [0, 0.5] | Default: 1e-4
AdamW's decoupled weight decay of every trainable parameter: each step shrinks every weight by its group's
learning rate times weight_decay.

### clip_grad_norm
Family: optimizer | Type: number | Range: [0, 100] | Default: 0
Clips the global gradient norm of all trainable parameters (adapter, unfrozen blocks, projector) to this value before each optimizer step; 0 turns clipping off. The call logs its gradient norms;
they are not among the readouts shown here.

### warmup_fraction
Family: optimizer | Type: number or null | Range: [0, 0.5] | Default: null
null keeps the tool's rule: a linear warmup over 5 % of the call's initial step budget on a fresh call and none
on a resumed call. A number warms the learning rate up linearly over that fraction of this call's own steps, on
fresh and resumed calls alike.

### koleo_weight
Family: objective | Type: number | Range: [0, 1] | Default: 0
Adds koleo_weight times a KoLeo spreading term on the backbone CLS tokens, the embedding the tool exports:
within a batch, for each crop of a view, the negative log distance to its nearest other crop after L2
normalisation, averaged, and summed over the two views. It pushes the crops of a batch apart on the unit sphere;
0 turns it off. Its size depends on the batch size. It acts on the embedding the alarms are computed on, so
effective_rank can move with it whatever else changes.

### intensity_norm
Family: input | Type: choice | Range: off, per_plate | Default: off
off reads every channel of a crop as stored. per_plate rescales each channel of every crop, in training and in
the embedding, so that its plate's channel mean and spread match those of all training crops. The statistics
come from the training crops only and are stored with the model: a delivered model embeds crops of a training
plate with that plate's training statistics, and crops of a plate absent from training with that plate's own
statistics. Plate metadata is needed on every crop.

### augmentation
Family: input | Type: choice | Range: strong, light | Default: strong
How each training view is made from a crop (in [0, 1], after intensity_norm, before the channel adapter); the
embedding transform is the same under both. strong, this tool's pipeline (stronger geometry and blur): a random resized crop to 224 px with an
area fraction from crop_scale_min to 1, a horizontal flip (p 0.5), a vertical flip (p 0.2), a rotation by up to
60 degrees, a random per-channel intensity scale between 0.8 and 1.2 (p 0.8), a Gaussian blur (p 0.8) and
random erasing (p 0.3). light, a lighter pipeline at 224 px (lighter geometry and blur, a stronger
per-channel intensity jitter): the same kind of resized crop, a
rotation by a multiple of 90 degrees and a mirror (p 0.5), shared by all channels of the crop; then, separately
per channel, a brightness, contrast and gamma jitter, a light Gaussian blur (p 0.2) and light additive noise; no
erasing and no free rotation.

### crop_scale_min
Family: input | Type: number | Range: [0.05, 1.0] | Default: 0.3
The smallest area fraction of a crop that the random resized crop keeps for each training view, under either
augmentation, before the view is resized to 224 px; higher values give views that cover more of the cell. The
embedding transform is unchanged.

### drop_path
Family: depth | Type: number | Range: [0, 0.5] | Default: 0
Stochastic depth of the backbone during training. Block i of n has the rate r = drop_path x i / (n - 1); in a
batch of b crops, its attention branch and its MLP branch each skip a random subset of b - max(int(b (1 - r)), 1)
crops, and the kept crops' branch outputs are scaled up by b over their number. So at any drop_path above 0
every block after the first drops at least one crop per batch, and the dropped share, rounded up to whole crops,
exceeds r at small batches and changes with the batch size (the automatic batch, an out-of-memory halving). It is off
when crops are embedded.

## Agentic VRR (RepresentationAgenticVerifier, StepVerifier wave mode)

Rules measure and guard; the agent decides. The agentic loop tunes this tool's self-supervised fine-tune per
input: the agent decides which configurations to run, from which parent, how many, when to stop and which call
to deliver under the guard. It is the loop every agentically tuned tool shares, `base.AgenticVerifier`
(`shape/agent/verification/base.py`: the rounds, the validation of the CARD levers, the repair question, the
plateau, the final choice and execution-error accounting); its subclass `RepresentationAgenticVerifier`
(`shape/agent/verification/verifiers/representation.py`, next to the linear `RepresentationVerifier`) adds the
analyzer's part, listed here. It is this tool's verifier on the `solve()` path (`stage_map.py`'s
`verifier_by_tool` entry); the RxRx1 benchmark (`benchmarks/representation_RxRx1/`) also runs it. The round
prompts show only the "Levers" section of this CARD; this section is for readers. No label is read anywhere in
the loop.

The shared admission gate permits training only on RUN. Search defaults to `legacy`; set
`SHAPE_SEARCH_STRATEGY_REPRESENTATION=four_stage` for cover, contrast, narrow and confirm.
Candidates remain multi-epoch runs; resumed checkpoints and actual training steps are comparison
context, not interchangeable calls. Confirmation uses existing checks, not independent holdout data.
Failed questions never trigger a deterministic training chain; nonfatal failures are recorded and
the frozen input remains available, while missing required engines and fatal provider errors propagate.

A caller passes a `RepresentationTask` in the stage context (`upstream_ctx["representation_task"]`): the cores
and GPUs, a deadline, a one-line brief and the split -- `split_key`, the metadata key holding "train" / "val"
per crop, or without it the val wells the tool assigns per plate at `val_fraction` (`quant.assign_val_wells`).
Val crops never train; every readout below is computed on them.

0. **Trial 0 is the frozen backbone**: the tool at `adaptation_depth` `zero_shot` on all crops, with the task's
   split (outputs under `ssl_calls/input`). Its val embedding is the reference every call is compared with.
1. **Measurability gate** (code, no question): the crops' metadata tier (`quant.tier_of`) is replicate, every
   plate has at least `MIN_VAL_WELLS` (2) val wells and trial 0's proxy has at least one query site. Otherwise
   trial 0 is delivered (satisfied, `not_measurable`) and no call runs.
2. **Rounds.** Each round the agent returns a plan of up to `cap` moves (1 per round while the concurrency is 1:
   the tool runs every call on `cuda:0` and seeds process-global random streams). A move names a parent --
   `input`, the tool at every lever's Default, or any finished call -- and one or several lever changes
   relative to the parent's configuration, with a diagnosis and the readouts it expects to move. Round 1 is the
   same question: the agent judges from trial 0 whether this input needs tuning at all, and an empty round-1
   plan, once confirmed (step 4), delivers trial 0 (satisfied, `no_tuning_needed`); an empty plan in a later round,
once confirmed, ends the search. The
   shared layer validates each move (the CARD Type and Range; no repeat of a configuration already run, failed
   or in this round) and asks one repair question for the rejected ones; a lever value that reads a metadata
   column needs it on every crop (`quant.identities`), and `init_from` parent from `input` is rejected. A move
   from a finished call starts at that call's full configuration and resumes that call (`init_from` parent)
   unless the move sets `init_from`; `init_from` pretrained restarts from the pretrained backbone with a fresh
   adapter and projector and keeps the parent's other levers (amendment A7). `init_from` parent passes the
   parent's `resume_best` (its best val-proxy point) as `init_checkpoint`, and a resumed move with no other
   change continues training the parent's state. Each arm's tool-call ledger records every call as
   `resumed: <parent>` or `fresh`. Each call writes under `ssl_calls/<hash of its configuration>` and trains
   for at most `max_minutes`, the minutes left before the deadline minus a reserve (`RESERVE_MINUTES`, 15).
   No narrowing and no forced coverage of the Families (amendment A12: `narrowing` off); the Families only label
   the levers. The round prompt states the goal (`goal_text`): the largest improvement of the val proxy over
   trial 0 that the call budget can reach, the levers being the means. Every lever change of a move carries a
   `basis` (`BasedPlan`): an observation from this run's readouts (the call and the readout) or a testable
   hypothesis; a change without one is rejected through the repair question. The prompts list no levers that no
   call has changed and no reminder to review them (amendment A21): which levers to change is the agent's, from
   the evidence. With an expert start (`expert_config`, default off; one entry of {lever: value} per round;
   amendments A18, A19, A21), each of its rounds' prompts states it in one block: a move of expert round k changes
   only the expert levers of rounds 1..k, to their expert values (init_from aside), and its configuration holds
   all of them; a move that does not is rejected through the repair question (`_expert_problem`). The RxRx1
   SHAPE arm uses one round: view_pairing same_well, contrastive_mask same_well, batch_sampling per_plate,
   intensity_norm per_plate, augmentation light.
3. **Readouts.** Per call, on the val crops and on the comparison draws: the proxy `sibling_site_map` with its
   chance level; its paired comparison with trial 0 and with the parent (delta = the mean per-site difference,
   SE from a bootstrap over val wells, the number of shared query sites, beyond or within 2 SE; `quant.beats`);
   each expected movement against the observed one; the ALARMS `effective_rank`, `dead_dim_frac` and
   `image_leakage`, labelled as alarms, shown next to trial 0's values and read by no rule; the training facts
   (epochs trained, best epoch, what picked the best point, batch size, trainable blocks, seconds). The prompts
   say what each readout measures and how far it can be trusted. Everything else a call records -- the proxy on
   the selection draws, the val and training losses, gradient norms, the other health statistics, the val
   history, the clusters, the resume paths -- is logged only.
4. **Stops.** The search ends when the agent returns an empty plan -- with rounds left it is asked once more
   (the shared stop check, `confirm_stop`, `stop_check_text`: the rounds that remain, the levers no call has
   changed, the best admissible call so far) and may continue or confirm the stop; while the last finished call
   still improved on its parent (paired proxy delta > 0, any size), that question instead refuses the stop,
   states the delta +/- 2 SE and asks for moves (A12, logged as `stop_refused`, recorded in `stop_refusals`);
   an empty answer to it does not stop either: the verifier runs one continuation of that call as the round's
   call (`continuation_<n>`: no lever change, init_from parent; logged as `forced_continuation`, recorded in
   `forced_continuations`), counted in the budget like any call -- the round budget is spent
   (`max_rounds`, 8 rounds after trial 0, shown as a limit), no valid move remains, or the deadline leaves no
   time for a call (no minutes left, or fewer than the shortest non-training time of a finished call). The
   final question still runs. A plateau does not stop the search and forces no move (amendment A20, which
   withdraws A13's exploration round): whether to continue from a call, go back to an earlier one, change
   direction or stop is the agent's; the round budget is the backstop. Without a reasoner (the deterministic
   chain, a smoke only) the shared plateau backstop holds (3 rounds without a call that beats the best trial
   so far by more than 2 SE on the comparison draws; trial 0 is the first best).
5. **Final choice and guard.** The agent picks one admissible call, or null to keep trial 0. A call is
   admissible when it finished, wrote an h5ad with a finite, non-constant embedding, and its proxy beats trial
   0's by more than 2 SE on the comparison draws (`quant.beats`). An inadmissible pick gets the reason and one
   retry. An empty plan may name the call to deliver (`deliver`; after a stop check, the confirming
   answer's): an admissible one is delivered without the
   final question (`agent_choice`); otherwise, or with none named, the final question runs. When no call is delivered, trial 0 is, and the case is recorded: `not_measurable`,
   `no_tuning_needed`, `agent_kept_input` (a null choice), `final_choice_inadmissible`, `no_admissible_call`,
   `every_call_failed` or `no_call_ran`. The delivered call's (or trial 0's) h5ad and `resume_best` are linked
   to the canonical `cell_state_analysis/` paths.
6. **Without an engine** the tool at its defaults runs, then a fixed chain of configurations in the verifier
   module (`_FALLBACK_CHAIN`); with an engine required (`SHAPE_REQUIRE_AGENTIC=1`) it never runs.

**Evaluation facts.** With a split the tool evaluates the val wells every `eval_every_epochs` epochs and after
the last one (`val_history`), and keeps the evaluation with the highest val proxy as the call's best point
(`best_by` = `val_proxy`; without a split, the epoch with the lowest training loss, `train_loss`); it saves the
full training state there and at the end (`resume_best`, `resume_end`). A zero_shot call with a split
evaluates the val wells once, as `val_history` entry epoch 0 (the proxy only, no eval loss). The proxy is sibling-site retrieval
(`quant.sibling_site_map`): each val site's profile is the mean of 2 of its L2-normalised cells, drawn at
random, L2-normalised; each site ranks the other val sites of its plate by cosine, the positives being the other
site(s) of its own well; the value is the average precision over the query sites, averaged over 20 seeded draws
of the cells. It measures how much more alike the cells of one well are than cells of other wells on the same
plate; chance is the average precision of a random ranking of the same queries, set by how many val sites share
a plate and how many sites each well has. The tool picks the best point on the selection draws
(`quant.SELECTION_SEED`); the verifier recomputes the proxy from each call's h5ad, on the val rows in the tool's
validation order, and compares calls on separate comparison draws (`quant.FRESH_SEED`), so the draws that
picked a call's best point never score it. The val wells are the same for both sets of draws.

## Notes
- This stage's mid-loop recommend IS reasoned by an LLM (`vision_recommend_representation`),
  even though embeddings have no mask overlay to show it -- the evidence is the label-free QC
  metrics plus, when available, a sampled-crop montage and the up-front crop-vision profile.
  The deterministic gates below are the safety floor + no-engine fallback, not a hard-coded
  terminal script (task 021 retired the earlier hard-coded `batch_driven` script).
- **Linear-loop calibration:** in the linear loop, `learning_rate` / training length are
  confirmed by a bounded, few-round VLRR loop on a SMALL, data-derived subset (grown until the label-free health
  signal is stable and source coverage is adequate -- not a hardcoded size) BEFORE the full
  run. A round that worsens health is reverted and logged. Mid-run faults below still apply
  to the full run unchanged. The agentic loop runs no such calibration: it changes them call
  by call ("Agentic VRR").
- EMBEDDING COLLAPSE (NaN / effective rank below min) is a mandatory UPSTREAM escalation
  (bypasses the LLM) -- the fault is input/preprocessing, not tunable here.
- BATCH-DRIVEN structure (`silhouette_by_batch` exceeds `silhouette_by_group`) is REASONED,
  not an upstream escalation: the deterministic floor proposes `field_aware` when the
  per-crop source id is reliable, or `augment_color` when it is not (see their entries
  above); the LLM may confirm or tighten that floor from the evidence, or concede to a
  data-side fault (e.g. genuinely multi-channel) once the floor lever is exhausted.
- Clustering knobs (`leiden_resolution`, `n_neighbors`) are governed by the morphological-
  state verifier, not this stage.
