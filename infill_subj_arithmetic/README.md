# GFlowNet fine-tuning for story infilling, subjectivity classification and integer arithmetic

To install the dependencies required to run the code in this directory use `pip install -r requirements.txt`



### Story infilling

#### Validate the pod environment before full training

From your local terminal:

```bash
bash run_preflight.sh root@YOUR_POD_IP YOUR_SSH_PORT
```

This uploads only training/preflight code and configs, uses the separate
`.venv-validated-py311` environment, installs the pinned requirements, and
downloads ROCStories on the pod if absent. It tests one GPT-2 Large SFT update
with batch 1, model/tokenizer save and reload, then two short GFN steps with
batch 1 and adapter reload. Two GFN steps are needed because its first step
has zero learning rate during warmup. GFN uses one synthetic infilling example;
this is a compatibility check, not the held-out paper benchmark or a full-batch
GPU memory test.

On success it records finite metrics, checks changed GFN adapter weights,
exports all installed versions to `requirements.lock.txt`, and writes
`preflight-passed.json`. This version freeze is not a hash-verified lockfile.
Failures return a nonzero exit status and leave diagnostics without a pass
marker. Results are downloaded under local `results/preflight-download-*`.
Preflight is optional, not a launch requirement. The shared remote SFT/GFN
launcher reuses this environment or automatically creates it on a new pod.
It installs requirements when their installation snapshot is absent or changed,
checks package metadata and CUDA, and starts training without a smoke-test gate.
Keep the local terminal open, do not run alongside active training, and
stop the pod manually afterward: this preflight does not use RunPod credentials
or automatically shut down the pod.

SFT reporting integrations are disabled unless requested with `--log_with`
(for example, `--log_with wandb`); local training logs are still saved. The
legacy W&B dependency requires `pkg_resources`, supplied by pinned setuptools
80.9.0, including for the downstream GFlowNet script.

For the infilling task please download the ROC Stories dataset by filling the form at https://cs.rochester.edu/nlp/rocstories/. Once you receive the email, download  `ROCStories_winter2017 - ROCStories_winter2017.csv`, rename it to `stories.csv` and place it in the `data` directory. We also need to fine-tune a GPT-2 Large model on this dataset. First convert the dataset to the required format using `make_sft_data.py` and then run the `sft.py` with the appropriate `--output_dir` parameter. 

#### Launch from your local terminal onto RunPod

Before training, provision a nonzero persistent volume mounted at `/workspace`.
The launchers save code and outputs beneath that exact path; configuring a
volume at `/workspace-2` does not protect files written to `/workspace`.
The directory existing is not evidence that a persistent volume is mounted.
Verify the pod's volume size and mount path in RunPod and check `df -h / /workspace`
on the pod. If both paths use the container's `overlay` filesystem, the outputs
are on disposable container storage and will not survive a stop/restart.
Automatic shutdown does not back up outputs. CPU-only restart cannot recover
files that were saved only on the discarded container disk.

Use the host and port from RunPod's **SSH over exposed TCP** section, not its
SSH proxy endpoint (the latter does not support SCP). Configure your SSH key
in `~/.ssh/config` or load it into your SSH agent, then run:

```bash
bash run_sft_remote.sh root@YOUR_POD_IP YOUR_SSH_PORT
```

Before launching, set `RUNPOD_POD_ID` to that same pod's ID and
`RUNPOD_API_KEY` in your local environment or the repository-root `../.env`.
The pod ID can instead be passed on the command line with `--pod-id POD_ID`
(accepted by both remote launchers and `download_results.sh`); the flag
overrides any environment or `.env` value.
The shell launcher selects that file automatically; the Python helper parses
only these two variables without executing shell commands. Existing environment
values take precedence. Override `RUNPOD_ENV_FILE` for another file.
Create an API key in RunPod account
settings with permission to stop the pod; do not put it in scripts or commits.
The launcher securely forwards the two credentials over SSH stdin into a
temporary mode-0600 file on the pod. Only its path, not the key, is passed to
tmux. The training process reads and deletes the file immediately and registers
an `atexit` shutdown callback before importing training dependencies.
On successful completion or an unhandled Python exception, the callback uses
the RunPod SDK's `runpod.stop_pod(pod_id)`, not the previous REST endpoint.
The SDK lives in a separate `.venv-runpod-sdk` environment (`runpod>=2.0,<3`)
so installing it does not change the preflight-validated training packages.
Credentials are not written into code, training logs, or command arguments.

Once the training worker has started, disconnecting or shutting down your laptop
does not stop training or cancel its shutdown callback. There is no longer a
laptop-side shutdown trap. Setup failures before callback registration, SIGKILL,
native crashes, and pod power loss cannot invoke it. SDK/API failures can still
prevent shutdown; verify the pod's billing/status and stop it manually if needed.
Existing completed/running jobs are not retroactively given this callback.
Stopping does not delete the pod and may not eliminate storage charges.

By default, the script downloads the authorized winter-2017 CSV from
`https://goo.gl/0OYkPK` directly on the pod; no dataset is downloaded locally.
Override `STORY_DATA_URL` to use another authorized CSV URL. Local CSV files
are not automatically uploaded. If explicitly desired, supply one with:

```bash
STORY_DATA_PATH="/path/to/ROCStories_winter2017.csv" \
  bash run_sft_remote.sh root@YOUR_POD_IP YOUR_SSH_PORT
```

It uploads `run_sft.sh`, `sft.py`, and `requirements.txt` (plus the local CSV,
if supplied), along with the environment validator, to
`/workspace/gfns-llm/infill_subj_arithmetic`. It installs missing
tmux/curl/Python tools on root-access Debian/Ubuntu pods, verifies the existing
`.venv-validated-py311` environment (creating it if missing), checks CUDA,
prepares data, and starts the `stories-sft`
tmux session. GPU charges begin according to your pod's billing, and the launcher
starts the job using the epoch count in `run_sft.sh` (currently 2). It prints the SSH command to attach and inspect
training; detach with Ctrl-b, then d. It does not launch the subsequent GFlowNet
stage or copy its code. A Python version compatible with the pinned dependencies
and working `venv` support are required on the pod.

Automatic results downloads are disabled in `run_sft_remote.sh`
(also used by the remote GFN launcher). Checkpoints, final weights, metrics, and
logs remain on the pod. Completion monitoring and the shutdown attempt remain
enabled, including on failure. Keep the pod's persistent storage to retain outputs.
Training logs stream in the same local launcher terminal while it waits;
no second terminal is required. Remote training still runs inside tmux.
Download results separately from your Mac (CPU-only pod startup is sufficient):

```bash
bash download_results.sh root@YOUR_POD_IP YOUR_SSH_PORT
```

This copies the SFT output entries with SCP's per-file progress display,
excluding top-level `checkpoint-*` folders,
into a unique local `results/output-download-*` directory. Final model weights,
tokenizer, metrics, and logs are retained. Existing downloads are not removed. Ensure
sufficient disk space. It does not wait for training. It requires the same
`RUNPOD_POD_ID` and `RUNPOD_API_KEY` used by the remote launcher, from your local
environment or repository-root `../.env`. It requests pod shutdown after the
download attempt, including later failures and caught interrupt/termination
signals. Shutdown is best-effort; if the request fails, stop the pod manually
and verify its status in the RunPod console. A forced kill or loss of
connectivity may prevent cleanup.
Use `TRAINING_STAGE=gfn bash download_results.sh USER@HOST SSH_PORT` for GFN
outputs.

For a local CPU smoke test of the SFT save path, use a Python environment with
the pinned training dependencies:

```bash
python -m unittest -v test_sft_save
```

This creates a tiny random GPT-2 model and tokenizer locally, runs one update
through `sft.py`, checks finite training loss and changed weights, and reloads
the final model/tokenizer for inference. It also compares the final weights
with the saved checkpoint. It needs no GPU, model or dataset downloads, or
RunPod credentials, disables pod shutdown, and cleans up its temporary files.
It verifies saving and reloading, not storage survival across a pod restart.

After an internet interruption, reconnect to an existing run without launching
another job:

```bash
bash run_sft_remote.sh --monitor root@YOUR_POD_IP YOUR_SSH_PORT
```

Use the pod's current SSH address and port; they may change after a restart.
The monitor shows the last 50 log lines and follows new output until completion.
Press Ctrl-C to stop monitoring without stopping training. The legacy
`--download-only` option also monitors and does not transfer files, but prints
the entire log. Monitoring requires SSH access to the pod and a completion
marker generated by `run_sft.sh`; it does not require credentials or request
shutdown itself. Pod shutdown may disconnect monitoring before the worker writes
its final completion marker; verify status in RunPod in that case. If the pod
has stopped, it must be accessible again before its logs can be inspected.

To launch story SFT in a persistent tmux session on your GPU machine, prepare
the training environment and request dataset access using the official
ROCStories form linked above. Once you receive the direct CSV download link:

```bash
STORY_DATA_URL="https://your-authorized-csv-download-link" bash run_sft.sh
tmux attach -t stories-sft
```

The launcher uses GPT-2 Large, learning rate `0.0005`, batch size `64`,
gradient accumulation `32`, and `20` epochs. Only the latest intermediate
checkpoint is retained (`--save_total_limit 1`); after saving a new checkpoint,
Trainer deletes older ones. Saving temporarily needs space for both the old
and new checkpoints. The final model is also saved separately in the output
root. Existing checkpoints are not removed until a new checkpoint saves
successfully. Model checkpoints and timestamped
training logs are saved under `output/stories_sft`. Each log includes the exact
command, script and requirements snapshots, training output, completion status,
and, on success, final epoch, optimizer step, aggregate training loss, and runtime
metrics. Final metrics are also saved as `train_results.json`; the reported
training loss is an overall training average, not a held-out validation score or
an isolated final-epoch average. Detach with Ctrl-b, then d;
the session remains open after completion or failure so the exit status is visible.
The launcher downloads and validates the CSV, saves it as `data/stories.csv`,
and prepares all rows as `data/stories_sft.csv` before starting tmux. If
`data/stories.csv` already exists, simply run `bash run_sft.sh`; no download is
needed. An existing prepared CSV can also be used when no source CSV is available,
but its full-corpus coverage is not verified. The launcher uses the validated
environment's Python (or active Python if absent) and does not install dependencies.
It records a fresh installed-version freeze, not a preflight pass claim, in the
training output and log. `--prepare-only`
still uses the active Python so preflight can prepare data before validation.
Unlike `make_sft_data.py`,
which exports only 900 stories, this launcher prepares the full supplied corpus.
Direct pod launches do not enable automatic shutdown unless you explicitly set
`STOP_RUNPOD_ON_EXIT=1` and supply RunPod credentials (and have the SDK environment).
Preflight always disables this callback.

Command to run GFlowNet fine-tuning for story infilling:
```bash
python main.py +experiment=infilling model.name="<path of fine-tuned gpt2>"
```

To compute the metrics you can load the checkpoint stored during training with the `eval_infill.py` script.

```bash
python eval_infill.py --batch_size 1024 --temp 0.9  --max_eval_len 25 --load_checkpoint_path "<path_here>"
```

### Downstream GFlowNet infilling launcher

After successful story SFT, start/restart a GPU pod, update its SSH host/port
and `RUNPOD_POD_ID`, and run from your local terminal:

```bash
bash run_gfn_remote.sh root@YOUR_POD_IP YOUR_SSH_PORT
```

By default it uses the full SFT model already on that pod at
`/workspace/gfns-llm/infill_subj_arithmetic/output/stories_sft`. If using a new
pod, upload a previously downloaded SFT output directory explicitly:

```bash
SFT_MODEL_PATH="$PWD/results/YOUR_SFT_DOWNLOAD/stories_sft" \
  GFN_TRAIN_STEPS=10 bash run_gfn_remote.sh root@YOUR_POD_IP YOUR_SSH_PORT
```

The default is 1,000 optimizer steps (not epochs), batch 64, accumulation 16,
learning rate `0.0001`, LoRA rank 64, buffer capacity 25, and generation length
5–25. `GFN_TRAIN_STEPS=10` is a shorter run but still uses full-size training
batches and repository evaluation. Sampling evaluation temperature is 0.9;
the paper table lists 0.8 whereas Appendix C's prose lists 0.9.

The launcher reuses the SFT launcher's Python 3.11 installation, dataset
download, live log streaming, local `.env`, and automatic shutdown policy.
It uploads the GFN code and YAML configs and starts `stories-gfn` in tmux.
Download `output/stories_gfn` separately with `download_results.sh` as described
above. A pod-side supervisor requests shutdown when the training process exits,
including on an unhandled exception, native crash, or SIGKILL of that process.
Shutdown remains best-effort: a killed supervisor, pod failure, setup failure
before supervisor startup, or API failure can prevent it.
The remote output contains the final LoRA `final_adapter/`, periodic adapters,
replay buffers, final metrics, optimizer/logZ state, resolved Hydra configs, and
script snapshots. Loading the adapter still requires the original SFT base
model; it is not a standalone full model. Saved optimizer state does not add
an automatic training-resume interface.

Run directly on an already prepared pod with
`bash run_gfn.sh` (activate its Python environment first). Direct launches
require explicit RunPod credentials and `STOP_RUNPOD_ON_EXIT=1` for shutdown.
To monitor an existing GFN run without relaunching:

```bash
bash run_gfn_remote.sh --download-only root@YOUR_POD_IP YOUR_SSH_PORT
```

Close the completed `stories-sft` tmux session before launching GFN on the same
pod. No formal 100-story paper benchmark or GPT-4 judging is launched here;
the training script's built-in evaluations use its training-story queries.

#### Compare held-out infilling with the paper

The separate evaluator in `results/gfn-final/` runs both the general held-out
evaluation and a Table 2 comparison by default:

```bash
cd results/gfn-final
bash run_eval.sh \
  --bert-model-dir /path/to/local/deberta-xlarge-mnli --bert-num-layers 40
```

The pod must already have the SFT checkpoint, GFN adapter, and local BERTScore
model. The launcher reuses the default CSVs in `data/`, preferring the training
file `data/stories.csv` when present, and downloads missing defaults
from the supplied HTTPS links. Downloads are validated before being saved.
Explicit `--validation-csv` and `--training-csv` paths are used as supplied
and are not downloaded. `--check-only` also prepares missing default datasets;
`--help` does not. Install `nltk` and `bert-score` alongside
the project dependencies in the Python environment used by the launcher.
If present, `.venv-eval-py311` is selected before the validated training
environment. The launcher also detects `output/stories_gfn/final_adapter`
when the default adapter is absent, and `models/deberta-xlarge-mnli` at
layer 40 when no BERTScore model is specified. Explicit flags and environment
variables take precedence.
Use `bash run_eval.sh --help` for paths and other options. Set
`--general-stories` (or its existing alias `--stories`) and `--paper-stories`
independently; they default to 900 and 100. For example:

```bash
bash run_eval.sh --general-stories 900 --paper-stories 100
```

To launch from your local computer on an already prepared pod, use the pod's
current direct SSH endpoint (or an SSH config alias):

```bash
bash run_eval.sh --ssh-host USER@HOST --ssh-port SSH_PORT \
  --general-stories 900 --paper-stories 100
ssh -t -p SSH_PORT USER@HOST tmux attach -t stories-eval
```

SSH credentials must be configured locally. The evaluator, launcher, RunPod
helper and SDK environment must already be installed on the pod; this command
does not upload local changes. The evaluation runs in tmux, and paths passed as
evaluation flags refer to the pod. No pod ID, SSH address, or shutdown
credentials are hardcoded.

Automatic shutdown is enabled by default for evaluation runs, including
failures. Set `RUNPOD_API_KEY` locally, or in the repository-root `../.env`
using the existing training convention. The connected pod ID is detected
automatically; if `RUNPOD_POD_ID` is configured, it must match. Credentials
are sent over SSH through stdin and stored briefly in a mode-0600 file on the
pod. Use `--no-stop-pod-on-exit` to
disable shutdown. `--help` and `--check-only` never stop the pod. The worker
writes results and exit status before the shutdown request, but a killed
supervisor, launch/preflight failure, lost connectivity or RunPod API failure
can prevent shutdown; inspect `shutdown.log` and verify pod status and billing.
Results are not downloaded automatically, so keep persistent storage.

Both use the same seeded held-out sample; each takes its requested number of
stories from the start, and generation is shared for overlapping stories.
Each enabled evaluation needs at least two stories for confidence intervals.
Using a paper count other than 100 further changes the comparison protocol.
Set `--paper-eval false` for general only, or `--general-eval false` for paper
only. Both flags accept `true` or `false` and default to `true`; disabling both
is an error. The previous `--paper-comparison` option remains an alias for
`--paper-eval true`.
Generation runs on CUDA by default and completed chunks are cached.
Results stay on the pod; downloads are not automatic.

`paper_comparison/comparison.csv` lists our SFT and GFN results beside the
paper's reported numbers, and `confidence_intervals.csv` contains
story-bootstrap intervals. BLEU-4 and GLEU-4 are percentages; BERTScore F1
uses the local DeBERTa-compatible model, layer 40 and the bundled rescaling
baseline. Verify that the local weights are the intended
`microsoft/deberta-xlarge-mnli` checkpoint: configuration checks do not prove
weight identity. The scores use full stripped strings, not the legacy script's
character slicing. The held-out Story Cloze data, SFT training set, sampling,
and text preprocessing differ from the paper, so this is not an exact
reproduction. The paper's plus/minus notation is transcribed without assuming
it is equivalent to our bootstrap intervals. Prompting and GPT4Eval are shown
as unmeasured; the evaluator does not call an external judge.

### Subjectivity Classification

To run the GFlowNet fine-tuning experiment, run `Subj_obj_GFN_FT_E_step.ipynb` (for the M step see `Subj_obj_GFN_FT_M_step.ipynb`)

### Arithmetic with tool use

We include the datasets used for the arithmetic task and to generate the dataset we used `data_integer_arithmetic.ipynb`.


Command to run GFlowNet fine-tuning for arithmetic with calculator:
```bash
python main.py +experiment=arithmetic_varying 
```

To compute the metrics you can load the checkpoint stored during training with the `eval_arithmetic.py` script.

```bash
python prompting.py +experiment=arithmetic_varying test_only=True max_eval_len=22 dataset.test_path=data/arithmetic_with_tool/1digit_3_op+-_test.json use_tools=True limit_capability=2 reward.reward_config.prompt_data.num_points=0 data.num_test=10000 load_checkpoint_path="<path_here>"
```

Please refer to `configs/rationale_buffer.yaml` for the hyperparameters.
