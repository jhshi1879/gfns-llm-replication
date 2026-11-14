# GFlowNet story-infilling replication

This repository copies the story-infilling experiment and its dependencies from
[GFNOrg/gfn-lm-tuning](https://github.com/GFNOrg/gfn-lm-tuning), along with
local training, evaluation, and analysis work. The upstream MIT license is
included in [LICENSE](LICENSE).

Start with [the experiment README](infill_subj_arithmetic/README.md). The
analysis notebook is [eda.ipynb](infill_subj_arithmetic/results/gfn-final/eda.ipynb);
start Jupyter from this repository, the experiment folder, or its results folder.
Install `matplotlib` and `PyYAML` in addition to the experiment requirements for
the notebook. Evaluation additionally requires `nltk` and `bert-score`.

## Training and fine-tuning

The recorded story-infilling run first trained the full GPT-2 Large model with
[`sft.py`](infill_subj_arithmetic/sft.py), using a `text` CSV made by joining
sentences 1–5 of each supplied ROCStories training story. Run from
`infill_subj_arithmetic/`, the recorded arguments were
`--model_name gpt2-large --use_auth_token False --output_dir output/stories_sft --learning_rate 0.0005 --batch_size 64 --gradient_accumulation_steps 32 --save_total_limit 1 --num_train_epochs 20`;
the default sequence length was 128. It completed 500 optimizer steps and
reported epoch 19.44. The second stage ran
[`main.py`](infill_subj_arithmetic/main.py) with `+experiment=infilling`,
the saved SFT model as `model.name`, `replay_buffer.size=25`,
`max_eval_len=25`, `eval_temp=0.9`, and `+save_final=true`. Its
[saved configuration](infill_subj_arithmetic/results/gfn-final/.hydra/config.yaml)
records seed 0, 1,000 training steps, batch size 64, gradient accumulation 16,
learning rate 0.0001, configured logZ learning rate 1.0, modified SubTB loss with
lambda 1.0, LoRA rank 64/alpha 16/dropout 0.1, generation length 5–25, and
reward-temperature schedule 1.1 to 0.85 over 100 steps. Despite the configured
logZ rate, the training loop applies the same 20-step warmup to 0.0001 to both
optimizer groups. The SFT data-preparation
launcher is local-only; the included `make_sft_data.py` exports only 900 stories
and does not reproduce the full-corpus preparation used in the recorded run.

Shell launchers (`*.sh`) are local-only and are not included in this repository.
Instructions in the experiment README that invoke them require separate copies.
The evaluation tests and some RunPod helper tests also depend on those launchers.

Model checkpoints, virtual environments, credentials, and downloaded ROCStories
datasets are not included. Obtain the datasets as described in the experiment
README and provide your own SFT model and GFN adapter for inference. Recorded
metrics and small result files are included. RunPod launchers deliberately keep
the original `/workspace/gfns-llm/infill_subj_arithmetic` path on the remote pod;
the local credential file, if used, is `.env` at this repository's root.
Paths in the two saved Story Cloze evaluation metadata sets are relative to the
repository root; the referenced datasets and model weights must be supplied
separately. Other saved logs and configurations retain their original pod paths
as historical records, not as paths required on the local machine.
