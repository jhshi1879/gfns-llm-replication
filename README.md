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
