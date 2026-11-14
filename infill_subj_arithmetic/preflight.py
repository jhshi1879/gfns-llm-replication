import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT = Path(__file__).resolve().parent
OUTPUT = Path(sys.argv[1]).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)


def check_weights(model):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Non-finite model weights: {name}")


def run_sft():
    if not torch.cuda.is_available():
        raise RuntimeError("Preflight requires an NVIDIA CUDA GPU.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    with (PROJECT / "data/stories.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [next(reader) for _ in range(4)]
    with (OUTPUT / "stories.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text"])
        for row in rows:
            writer.writerow([" ".join(row[f"sentence{i}"] for i in range(1, 6))])
    subprocess.run([
        sys.executable, str(PROJECT / "sft.py"),
        "--model_name", "gpt2-large", "--use_auth_token", "False",
        "--dataset_path", str(OUTPUT / "stories.csv"),
        "--output_dir", str(OUTPUT / "sft"),
        "--batch_size", "1", "--gradient_accumulation_steps", "1",
        "--seq_length", "128", "--max_steps", "1", "--save_steps", "1",
        "--learning_rate", "0.0005",
    ], cwd=PROJECT, check=True)
    metrics = json.loads((OUTPUT / "sft/train_results.json").read_text())
    if metrics["global_step"] != 1 or not math.isfinite(metrics["train_loss"]):
        raise RuntimeError("SFT did not complete one finite-loss update.")
    model = AutoModelForCausalLM.from_pretrained(OUTPUT / "sft")
    check_weights(model)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT / "sft")
    with torch.no_grad():
        result = model(**tokenizer("A short story.", return_tensors="pt"))
    if not torch.isfinite(result.logits).all():
        raise RuntimeError("Reloaded SFT model produced non-finite logits.")
    print("PASS: SFT update, checkpoint, final model/tokenizer reload.", flush=True)


def run_gfn():
    import main
    import wandb
    from replay_buffer import load_replay_buffer
    from rewards import load_reward

    with initialize_config_dir(version_base=None, config_dir=str(PROJECT / "configs")):
        config = compose(config_name="rationale_buffer", overrides=[
            "+experiment=infilling", f"model.name={OUTPUT / 'sft'}",
            "batch_size=1", "grad_acc=1", "train_steps=2",
            "max_len=8", "min_len=2", "max_eval_len=8",
            "replay_buffer.size=25", "+save_final=true",
        ])
    main.seed_everything(0)
    model, tokenizer = main.load_model(config.model)
    query = "Beginning: Sam was tired.\nEnding: He felt rested.\nMiddle:"
    dataset = {
        "train_queries": [query], "test_queries": [query],
        "train_sols": ["He felt rested."], "test_sols": ["He felt rested."],
        "train_rationales": [" He slept."], "train_weight": None,
        "test_num_sols": None,
        "encoded_train_queries": [tokenizer(query, return_tensors="pt").input_ids.cuda()],
        "encoded_test_queries": [tokenizer(query, return_tensors="pt").input_ids.cuda()],
        "encoded_train_sols": [tokenizer("He felt rested.", return_tensors="pt").input_ids.cuda()],
    }
    reward, vocab = load_reward(config.reward, model, tokenizer, dataset, config.min_len)
    buffer = load_replay_buffer(config.replay_buffer, model, dataset, tokenizer,
                                reward, config.max_len, config.min_len, vocab, True)
    before = {name: value.detach().clone() for name, value in model.named_parameters()
              if value.requires_grad}
    original = Path.cwd()
    directory = OUTPUT / "gfn"
    directory.mkdir()
    os.chdir(directory)
    try:
        OmegaConf.save(config, "resolved_config.yaml", resolve=True)
        wandb.init(mode="disabled", project="gfn-preflight")
        try:
            main.train(config, model, tokenizer, vocab, reward, buffer, dataset)
        finally:
            wandb.finish()
        check_weights(model)
        if not any(not torch.equal(before[name], value) for name, value in model.named_parameters()
                   if name in before):
            raise RuntimeError("GFN adapter did not change after training.")
        metrics = json.loads(Path("train_results.json").read_text())
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Non-finite final GFN metrics.")
        base = AutoModelForCausalLM.from_pretrained(OUTPUT / "sft")
        reloaded = PeftModel.from_pretrained(base, directory / "final_adapter")
        check_weights(reloaded)
        saved_tokenizer = AutoTokenizer.from_pretrained(directory / "final_adapter")
        with torch.no_grad():
            logits = reloaded(**saved_tokenizer(query, return_tensors="pt")).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("Reloaded GFN adapter produced non-finite logits.")
    finally:
        os.chdir(original)
    print("PASS: two GFN steps (first has zero warmup LR), adapter update and reload.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] == "--gfn":
        run_gfn()
    else:
        run_sft()
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(OUTPUT), "--gfn"],
                       check=True, cwd=PROJECT)
