import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)


class SFTSaveSmokeTest(unittest.TestCase):
    def test_training_saves_reloadable_final_model_and_tokenizer(self):
        with tempfile.TemporaryDirectory(prefix="sft-save-smoke-") as directory:
            root = Path(directory)
            source = root / "input"
            output = root / "output"
            vocabulary = {
                token: index for index, token in enumerate(
                    ["[UNK]", "[PAD]", "[EOS]", "Sam", "was", "tired",
                     "He", "slept", "and", "felt", "rested", "."]
                )
            }
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
                model_max_length=32,
            )
            torch.manual_seed(0)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=len(vocabulary), n_positions=32, n_ctx=32,
                n_embd=16, n_layer=1, n_head=2,
                bos_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ))
            before = {name: value.detach().clone()
                      for name, value in model.named_parameters()}
            model.save_pretrained(source)
            tokenizer.save_pretrained(source)
            dataset = root / "stories.csv"
            with dataset.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["text"])
                writer.writerows(
                    [["Sam was tired . He slept and felt rested ."]] * 4
                )
            env = dict(
                os.environ, STOP_RUNPOD_ON_EXIT="0",
                CUDA_VISIBLE_DEVICES="", ACCELERATE_USE_CPU="true",
                HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled",
                TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
            )
            env.pop("RUNPOD_CREDENTIALS_FILE", None)
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name("sft.py")),
                "--model_name", str(source), "--use_auth_token", "False",
                "--dataset_path", str(dataset), "--output_dir", str(output),
                "--batch_size", "1", "--gradient_accumulation_steps", "1",
                "--seq_length", "16", "--max_steps", "1", "--save_steps", "1",
                "--learning_rate", "0.001",
            ], env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("Requesting stop for pod", result.stdout)
            metrics = json.loads((output / "train_results.json").read_text())
            self.assertEqual(metrics["global_step"], 1)
            self.assertTrue(math.isfinite(metrics["train_loss"]))
            reloaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True)
            saved_tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
            self.assertTrue(any(
                not torch.equal(before[name], value)
                for name, value in reloaded.named_parameters()
            ), "Training did not update model weights.")
            reloaded.eval()
            with torch.no_grad():
                logits = reloaded(**saved_tokenizer(
                    "Sam was tired .", return_tensors="pt"
                )).logits
            self.assertTrue(torch.isfinite(logits).all())
            checkpoint = AutoModelForCausalLM.from_pretrained(
                output / "checkpoint-1", local_files_only=True
            )
            for name, value in reloaded.state_dict().items():
                self.assertTrue(torch.equal(value, checkpoint.state_dict()[name]), name)
            print("PASS: one CPU SFT update; checkpoint and final weights match; "
                  "final tokenizer reload and finite inference; no pod shutdown.")


if __name__ == "__main__":
    unittest.main()
