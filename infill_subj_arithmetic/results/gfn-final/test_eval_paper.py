import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location('standalone_eval', Path(__file__).with_name('eval.py'))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class PaperComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manager = evaluation.EvalManager(self.root, self.root, device='cpu')
        self.stories = pd.DataFrame({'story_id': [f'story-{i}' for i in range(103)]})
        rows = []
        for story_id in self.stories.story_id:
            for model, bert, bleu, gleu in [('SFT', 0.6, 0.1, 0.2), ('GFN', 0.7, 0.2, 0.3)]:
                for sample in range(2):
                    rows.append({'story_id': story_id, 'model': model, 'sample_id': sample,
                                 'generated_middle': 'text', 'bertscore_f1': bert, 'bleu': bleu, 'gleu': gleu,
                                 'generated_tokens': 3, 'stopped_on_eos': True, 'hit_length_cap': False,
                                 'empty_output': False, 'sentence_end': True, 'repeated_trigram_fraction': 0.0,
                                 'sft_terminal_logprob': -2.0, 'sft_per_token_logprob': -1.0,
                                 'sft_ending_per_token_logprob': -1.0, 'reference_middle_nll': 1.0})
        self.manager.results = pd.DataFrame(rows)
        self.manager.dataset_info = {'selected_stories': 103}
        self.manager.settings = {'samples_per_story': 2, 'text_metrics': {
            'model_dir': str(self.root), 'num_layers': 40, 'rescale_with_baseline': False}}
        self.config = SimpleNamespace(model_type='deberta', num_hidden_layers=48, hidden_size=1024)

    def tearDown(self):
        self.temporary.cleanup()

    def test_selects_100_and_reports_rescaled_scores_and_intervals(self):
        with patch('transformers.AutoConfig.from_pretrained', return_value=self.config):
            output = self.root / 'comparison'
            evaluation.save_paper_comparison(self.manager, self.stories, output, n_bootstrap=100, seed=3)
        scored = pd.read_json(output / 'paired_results.jsonl', lines=True)
        table = pd.read_csv(output / 'comparison.csv').set_index('method')
        metadata = json.loads((output / 'evaluation_metadata.json').read_text())
        baseline = metadata['settings']['paper_comparison']['bertscore_baseline_f1']
        self.assertEqual(len(scored), 400)
        self.assertEqual(set(scored.story_id), set(self.stories.story_id.iloc[:100]))
        self.assertTrue(np.allclose(scored['bertscore_f1'], (scored['bertscore_f1_unrescaled'] - baseline) / (1 - baseline)))
        self.assertAlmostEqual(table.loc['Supervised fine-tuning', 'local_bleu_4_percent'], 10)
        self.assertAlmostEqual(table.loc['GFlowNet fine-tuning', 'local_gleu_4_percent'], 30)
        self.assertAlmostEqual(table.loc['GFlowNet fine-tuning', 'paper_bertscore'], 0.184)
        self.assertTrue(pd.isna(table.loc['Prompting', 'local_bertscore_f1_rescaled']))
        self.assertTrue(table['local_gpt4eval'].isna().all())
        self.assertTrue(metadata['settings']['text_metrics']['rescale_with_baseline'])
        self.assertFalse(self.manager.settings['text_metrics']['rescale_with_baseline'])
        intervals = pd.read_csv(output / 'confidence_intervals.csv')
        self.assertEqual(set(intervals['metric']), {'bertscore_f1_rescaled', 'bleu_4_percent', 'gleu_4_percent'})
        paired = intervals[(intervals.metric == 'bleu_4_percent') & (intervals.comparison == 'GFN - SFT (paired)')].iloc[0]
        self.assertAlmostEqual(paired['estimate'], 10)
        self.assertAlmostEqual(paired['ci_low'], 10)
        self.assertEqual(paired['stories'], 100)

    def test_rejects_wrong_model_and_incomplete_pairs(self):
        wrong_config = SimpleNamespace(model_type='bert', num_hidden_layers=48, hidden_size=1024)
        with patch('transformers.AutoConfig.from_pretrained', return_value=wrong_config):
            with self.assertRaisesRegex(ValueError, 'DeBERTa'):
                evaluation.save_paper_comparison(self.manager, self.stories, self.root / 'wrong')
        self.manager.results = self.manager.results.iloc[1:].copy()
        with patch('transformers.AutoConfig.from_pretrained', return_value=self.config):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                evaluation.save_paper_comparison(self.manager, self.stories, self.root / 'incomplete')

    def test_rejects_missing_or_wrong_baseline(self):
        with self.assertRaises(FileNotFoundError):
            evaluation.paper_baseline(self.root / 'missing.tsv')
        path = self.root / 'bad.tsv'
        path.write_text('LAYER,F\n0,0.5\n')
        with self.assertRaisesRegex(ValueError, 'layer-40'):
            evaluation.paper_baseline(path)

    def test_evaluation_flags_select_outputs(self):
        manager = self.manager
        all_results = manager.results.copy()
        text_settings = manager.settings['text_metrics'].copy()
        cases = [(True, True, 103, 100), (True, True, 4, 7), (True, True, 7, 4),
                 (True, False, 3, 100), (False, True, 103, 4)]
        for general, paper, general_count, paper_count in cases:
            with self.subTest(general=general, paper=paper, general_count=general_count, paper_count=paper_count):
                selected_count = max(general_count if general else 0, paper_count if paper else 0)
                manager.stories = self.stories.iloc[:selected_count].copy()
                manager.dataset_info = {'selected_stories': selected_count}

                def fake_generate(**kwargs):
                    manager.results = all_results[all_results.story_id.isin(manager.stories.story_id)].copy()
                    manager.settings = {'samples_per_story': kwargs['samples_per_story']}
                    return manager.results.copy()

                def fake_score(*args, **kwargs):
                    manager.settings['text_metrics'] = text_settings.copy()
                    return manager.results.copy()

                output = self.root / f'pipeline-{general}-{paper}-{general_count}-{paper_count}'
                options = ['--general-stories', str(general_count), '--paper-stories', str(paper_count),
                           '--chunk-size', '50', '--samples-per-story', '2',
                           '--bootstrap-repetitions', '100', '--bert-model-dir', str(self.root),
                           '--bert-num-layers', '40', '--device', 'cpu', '--output-dir', str(output)]
                if not general:
                    options += ['--general-eval', 'false']
                if not paper:
                    options += ['--paper-eval', 'false']
                args = evaluation.build_parser().parse_args(options)
                self.assertEqual((args.general_eval, args.paper_eval), (general, paper))
                with patch.object(evaluation, 'prepare_evaluation', return_value=manager), \
                     patch.object(manager, 'evaluate', side_effect=fake_generate), \
                     patch.object(manager, 'score_text_metrics', side_effect=fake_score), \
                     patch('transformers.AutoConfig.from_pretrained', return_value=self.config):
                    evaluation.run_evaluation(args)
                self.assertEqual((output / 'scored/paired_results.jsonl').is_file(), general)
                self.assertEqual((output / 'confidence_intervals.csv').is_file(), general)
                self.assertEqual((output / 'paper_comparison/comparison.csv').is_file(), paper)
                self.assertEqual((output / 'paper_comparison/confidence_intervals.csv').is_file(), paper)
                raw = output / 'raw' if general else output / 'paper_comparison/raw'
                self.assertTrue((raw / 'paired_results.jsonl').is_file())
                expected_rows = (general_count if general else paper_count) * 4
                self.assertEqual(len(pd.read_json(raw / 'paired_results.jsonl', lines=True)), expected_rows)
                raw_metadata = json.loads((raw / 'evaluation_metadata.json').read_text())
                self.assertEqual(raw_metadata['dataset']['selected_stories'], expected_rows // 4)
                if general:
                    scored = pd.read_json(output / 'scored/paired_results.jsonl', lines=True)
                    self.assertEqual(set(scored.story_id), set(self.stories.story_id.iloc[:general_count]))
                    metadata = json.loads((output / 'scored/evaluation_metadata.json').read_text())
                    self.assertEqual(metadata['dataset']['selected_stories'], general_count)
                if paper:
                    comparison = output / 'paper_comparison'
                    scored = pd.read_json(comparison / 'paired_results.jsonl', lines=True)
                    self.assertEqual(set(scored.story_id), set(self.stories.story_id.iloc[:paper_count]))
                    metadata = json.loads((comparison / 'evaluation_metadata.json').read_text())
                    self.assertEqual(metadata['dataset']['selected_stories'], paper_count)
                    table = pd.read_csv(comparison / 'comparison.csv')
                    self.assertTrue(table.loc[table.local_model.notna(), 'local_stories'].eq(paper_count).all())
                    self.assertTrue(pd.read_csv(comparison / 'confidence_intervals.csv').stories.eq(paper_count).all())

    def test_preflight_selects_story_count_and_validates_flags(self):
        sft, adapter, bert = [self.root / name for name in ('sft', 'adapter', 'bert')]
        for directory, config, weights in [(sft, 'config.json', 'model.safetensors'),
                                           (adapter, 'adapter_config.json', 'adapter_model.bin'),
                                           (bert, 'config.json', 'model.safetensors')]:
            directory.mkdir()
            (directory / config).touch()
            (directory / weights).touch()

        def fake_load(manager, path, n_stories, seed, training_csv):
            manager.stories = pd.DataFrame({'story_id': range(n_stories)})
            manager.dataset_info = {'excluded_training_matches': 0}
            manager._training_csv_sha256 = 'test'

        base = ['--sft-dir', str(sft), '--adapter-dir', str(adapter), '--bert-model-dir', str(bert),
                '--bert-num-layers', '40', '--device', 'cpu']
        with patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.AutoConfig.from_pretrained', return_value=self.config), \
             patch.object(evaluation.EvalManager, 'load_rocstories', autospec=True, side_effect=fake_load):
            for flags, expected in [(['--stories', '103'], 103),
                                    (['--general-eval', 'false'], 100),
                                    (['--paper-eval', 'false', '--stories', '3'], 3),
                                    (['--general-stories', '4', '--paper-stories', '7'], 7),
                                    (['--general-eval', 'false', '--paper-stories', '5'], 5)]:
                args = evaluation.build_parser().parse_args(base + flags)
                self.assertEqual(len(evaluation.prepare_evaluation(args).stories), expected)
            args = evaluation.build_parser().parse_args(base + ['--paper-stories', '1'])
            with self.assertRaisesRegex(ValueError, 'at least two'):
                evaluation.prepare_evaluation(args)
        args = evaluation.build_parser().parse_args(['--general-eval', 'false', '--paper-eval', 'false'])
        with self.assertRaisesRegex(ValueError, 'At least one'):
            evaluation.prepare_evaluation(args)
        self.assertTrue(evaluation.build_parser().parse_args(['--paper-comparison']).paper_eval)

    def test_launcher_forwards_flags_and_rejects_both_disabled(self):
        env = os.environ.copy()
        env['EVAL_PYTHON'] = sys.executable
        result = subprocess.run(['bash', str(Path(__file__).with_name('run_eval.sh')), '--check-only',
                                 '--general-eval', 'false', '--paper-eval', 'false',
                                 '--validation-csv', 'unused-validation', '--training-csv', 'unused-training'],
                                env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('At least one', result.stderr)


if __name__ == '__main__':
    unittest.main()
