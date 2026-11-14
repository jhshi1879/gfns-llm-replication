"""Standalone evaluation of SFT and GFN story-infilling checkpoints."""

import json
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


class EvalManager:
    """Separate or paired ROCStories/Story Cloze adapted infilling evaluation."""

    SENTENCES = [f'sentence{i}' for i in range(1, 6)]

    def __init__(self, sft_dir: Union[str, Path], gfn_adapter_dir: Union[str, Path], device: str = 'auto',
                 cache_dir: Optional[Union[str, Path]] = None):
        self.sft_dir = Path(sft_dir).expanduser().resolve()
        self.gfn_adapter_dir = Path(gfn_adapter_dir).expanduser().resolve()
        self.cache_dir = (Path(cache_dir).expanduser().resolve() if cache_dir is not None
                          else self.gfn_adapter_dir.parent / '.evaluation_cache')
        if device not in ('auto', 'cpu', 'mps', 'cuda'):
            raise ValueError('device must be auto, cpu, mps, or cuda.')
        self.device = device
        self.stories: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self.dataset_info = {}
        self._training_csv_sha256 = None
        self.settings = {}
        self.cache_hit = False

    @classmethod
    def _read_stories(cls, path: Path, gfn_training_slice: bool = False) -> pd.DataFrame:
        import hashlib
        if not path.is_file():
            raise FileNotFoundError(f'ROCStories CSV is missing: {path}')
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        if gfn_training_slice:
            if len(frame) < 1000:
                raise ValueError('The GFN training slice requires at least 1,000 CSV rows.')
            frame = frame.iloc[100:1000].copy()
        source_format = 'rocstories'
        if not set(cls.SENTENCES).issubset(frame.columns):
            cloze_columns = [f'InputSentence{i}' for i in range(1, 5)] + ['RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding']
            missing = set(cloze_columns) - set(frame.columns)
            if missing:
                raise ValueError(f'Expected sentence1-sentence5 or Story Cloze columns; missing Cloze columns: {sorted(missing)}')
            labels = frame['AnswerRightEnding'].str.strip()
            if not labels.isin(['1', '2']).all():
                raise ValueError('Story Cloze answer labels must be 1 or 2.')
            for i in range(1, 5):
                frame[f'sentence{i}'] = frame[f'InputSentence{i}']
            frame['sentence5'] = np.where(labels.eq('1'), frame['RandomFifthSentenceQuiz1'], frame['RandomFifthSentenceQuiz2'])
            source_format = 'story_cloze_adapted_infilling'
        if frame.empty:
            raise ValueError('ROCStories CSV is empty.')
        for column in cls.SENTENCES:
            frame[column] = frame[column].str.strip()
        if frame[cls.SENTENCES].eq('').any(axis=None):
            raise ValueError('ROCStories contains empty sentences.')
        normalized = [[' '.join(sentence.casefold().split()) for sentence in sentences]
                      for sentences in frame[cls.SENTENCES].itertuples(index=False, name=None)]
        fingerprint = lambda parts: hashlib.sha256(json.dumps(parts, ensure_ascii=True).encode()).hexdigest()
        frame['story_id'] = [fingerprint(sentences) for sentences in normalized]
        frame['prompt_id'] = [fingerprint(sentences[:3] + sentences[4:]) for sentences in normalized]
        frame['context_id'] = [fingerprint(sentences[:4]) for sentences in normalized]
        frame['csv_row'] = frame.index.to_numpy() + 2
        duplicates = frame['story_id'].duplicated()
        duplicate_count = int(duplicates.sum())
        if duplicate_count:
            warnings.warn(f'Removed {duplicate_count} exact duplicate stories from {path.name}.', stacklevel=2)
            frame = frame.loc[~duplicates].copy()
        frame.attrs['duplicates_removed'] = duplicate_count
        frame.attrs['source_format'] = source_format
        return frame

    def load_rocstories(self, csv_path: Union[str, Path], n_stories: int = 10, seed: int = 0,
                       training_csv: Optional[Union[str, Path]] = None,
                       gfn_training_slice: bool = False) -> pd.DataFrame:
        import hashlib
        if isinstance(n_stories, bool) or not isinstance(n_stories, int) or n_stories < 1:
            raise ValueError('n_stories must be a positive integer.')
        self._validate_seed(seed)
        if not isinstance(gfn_training_slice, bool):
            raise ValueError('gfn_training_slice must be True or False.')
        if gfn_training_slice and training_csv is not None:
            raise ValueError('Do not exclude training stories from the training cohort.')
        self.results = None
        self.stories = None
        path = Path(csv_path).expanduser().resolve()
        frame = self._read_stories(path, gfn_training_slice=gfn_training_slice)
        duplicate_count = frame.attrs['duplicates_removed']
        source_format = frame.attrs['source_format']
        if gfn_training_slice and source_format != 'rocstories':
            raise ValueError('The GFN training slice requires a ROCStories CSV.')
        source_count, excluded = len(frame) + duplicate_count, 0
        overlap_status = 'unknown'
        self._training_csv_sha256 = None
        if gfn_training_slice:
            overlap_status = 'gfn_training_slice_assumed'
        elif training_csv is not None:
            training_path = Path(training_csv).expanduser().resolve()
            training = self._read_stories(training_path)
            self._training_csv_sha256 = hashlib.sha256(training_path.read_bytes()).hexdigest()
            matches = (frame['story_id'].isin(training['story_id']) |
                       frame['prompt_id'].isin(training['prompt_id']) |
                       frame['context_id'].isin(training['context_id']))
            excluded = int(matches.sum())
            frame = frame.loc[~matches].copy()
            overlap_status = 'no_normalized_story_prompt_or_context_match_in_supplied_training_csv'
        else:
            warnings.warn('Training overlap is unknown: these examples are not certified held-out.', stacklevel=2)
        if len(frame) < n_stories:
            raise ValueError(f'Requested {n_stories} stories, but only {len(frame)} remain after overlap exclusion.')
        frame = frame.sample(n=n_stories, random_state=seed).reset_index(drop=True)
        frame['beginning'] = frame[['sentence1', 'sentence2', 'sentence3']].agg(' '.join, axis=1)
        frame['ending'] = frame['sentence5']
        frame['reference_middle'] = frame['sentence4']
        frame['prompt'] = 'Beginning: ' + frame['beginning'] + '\nEnding: ' + frame['ending'] + '\nMiddle:'
        frame['overlap_status'] = overlap_status
        self.dataset_info = {'csv_path': str(path), 'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'source_stories': source_count, 'source_format': source_format,
                             'duplicates_removed': duplicate_count, 'excluded_training_matches': excluded,
                             'selected_stories': n_stories, 'selection_seed': seed, 'overlap_status': overlap_status,
                             'training_csv': str(Path(training_csv).expanduser().resolve()) if training_csv is not None else None}
        if gfn_training_slice:
            self.dataset_info['gfn_training_iloc'] = [100, 1000]
        self.stories = frame
        return frame.copy()

    @staticmethod
    def _validate_seed(seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
            raise ValueError('seed must be an integer in [0, 2**32).')

    @staticmethod
    def _completion_tokens(token_ids: list[int], eos_token_id: int) -> tuple[list[int], bool]:
        if eos_token_id in token_ids:
            return token_ids[:token_ids.index(eos_token_id)], True
        return token_ids, False

    @staticmethod
    def _repeated_trigram_fraction(text: str) -> float:
        words = re.findall(r'\w+', text.lower())
        grams = list(zip(words, words[1:], words[2:]))
        return 1 - len(set(grams)) / len(grams) if grams else 0.0

    def _load_models(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError('Inference dependencies are missing: install torch, transformers and peft.') from error
        for path in [self.sft_dir / 'config.json', self.gfn_adapter_dir / 'adapter_config.json']:
            if not path.is_file():
                raise FileNotFoundError(path)
        device = self.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA device is unavailable.')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Requested MPS device is unavailable.')
        tokenizer = AutoTokenizer.from_pretrained(str(self.sft_dir), local_files_only=True)
        adapter_tokenizer = AutoTokenizer.from_pretrained(str(self.gfn_adapter_dir), local_files_only=True)
        if tokenizer.get_vocab() != adapter_tokenizer.get_vocab() or tokenizer.eos_token_id != adapter_tokenizer.eos_token_id:
            raise ValueError('SFT and GFN tokenizers disagree; paired token-level evaluation is invalid.')
        if tokenizer.eos_token_id is None:
            raise ValueError('An EOS token is required for termination diagnostics.')
        dtype = torch.float16 if device == 'mps' else torch.float32
        base = AutoModelForCausalLM.from_pretrained(str(self.sft_dir), local_files_only=True,
            torch_dtype=dtype, device_map={'': device}, low_cpu_mem_usage=True)
        if base.get_input_embeddings().num_embeddings < len(tokenizer):
            raise ValueError('Tokenizer vocabulary exceeds checkpoint embedding size.')
        model = PeftModel.from_pretrained(base, str(self.gfn_adapter_dir), is_trainable=False, local_files_only=True)
        model.to(device)
        model.eval()
        return model, tokenizer, device

    @staticmethod
    def _conditional_logprob(model, context: list[int], continuation: list[int], device: str) -> tuple[float, float]:
        import torch
        if not context:
            raise ValueError('Likelihood scoring requires a non-empty context.')
        if not continuation:
            return 0.0, float('nan')  # Per-token likelihood is undefined for an empty completion.
        limit = model.config.max_position_embeddings
        if len(context) + len(continuation) > limit:
            raise ValueError('Scoring sequence exceeds model context capacity; do not silently truncate the story.')
        tokens = torch.tensor([context + continuation], device=device)
        with torch.inference_mode():
            logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits
            logprobs = logits[0, len(context) - 1:-1].float().log_softmax(-1)
            targets = torch.tensor(continuation, device=device).unsqueeze(-1)
            selected = logprobs.gather(-1, targets).squeeze(-1)
            if not torch.isfinite(selected).all():
                raise RuntimeError('Model produced non-finite token log-probabilities during evaluation.')
            return float(selected.sum().item()), float(selected.mean().item())

    def _cache_path(self, options: dict) -> Path:
        import hashlib
        import importlib.metadata
        import platform
        names = {'config.json', 'generation_config.json', 'adapter_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
                 'merges.txt', 'pytorch_model.bin', 'model.safetensors', 'adapter_model.bin', 'adapter_model.safetensors'}
        assets = {}
        for label, directory in [('sft', self.sft_dir), ('adapter', self.gfn_adapter_dir)]:
            if not directory.is_dir():
                raise FileNotFoundError(f'Model directory not found: {directory}')
            files = sorted(path for path in directory.rglob('*') if path.is_file() and
                           (path.name in names or path.name.startswith(('pytorch_model-', 'model-', 'adapter_model-'))))
            if not files:
                raise FileNotFoundError(f'No model or tokenizer files found in {directory}')
            assets[label] = [{'path': str(path.relative_to(directory)), 'size': path.stat().st_size,
                              'mtime_ns': path.stat().st_mtime_ns, 'ctime_ns': path.stat().st_ctime_ns}
                             for path in files]
        columns = ['story_id', 'prompt', 'beginning', 'ending', 'reference_middle', 'overlap_status']
        identity = {'schema': 1, 'stories': self.stories[columns].to_dict('split'),
                    'dataset': self.dataset_info, 'options': options, 'device': self.device,
                    'sft_dir': str(self.sft_dir), 'adapter_dir': str(self.gfn_adapter_dir),
                    'assets': assets, 'python': platform.python_version(),
                    'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft')}}
        if self.device == 'auto':
            import torch
            identity['resolved_device'] = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        if self.device == 'cuda' or identity.get('resolved_device') == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA device is unavailable.')
            identity['cuda'] = {'name': torch.cuda.get_device_name(0),
                                'capability': torch.cuda.get_device_capability(0),
                                'runtime': torch.version.cuda}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=True).encode('utf-8')).hexdigest()
        prefix = (options['model_name'] or 'paired').lower()
        return self.cache_dir / f'{prefix}-{digest}.json'

    def _save_cache(self, cache_file: Path) -> None:
        import os
        import tempfile
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=cache_file.parent,
                                             prefix='.eval-', suffix='.tmp', delete=False) as temporary:
                temporary_path = Path(temporary.name)
                json.dump({'cache_key': cache_file.stem, 'settings': self.settings,
                           'results': self.results.to_json(orient='records', double_precision=15)},
                          temporary, ensure_ascii=False, allow_nan=False)
            os.replace(temporary_path, cache_file)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def evaluate(self, samples_per_story: int = 3, max_new_tokens: int = 25, min_new_tokens: int = 0,
                 temperature: float = 0.9, seed: int = 0, model_name: Optional[str] = None,
                 verbose: bool = True, force_recompute: bool = False) -> pd.DataFrame:
        from contextlib import nullcontext
        from tqdm import tqdm
        if self.stories is None or self.stories.empty:
            raise RuntimeError('Call load_rocstories() before evaluate().')
        for name, value in [('samples_per_story', samples_per_story), ('max_new_tokens', max_new_tokens)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if isinstance(min_new_tokens, bool) or not isinstance(min_new_tokens, int) or not 0 <= min_new_tokens < max_new_tokens:
            raise ValueError('min_new_tokens must be an integer in [0, max_new_tokens).')
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive.')
        self._validate_seed(seed)
        if model_name is not None and model_name not in ('SFT', 'GFN'):
            raise ValueError('model_name must be SFT, GFN, or None for a paired evaluation.')
        if not isinstance(verbose, bool):
            raise ValueError('verbose must be True or False.')
        if not isinstance(force_recompute, bool):
            raise ValueError('force_recompute must be True or False.')
        models = ['SFT', 'GFN'] if model_name is None else [model_name]
        self.results = None
        self.cache_hit = False
        options = {'model_name': model_name, 'samples_per_story': samples_per_story,
                   'max_new_tokens': max_new_tokens, 'min_new_tokens': min_new_tokens,
                   'temperature': float(temperature), 'seed': seed}
        cache_file = self._cache_path(options)
        if cache_file.is_file() and not force_recompute:
            cached = json.loads(cache_file.read_text(encoding='utf-8'))
            if cached['cache_key'] != cache_file.stem:
                raise ValueError(f'Invalid evaluation cache key: {cache_file}')
            self.results = pd.DataFrame(json.loads(cached['results']))
            for field in ('sft_terminal_logprob', 'sft_per_token_logprob', 'sft_ending_per_token_logprob',
                          'reference_middle_nll', 'repeated_trigram_fraction'):
                self.results[field] = pd.to_numeric(self.results[field], errors='raise')
            self.settings = cached['settings']
            results = self._validated_results()
            expected = {(story_id, sample_id, name) for story_id in self.stories['story_id']
                        for sample_id in range(samples_per_story) for name in models}
            actual = set(results[['story_id', 'sample_id', 'model']].itertuples(index=False, name=None))
            if len(results) != len(expected) or actual != expected:
                raise ValueError(f'Incomplete or mismatched evaluation cache: {cache_file}')
            self.cache_hit = True
            if verbose:
                print(f'Evaluation cache HIT: {cache_file}', flush=True)
            return results
        if verbose:
            print(f'Evaluation cache MISS; loading checkpoints for {", ".join(models)}...', flush=True)
        model, tokenizer, device = self._load_models()
        import torch
        self.settings = {'samples_per_story': samples_per_story, 'max_new_tokens': max_new_tokens,
                         'min_new_tokens': min_new_tokens, 'temperature': temperature, 'generation_seed': seed,
                         'device': device, 'model_dtype': str(next(model.parameters()).dtype), 'models': models,
                         'sft_dir': str(self.sft_dir), 'gfn_adapter_dir': str(self.gfn_adapter_dir)}
        if device == 'cuda':
            self.settings['cuda'] = {'name': torch.cuda.get_device_name(0),
                                     'capability': list(torch.cuda.get_device_capability(0)),
                                     'runtime': torch.version.cuda}
        records = []
        with torch.inference_mode(), tqdm(total=len(self.stories) * samples_per_story * len(models),
                                          desc='Evaluating stories', unit='sample', disable=not verbose) as progress:
            for position, (story_index, story) in enumerate(self.stories.iterrows(), start=1):
                prompt_ids = tokenizer.encode(story['prompt'], add_special_tokens=False)
                reference_ids = tokenizer.encode(' ' + story['reference_middle'], add_special_tokens=False)
                ending_ids = tokenizer.encode(' ' + story['ending'], add_special_tokens=False)
                if len(prompt_ids) + max_new_tokens + len(ending_ids) > model.config.max_position_embeddings:
                    raise ValueError(f'Story {story["story_id"]} exceeds context capacity.')
                inputs = torch.tensor([prompt_ids], device=device)
                for model_name in models:
                    with model.disable_adapter() if model_name == 'SFT' else nullcontext():
                        _, reference_mean = self._conditional_logprob(model, prompt_ids, reference_ids, device)
                    for sample_id in range(samples_per_story):
                        if verbose:
                            progress.set_postfix_str(f'{model_name}, story {position}/{len(self.stories)}, sample {sample_id + 1}/{samples_per_story}')
                        sample_seed = (seed + int(story_index) * samples_per_story + sample_id) % (2 ** 32)
                        torch.manual_seed(sample_seed)
                        with model.disable_adapter() if model_name == 'SFT' else nullcontext():
                            generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                                num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                                max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                                use_cache=True, return_dict_in_generate=False)
                        raw_ids = generated[0, len(prompt_ids):].tolist()
                        completion, stopped = self._completion_tokens(raw_ids, tokenizer.eos_token_id)
                        middle = tokenizer.decode(completion, skip_special_tokens=True).strip()
                        with model.disable_adapter():
                            total, mean = self._conditional_logprob(model, prompt_ids, completion, device)
                            _, ending_mean = self._conditional_logprob(model, prompt_ids + completion, ending_ids, device)
                        records.append({'story_id': story['story_id'], 'sample_id': sample_id, 'model': model_name,
                            'seed': sample_seed, 'prompt': story['prompt'], 'beginning': story['beginning'],
                            'ending': story['ending'], 'reference_middle': story['reference_middle'],
                            'overlap_status': story['overlap_status'], 'generated_middle': middle,
                            'generated_token_ids': completion, 'generated_tokens': len(completion),
                            'stopped_on_eos': stopped, 'hit_length_cap': not stopped and len(raw_ids) >= max_new_tokens,
                            'empty_output': not completion, 'sentence_end': middle.endswith(('.', '!', '?')),
                            'repeated_trigram_fraction': self._repeated_trigram_fraction(middle),
                            'sft_terminal_logprob': total, 'sft_per_token_logprob': mean,
                            'sft_ending_per_token_logprob': ending_mean, 'reference_middle_nll': -reference_mean})
                        progress.update(1)
        self.results = pd.DataFrame(records)
        self._validated_results()
        self._save_cache(cache_file)
        if verbose:
            print(f'Saved evaluation cache: {cache_file}', flush=True)
        return self.results.copy()

    def _validated_results(self) -> pd.DataFrame:
        if self.results is None or self.results.empty:
            raise RuntimeError('No evaluation results available; run evaluate() first.')
        if not self.results['model'].isin(['SFT', 'GFN']).all():
            raise ValueError('Evaluation results contain an unknown model.')
        if self.results.duplicated(['story_id', 'sample_id', 'model']).any():
            raise ValueError('Evaluation results contain duplicate story/sample/model rows.')
        return self.results.copy()

    def _paired_results(self) -> pd.DataFrame:
        results = self._validated_results()
        counts = results.groupby(['story_id', 'sample_id', 'model']).size().unstack('model', fill_value=0)
        if set(counts.columns) != {'SFT', 'GFN'} or not counts.eq(1).all(axis=None):
            raise ValueError('Every story/sample must have exactly one SFT and one GFN result.')
        return self.results.copy()

    def score_text_metrics(self, bert_model_dir: Union[str, Path], bert_num_layers: int,
                           device: str = 'cpu', batch_size: int = 16, verbose: bool = True) -> pd.DataFrame:
        """Score cached or new generations using a local BERTScore model, without changing the generation cache."""
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from nltk.translate.gleu_score import sentence_gleu
        from bert_score import BERTScorer
        import importlib.metadata
        results = self._validated_results()
        model_dir = Path(bert_model_dir).expanduser().resolve()
        if not model_dir.is_dir() or not (model_dir / 'config.json').is_file():
            raise FileNotFoundError(f'Provide a local BERTScore model directory containing config.json: {model_dir}')
        assets = [{'path': str(p.relative_to(model_dir)), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                   'ctime_ns': p.stat().st_ctime_ns} for p in sorted(model_dir.rglob('*')) if p.is_file()]
        if isinstance(bert_num_layers, bool) or not isinstance(bert_num_layers, int) or bert_num_layers < 1:
            raise ValueError('bert_num_layers must be a positive integer appropriate for the model.')
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        if not isinstance(verbose, bool):
            raise ValueError('verbose must be True or False.')
        if not results['generated_middle'].map(lambda x: isinstance(x, str)).all() or not results['reference_middle'].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError('Text metrics require string generations and nonempty string references.')
        candidates = results['generated_middle'].str.strip().tolist()
        references = results['reference_middle'].str.strip().tolist()
        smooth = SmoothingFunction().method1
        bleu, gleu = [], []
        for candidate, reference in zip(candidates, references):
            tokens, reference_tokens = candidate.split(), reference.split()
            bleu.append(float(sentence_bleu([reference_tokens], tokens, smoothing_function=smooth)) if tokens else 0.0)
            gleu.append(float(sentence_gleu([reference_tokens], tokens)) if tokens else 0.0)
        f1 = np.zeros(len(results), dtype=float)
        nonempty = [i for i, candidate in enumerate(candidates) if candidate]
        if nonempty:
            scorer = BERTScorer(model_type=str(model_dir), num_layers=bert_num_layers, lang='en',
                                device=device, rescale_with_baseline=False, batch_size=batch_size)
            _, _, scores = scorer.score([candidates[i] for i in nonempty], [references[i] for i in nonempty],
                                        batch_size=batch_size, verbose=verbose)
            f1[nonempty] = scores.detach().cpu().numpy()
            if not np.isfinite(f1).all():
                raise ValueError('BERTScore returned non-finite scores.')
        results['bleu'], results['gleu'], results['bertscore_f1'] = bleu, gleu, f1
        self.results = results
        self.settings['text_metrics'] = {'model_dir': str(model_dir), 'num_layers': bert_num_layers,
            'rescale_with_baseline': False, 'device': device, 'batch_size': batch_size,
            'model_assets': assets,
            'nltk': importlib.metadata.version('nltk'), 'bert_score': importlib.metadata.version('bert-score'),
            'transformers': importlib.metadata.version('transformers')}
        return results.copy()

    def story_metrics(self) -> pd.DataFrame:
        """Aggregate samples before inference so stories, not generations, are the statistical units."""
        results = self._validated_results()
        columns = ['generated_tokens', 'stopped_on_eos', 'hit_length_cap', 'empty_output', 'sentence_end',
                   'repeated_trigram_fraction', 'sft_terminal_logprob', 'sft_per_token_logprob',
                   'sft_ending_per_token_logprob', 'reference_middle_nll', 'bleu', 'gleu', 'bertscore_f1']
        columns = [column for column in columns if column in results.columns]
        grouped = results.groupby(['model', 'story_id'], sort=True)
        means = grouped[columns].mean().mask(grouped[columns].count().ne(grouped.size(), axis=0))
        diversity = grouped['generated_middle'].agg(['size', 'nunique'])
        means['within_prompt_duplicate_fraction'] = 1 - diversity['nunique'] / diversity['size']
        return means.reset_index()

    @staticmethod
    def _bootstrap_config(n_bootstrap: int, confidence: float, seed: int):
        if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 100:
            raise ValueError('n_bootstrap must be an integer of at least 100.')
        if not np.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError('confidence must be between 0 and 1.')
        EvalManager._validate_seed(seed)

    @staticmethod
    def _bootstrap_interval(values, rng, n_bootstrap, confidence, other=None):
        values = np.asarray(values, dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError('Each metric requires at least two stories with finite values; inspect missing values first.')
        if other is not None:
            other = np.asarray(other, dtype=float)
            if len(other) < 2 or not np.isfinite(other).all():
                raise ValueError('Each cohort requires at least two stories with finite metric values.')
        estimate = values.mean() - (other.mean() if other is not None else 0)
        draws = np.empty(n_bootstrap)
        for start in range(0, n_bootstrap, 1000):
            size = min(1000, n_bootstrap - start)
            sample = values[rng.integers(len(values), size=(size, len(values)))].mean(axis=1)
            if other is not None:
                sample -= other[rng.integers(len(other), size=(size, len(other)))].mean(axis=1)
            draws[start:start + size] = sample
        low, high = np.quantile(draws, [(1 - confidence) / 2, (1 + confidence) / 2])
        return float(estimate), float(low), float(high)

    def confidence_intervals(self, metrics=None, n_bootstrap: int = 10000,
                             confidence: float = 0.95, seed: int = 0) -> pd.DataFrame:
        """Story-bootstrap model means and paired GFN-minus-SFT differences."""
        self._bootstrap_config(n_bootstrap, confidence, seed)
        stories = self.story_metrics()
        if set(stories['model']) == {'SFT', 'GFN'}:
            self._paired_results()
        metrics = list(stories.columns[2:]) if metrics is None else list(metrics)
        if not metrics or any(metric not in stories.columns[2:] for metric in metrics):
            raise ValueError('Select available numeric story metrics.')
        rng = np.random.default_rng(seed)
        rows = []
        for metric in metrics:
            table = stories.pivot(index='story_id', columns='model', values=metric)
            for model in table.columns:
                values = table[model].dropna().to_numpy()
                if len(values) != table[model].notna().sum() or len(values) != len(stories[stories['model'] == model]):
                    raise ValueError(f'{metric} has missing values for {model}; choose another metric or inspect them.')
                estimate, low, high = self._bootstrap_interval(values, rng, n_bootstrap, confidence)
                rows.append({'metric': metric, 'comparison': model, 'stories': len(values), 'estimate': estimate, 'ci_low': low, 'ci_high': high})
            if set(table.columns) == {'SFT', 'GFN'}:
                if table[['GFN', 'SFT']].isna().any(axis=None):
                    raise ValueError(f'{metric} requires matching, finite SFT and GFN story scores.')
                differences = (table['GFN'] - table['SFT']).to_numpy()
                estimate, low, high = self._bootstrap_interval(differences, rng, n_bootstrap, confidence)
                rows.append({'metric': metric, 'comparison': 'GFN - SFT (paired)', 'stories': len(differences), 'estimate': estimate, 'ci_low': low, 'ci_high': high})
        result = pd.DataFrame(rows)
        result['confidence'], result['bootstrap_repetitions'], result['bootstrap_seed'] = confidence, n_bootstrap, seed
        return result

    def compare_cohorts(self, training, metrics=None, n_bootstrap: int = 10000,
                        confidence: float = 0.95, seed: int = 0) -> pd.DataFrame:
        """Compare this held-out cohort against a separately evaluated training cohort."""
        self._bootstrap_config(n_bootstrap, confidence, seed)
        if self.dataset_info.get('overlap_status') != 'no_normalized_story_prompt_or_context_match_in_supplied_training_csv':
            raise ValueError('The held-out cohort must have been checked against the supplied full training corpus.')
        if training.dataset_info.get('overlap_status') != 'gfn_training_slice_assumed':
            raise ValueError('The training cohort must use gfn_training_slice=True.')
        if self.dataset_info.get('training_csv') != training.dataset_info.get('csv_path'):
            raise ValueError('Held-out exclusion must use the same full CSV as the training cohort.')
        import hashlib
        current_hash = hashlib.sha256(Path(training.dataset_info['csv_path']).read_bytes()).hexdigest()
        if current_hash != training.dataset_info.get('csv_sha256'):
            raise ValueError('The training CSV changed since the training cohort was loaded.')
        if current_hash != self._training_csv_sha256:
            raise ValueError('The training CSV changed since held-out overlap was checked; reload the cohort.')
        keys = ['samples_per_story', 'max_new_tokens', 'min_new_tokens', 'temperature', 'generation_seed',
                'device', 'model_dtype', 'models', 'sft_dir', 'gfn_adapter_dir', 'text_metrics']
        for key in keys:
            if key != 'text_metrics' and (key not in self.settings or key not in training.settings):
                raise ValueError(f'Cohort evaluation settings are missing: {key}')
            if self.settings.get(key) != training.settings.get(key):
                raise ValueError(f'Cohort evaluation settings differ: {key}')
        held = self.story_metrics()
        train = training.story_metrics()
        if set(held['story_id']) & set(train['story_id']):
            raise ValueError('Training and held-out story IDs overlap.')
        if set(held['model']) != set(train['model']):
            raise ValueError('Both cohorts must evaluate the same models.')
        for manager in (self, training):
            results = manager._validated_results()
            if set(results['model']) == {'SFT', 'GFN'}:
                manager._paired_results()
            counts = results.groupby(['model', 'story_id']).size()
            if not counts.eq(manager.settings['samples_per_story']).all():
                raise ValueError('Cohort results do not match samples_per_story.')
        metrics = list(held.columns[2:]) if metrics is None else list(metrics)
        if not metrics or any(m not in held.columns[2:] or m not in train.columns[2:] for m in metrics):
            raise ValueError('Select metrics available in both cohorts.')
        rng = np.random.default_rng(seed)
        rows = []
        for metric in metrics:
            for model in sorted(held['model'].unique()):
                h = held.loc[held['model'] == model, metric].to_numpy()
                t = train.loc[train['model'] == model, metric].to_numpy()
                estimate, low, high = self._bootstrap_interval(h, rng, n_bootstrap, confidence, other=t)
                rows.append({'metric': metric, 'model': model, 'comparison': 'held-out - training',
                             'held_out_stories': len(h), 'training_stories': len(t),
                             'held_out_mean': float(np.mean(h)), 'training_mean': float(np.mean(t)),
                             'estimate': estimate, 'ci_low': low, 'ci_high': high})
        result = pd.DataFrame(rows)
        result['confidence'], result['bootstrap_repetitions'], result['bootstrap_seed'] = confidence, n_bootstrap, seed
        return result

    def print_examples(self, n_examples: int = 3) -> None:
        if isinstance(n_examples, bool) or not isinstance(n_examples, int) or n_examples < 1:
            raise ValueError('n_examples must be a positive integer.')
        examples = self._validated_results().drop_duplicates(['model', 'story_id']).head(n_examples)
        for number, (_, row) in enumerate(examples.iterrows(), start=1):
            print(f'Example {number} | {row["model"]} | sample {row["sample_id"]} | seed {row["seed"]}')
            print(f'Beginning: {row["beginning"]}')
            print(f'Ending: {row["ending"]}')
            print(f'Reference middle: {row["reference_middle"]}')
            print(f'Generated middle: {row["generated_middle"]!r}')
            print(f'Tokens: {row["generated_tokens"]} | EOS: {row["stopped_on_eos"]} | cap hit: {row["hit_length_cap"]}')
            print(f'SFT log-prob/token: {row["sft_per_token_logprob"]:.3f} | ending log-prob/token: {row["sft_ending_per_token_logprob"]:.3f}\n')

    def side_by_side(self) -> pd.DataFrame:
        results = self._paired_results()
        fields = ['generated_middle', 'generated_tokens', 'stopped_on_eos', 'sentence_end',
                  'sft_terminal_logprob', 'sft_per_token_logprob', 'sft_ending_per_token_logprob', 'reference_middle_nll']
        paired = results.pivot(index=['story_id', 'sample_id'], columns='model', values=fields)
        paired.columns = [f'{model}_{metric}' for metric, model in paired.columns]
        details = results[['story_id', 'beginning', 'ending', 'reference_middle', 'overlap_status']].drop_duplicates('story_id')
        return paired.reset_index().merge(details, on='story_id', validate='many_to_one')

    def summary(self) -> pd.DataFrame:
        results = self._validated_results()
        summary = results.groupby('model').agg(outputs=('generated_middle', 'size'),
            mean_generated_tokens=('generated_tokens', 'mean'), eos_rate=('stopped_on_eos', 'mean'),
            cap_hit_rate=('hit_length_cap', 'mean'), empty_rate=('empty_output', 'mean'),
            sentence_end_rate=('sentence_end', 'mean'), repeated_trigram_fraction=('repeated_trigram_fraction', 'mean'),
            sft_terminal_logprob=('sft_terminal_logprob', 'mean'), sft_per_token_logprob=('sft_per_token_logprob', 'mean'),
            sft_ending_per_token_logprob=('sft_ending_per_token_logprob', 'mean'), reference_middle_nll=('reference_middle_nll', 'mean'))
        diversity = results.groupby(['model', 'story_id'])['generated_middle'].agg(['size', 'nunique'])
        summary['within_prompt_duplicate_fraction'] = (1 - diversity['nunique'] / diversity['size']).groupby('model').mean()
        for metric in ('bleu', 'gleu', 'bertscore_f1'):
            if metric in results.columns:
                summary[metric] = results.groupby('model')[metric].mean()
        return summary

    def plot_diagnostics(self) -> tuple:
        import matplotlib.pyplot as plt

        results = self._validated_results()
        summary = self.summary()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for model_name, subset in results.groupby('model'):
            axes[0].hist(subset['generated_tokens'], bins=np.arange(results['generated_tokens'].max() + 2) - 0.5,
                         alpha=0.5, label=model_name)
        axes[0].set(title='Exact non-EOS output length', xlabel='Generated tokens', ylabel='Count')
        axes[0].legend()
        summary[['eos_rate', 'cap_hit_rate', 'empty_rate', 'sentence_end_rate']].T.plot.bar(ax=axes[1])
        axes[1].set(title='Termination/completion heuristics', ylabel='Fraction')
        summary[['within_prompt_duplicate_fraction', 'repeated_trigram_fraction']].T.plot.bar(ax=axes[2])
        axes[2].set(title='Repetition diagnostics', ylabel='Fraction')
        fig.suptitle(' / '.join(summary.index) + ' evaluation diagnostics')
        fig.tight_layout()
        return fig, axes

    def save_results(self, output_dir: Union[str, Path]):
        results = self._validated_results()
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        result_name = 'paired_results.jsonl' if set(results['model']) == {'SFT', 'GFN'} else 'results.jsonl'
        paths = [target / result_name, target / 'evaluation_metadata.json']
        if any(path.exists() for path in paths):
            raise FileExistsError('Evaluation output already exists; select a new directory to preserve prior runs.')
        results.to_json(paths[0], orient='records', lines=True, force_ascii=False)
        paths[1].write_text(json.dumps({'dataset': self.dataset_info, 'settings': self.settings}, indent=2))


def build_parser():
    import argparse
    import os

    def boolean_value(value):
        if value.lower() not in ('true', 'false'):
            raise argparse.ArgumentTypeError('Expected true or false.')
        return value.lower() == 'true'

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description='Evaluate paired SFT/GFN infilling on held-out Story Cloze stories.',
        epilog=('Install the project requirements plus nltk and bert-score in the selected Python environment. '
                'Copy the SFT checkpoint, GFN adapter, both CSVs and a BERTScore model to the pod first. '
                'BERTScore uses a local model and unrescaled F1; no model is downloaded. '
                'Completed generation chunks are cached for resumption.'))
    parser.add_argument('--sft-dir', default=os.environ.get('GFN_BASE_MODEL') or str(project_dir / 'output/stories_sft'))
    parser.add_argument('--adapter-dir', default=os.environ.get('GFN_ADAPTER_DIR') or str(script_dir / 'final_adapter'))
    parser.add_argument('--validation-csv', default=str(project_dir / 'data/story-cloze-winter2018-val.csv'))
    parser.add_argument('--training-csv', default=str(project_dir / 'data/rocstories-eval-winter2017.csv'))
    parser.add_argument('--bert-model-dir', default=os.environ.get('EVAL_BERT_MODEL_DIR') or None,
                        help='Existing local BERTScore model directory (required).')
    parser.add_argument('--bert-num-layers', type=int, default=os.environ.get('EVAL_BERT_NUM_LAYERS') or None,
                        help='BERTScore layer number (required; 40 for DeBERTa-xlarge-MNLI).')
    parser.add_argument('--bert-batch-size', type=int, default=16)
    parser.add_argument('--stories', '--general-stories', dest='stories', type=int, default=900,
                        help='General-evaluation story count (default: 900).')
    parser.add_argument('--paper-stories', type=int, default=100,
                        help='Paper-comparison story count (default: 100).')
    parser.add_argument('--samples-per-story', type=int, default=2)
    parser.add_argument('--chunk-size', type=int, default=50,
                        help='Stories per reusable generation checkpoint (default: 50).')
    parser.add_argument('--max-new-tokens', type=int, default=25)
    parser.add_argument('--min-new-tokens', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default='cuda')
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--general-eval', type=boolean_value, metavar='{true,false}', default=True,
                        help='Run the general held-out evaluation (default: true).')
    parser.add_argument('--paper-eval', type=boolean_value, metavar='{true,false}', default=True,
                        help='Run the paper comparison (default: true); '
                             'requires local DeBERTa-xlarge-MNLI at layer 40.')
    parser.add_argument('--paper-comparison', dest='paper_eval', action='store_true', default=argparse.SUPPRESS,
                        help='Legacy alias for --paper-eval true.')
    parser.add_argument('--output-dir', help='New directory for this run; defaults to a timestamped directory.')
    parser.add_argument('--cache-dir', default=str(script_dir / 'evaluations' / '.evaluation_cache'))
    parser.add_argument('--force-recompute', action='store_true',
                        help='Regenerate all chunks instead of using matching cached chunks.')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs and GPU without generating.')
    return parser


def prepare_evaluation(args):
    import importlib.util

    if not args.general_eval and not args.paper_eval:
        raise ValueError('At least one of --general-eval or --paper-eval must be true.')
    required = {'torch': 'torch', 'transformers': 'transformers', 'peft': 'peft',
                'nltk': 'nltk', 'bert_score': 'bert-score', 'tqdm': 'tqdm', 'accelerate': 'accelerate'}
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError('Missing evaluation dependencies: ' + ', '.join(missing) +
                           '. Install the project requirements and nltk/bert-score in the selected Python environment.')
    import torch
    import transformers
    import peft
    import nltk
    import bert_score
    import tqdm

    del peft, nltk, tqdm
    for name in ('stories', 'paper_stories', 'samples_per_story', 'chunk_size', 'max_new_tokens', 'bert_batch_size'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    if args.general_eval and args.stories < 2:
        raise ValueError('At least two stories are required for confidence intervals.')
    if args.paper_eval and args.paper_stories < 2:
        raise ValueError('The paper comparison requires at least two stories for confidence intervals.')
    if not 0 <= args.min_new_tokens < args.max_new_tokens:
        raise ValueError('--min-new-tokens must be nonnegative and less than --max-new-tokens.')
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError('--temperature must be finite and positive.')
    EvalManager._validate_seed(args.seed)
    EvalManager._bootstrap_config(args.bootstrap_repetitions, args.confidence, args.seed)
    if not args.bert_model_dir or args.bert_num_layers is None or args.bert_num_layers < 1:
        raise ValueError('Provide --bert-model-dir and a positive --bert-num-layers for local BERTScore.')
    if args.output_dir and Path(args.output_dir).expanduser().exists():
        raise FileExistsError(f'Output directory already exists: {args.output_dir}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run on a GPU pod or explicitly select --device cpu.')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable.')

    manager = EvalManager(args.sft_dir, args.adapter_dir, device=args.device, cache_dir=args.cache_dir)
    for directory, config in [(manager.sft_dir, 'config.json'), (manager.gfn_adapter_dir, 'adapter_config.json'),
                              (Path(args.bert_model_dir).expanduser().resolve(), 'config.json')]:
        if not (directory / config).is_file():
            raise FileNotFoundError(f'Missing {config} in {directory}.')
    for directory, label in [(manager.sft_dir, 'SFT'), (manager.gfn_adapter_dir, 'GFN adapter'),
                             (Path(args.bert_model_dir).expanduser().resolve(), 'BERTScore')]:
        weights = list(directory.glob('*.safetensors')) + list(directory.glob('pytorch_model*.bin'))
        if label == 'GFN adapter':
            weights += list(directory.glob('adapter_model*.bin'))
        if not weights:
            raise FileNotFoundError(f'No local {label} model weights found in {directory}.')
        transformers.AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    bert_config = transformers.AutoConfig.from_pretrained(str(Path(args.bert_model_dir).expanduser().resolve()),
                                                           local_files_only=True)
    layer_count = getattr(bert_config, 'num_hidden_layers', None)
    if layer_count is not None and args.bert_num_layers > layer_count:
        raise ValueError(f'BERTScore layer {args.bert_num_layers} exceeds the model layer count {layer_count}.')
    if args.paper_eval:
        validate_paper_model(bert_config, args.bert_num_layers)
        baseline_path = Path(bert_score.__file__).resolve().parent / 'rescale_baseline/en/microsoft/deberta-xlarge-mnli.tsv'
        paper_baseline(baseline_path)

    n_stories = max(args.stories if args.general_eval else 0, args.paper_stories if args.paper_eval else 0)
    manager.load_rocstories(args.validation_csv, n_stories=n_stories, seed=args.seed,
                            training_csv=args.training_csv)
    manager.dataset_info['training_csv_sha256'] = manager._training_csv_sha256
    print(f"Selected {len(manager.stories)} held-out stories; excluded "
          f"{manager.dataset_info['excluded_training_matches']} known-training overlaps.", flush=True)
    if args.device == 'cuda':
        print(f'CUDA device: {torch.cuda.get_device_name(0)}', flush=True)
    return manager


def run_evaluation(args):
    from datetime import datetime, timezone

    manager = prepare_evaluation(args)
    if args.check_only:
        print('Preflight passed; no inference was run.', flush=True)
        return
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = Path(__file__).resolve().parent / 'evaluations' / (
            'heldout-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output_dir.mkdir(parents=True, exist_ok=False)
    stories = manager.stories.copy()
    chunks = []
    initial_settings = None
    print(f'Output directory: {output_dir}', flush=True)
    try:
        for start in range(0, len(stories), args.chunk_size):
            end = min(start + args.chunk_size, len(stories))
            print(f'Generation chunk {start + 1}-{end} of {len(stories)}', flush=True)
            manager.stories = stories.iloc[start:end].copy()
            chunk = manager.evaluate(samples_per_story=args.samples_per_story, max_new_tokens=args.max_new_tokens,
                                     min_new_tokens=args.min_new_tokens, temperature=args.temperature, seed=args.seed,
                                     verbose=True, force_recompute=args.force_recompute)
            if initial_settings is None:
                initial_settings = manager.settings.copy()
            elif manager.settings != initial_settings:
                raise ValueError('Generation settings differ between cached chunks.')
            chunks.append(chunk)
    finally:
        manager.stories = stories
    manager.results = pd.concat(chunks, ignore_index=True)
    manager._paired_results()
    manager.settings['generation_chunk_size'] = args.chunk_size
    manager.settings['evaluations'] = {'general': args.general_eval, 'paper': args.paper_eval}
    raw_dir = output_dir / 'raw' if args.general_eval else output_dir / 'paper_comparison' / 'raw'
    all_results = manager.results
    all_dataset_info = manager.dataset_info
    if args.general_eval:
        manager.stories = stories.iloc[:args.stories].copy()
        manager.results = all_results.loc[all_results['story_id'].isin(manager.stories['story_id'])].copy()
        manager.dataset_info = {**all_dataset_info, 'selected_stories': len(manager.stories)}
    manager.save_results(raw_dir)
    manager.stories, manager.results, manager.dataset_info = stories, all_results, all_dataset_info
    print(f'Saved raw generations to {raw_dir}', flush=True)

    manager.score_text_metrics(args.bert_model_dir, args.bert_num_layers, device=args.device,
                               batch_size=args.bert_batch_size, verbose=True)
    scored_results = manager.results
    if args.general_eval:
        general_stories = stories.iloc[:args.stories].copy()
        general_results = scored_results.loc[scored_results['story_id'].isin(general_stories['story_id'])].copy()
        general_dataset_info = {**all_dataset_info, 'selected_stories': len(general_stories)}
        manager.stories, manager.results, manager.dataset_info = general_stories, general_results, general_dataset_info
        manager.save_results(output_dir / 'scored')
    if args.paper_eval:
        manager.stories, manager.results, manager.dataset_info = stories, scored_results, all_dataset_info
        save_paper_comparison(manager, stories, output_dir / 'paper_comparison',
                              n_bootstrap=args.bootstrap_repetitions, confidence=args.confidence, seed=args.seed,
                              story_count=args.paper_stories)
    if not args.general_eval:
        print(f'Completed evaluation. Results saved to {output_dir}', flush=True)
        return
    manager.stories, manager.results, manager.dataset_info = general_stories, general_results, general_dataset_info
    manager.summary().to_csv(output_dir / 'summary.csv')
    story_metrics = manager.story_metrics()
    story_metrics.to_csv(output_dir / 'story_metrics.csv', index=False)
    metrics = list(story_metrics.columns[2:])
    unavailable = {}
    for metric in metrics:
        missing = int(story_metrics[metric].isna().sum())
        if missing:
            if metric != 'sft_per_token_logprob':
                raise ValueError(f'{metric} has {missing} missing story values; inspect the saved results.')
            unavailable[metric] = {'missing_story_model_values': missing,
                                   'reason': 'Per-token likelihood is undefined for empty completions.'}
        elif not np.isfinite(story_metrics[metric].to_numpy(dtype=float)).all():
            raise ValueError(f'{metric} contains non-finite story values; inspect the saved results.')
    if unavailable:
        print(f'Confidence intervals unavailable for undefined metrics: {unavailable}', flush=True)
        (output_dir / 'ci_unavailable.json').write_text(json.dumps(unavailable, indent=2) + '\n')
    metrics = [metric for metric in metrics if metric not in unavailable]
    intervals = manager.confidence_intervals(metrics=metrics, n_bootstrap=args.bootstrap_repetitions,
                                             confidence=args.confidence, seed=args.seed)
    intervals.to_csv(output_dir / 'confidence_intervals.csv', index=False)
    print(f'Completed evaluation. Results saved to {output_dir}', flush=True)


def paper_baseline(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f'Bundled DeBERTa-xlarge-MNLI BERTScore baseline is missing: {path}')
    baseline = pd.read_csv(path)
    if not {'LAYER', 'F'}.issubset(baseline.columns) or len(baseline) <= 40 or baseline.iloc[40]['LAYER'] != 40:
        raise ValueError(f'Invalid layer-40 BERTScore baseline: {path}')
    value = float(baseline.iloc[40]['F'])
    if not np.isfinite(value) or value >= 1:
        raise ValueError(f'Invalid BERTScore F1 baseline: {value}')
    return value


def validate_paper_model(config, num_layers: int):
    if (num_layers != 40 or config.model_type != 'deberta' or
            getattr(config, 'num_hidden_layers', None) != 48 or getattr(config, 'hidden_size', None) != 1024):
        raise ValueError('--paper-comparison requires DeBERTa-xlarge-MNLI-compatible weights at layer 40.')


def save_paper_comparison(manager: EvalManager, stories: pd.DataFrame, output_dir: Path,
                          n_bootstrap: int = 10000, confidence: float = 0.95, seed: int = 0,
                          story_count: int = 100):
    """Compare held-out stories with the paper's reported table, without claiming exact reproduction."""
    import copy
    import hashlib
    import bert_score
    from transformers import AutoConfig

    text_settings = manager.settings.get('text_metrics')
    if not text_settings or text_settings.get('rescale_with_baseline') is not False:
        raise ValueError('Paper comparison requires unrescaled text scores from score_text_metrics().')
    config = AutoConfig.from_pretrained(text_settings['model_dir'], local_files_only=True)
    validate_paper_model(config, text_settings['num_layers'])
    baseline_path = Path(bert_score.__file__).resolve().parent / 'rescale_baseline/en/microsoft/deberta-xlarge-mnli.tsv'
    baseline = paper_baseline(baseline_path)
    if isinstance(story_count, bool) or not isinstance(story_count, int) or story_count < 2:
        raise ValueError('The paper comparison requires at least two stories.')
    if len(stories) < story_count:
        raise ValueError(f'The paper comparison requires at least {story_count} selected stories.')
    selected = stories.iloc[:story_count].copy()
    results = manager._paired_results()
    results = results.loc[results['story_id'].isin(selected['story_id'])].copy()
    if results['story_id'].nunique() != story_count or not {'bleu', 'gleu', 'bertscore_f1'}.issubset(results.columns):
        raise ValueError(f'The paper comparison requires paired text scores for all {story_count} selected stories.')
    counts = results.groupby(['model', 'story_id']).size()
    if len(counts) != 2 * story_count or not counts.eq(manager.settings['samples_per_story']).all():
        raise ValueError('The paper comparison requires the configured number of samples for both models.')
    results['bertscore_f1_unrescaled'] = results['bertscore_f1']
    results['bertscore_f1'] = (results['bertscore_f1'] - baseline) / (1 - baseline)
    if not np.isfinite(results[['bleu', 'gleu', 'bertscore_f1']].to_numpy(dtype=float)).all():
        raise ValueError('Paper comparison text scores must be finite.')

    comparison = EvalManager(manager.sft_dir, manager.gfn_adapter_dir, device=manager.device, cache_dir=manager.cache_dir)
    comparison.stories = selected
    comparison.results = results
    comparison.dataset_info = copy.deepcopy(manager.dataset_info)
    comparison.dataset_info.update({'selected_stories': story_count, 'paper_comparison_selection':
                                    f'first {story_count} stories from the seeded held-out sample'})
    comparison.settings = copy.deepcopy(manager.settings)
    comparison.settings['text_metrics']['rescale_with_baseline'] = True
    comparison.settings['paper_comparison'] = {
        'paper': 'Hu et al., Amortizing Intractable Inference in Large Language Models, Table 2',
        'dataset_note': 'Adapted held-out validation stories, not the paper ROCStories test split.',
        'training_note': 'The SFT base used here was trained on the full supplied corpus, not the paper SFT setup.',
        'model_note': 'Config compatibility checked; exact DeBERTa-xlarge-MNLI weight provenance is not verified.',
        'metrics_note': 'BLEU-4 uses sentence BLEU method1; GLEU uses sentence GLEU. Both use full stripped text and are reported as percentages. BERTScore uses layer 40 and the bundled rescale baseline, matching the legacy scoring configuration but not its character slicing.',
        'bertscore_baseline_path': str(baseline_path),
        'bertscore_baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'bertscore_baseline_f1': baseline,
        'empty_generation_note': 'Empty strings have zero raw BLEU, GLEU and BERTScore; raw BERTScore is then rescaled.',
        'uncertainty_note': 'Paper plus/minus notation is transcribed as reported; its interpretation is not assumed. Our intervals are story-level percentile bootstraps.',
        'missing_baselines': 'Prompting and GPT4Eval are not run; no story text is sent to external services.',
    }
    comparison._paired_results()
    comparison.save_results(output_dir)
    story_values = comparison.story_metrics()
    our_means = story_values.groupby('model')[['bertscore_f1', 'bleu', 'gleu']].mean()
    references = [
        ('Prompting', None, 0.081, 0.009, 1.3, 0.5, 3.2, 0.1, 2.4),
        ('Supervised fine-tuning', 'SFT', 0.094, 0.007, 1.6, 0.8, 3.7, 0.4, 2.7),
        ('GFlowNet fine-tuning', 'GFN', 0.184, 0.004, 2.1, 0.2, 4.2, 0.7, 3.4),
    ]
    rows = []
    for label, model, bert, bert_pm, bleu, bleu_pm, gleu, gleu_pm, gpt4 in references:
        row = {'method': label, 'local_model': model, 'local_stories': None, 'local_outputs': None,
               'local_bertscore_f1_rescaled': None, 'local_bleu_4_percent': None, 'local_gleu_4_percent': None,
               'local_gpt4eval': None, 'paper_bertscore': bert, 'paper_bertscore_plus_minus': bert_pm,
               'paper_bleu_4': bleu, 'paper_bleu_4_plus_minus': bleu_pm, 'paper_gleu_4': gleu,
               'paper_gleu_4_plus_minus': gleu_pm, 'paper_gpt4eval': gpt4}
        if model is not None:
            subset = results.loc[results['model'] == model]
            row.update({'local_stories': subset['story_id'].nunique(), 'local_outputs': len(subset),
                        'local_bertscore_f1_rescaled': float(our_means.loc[model, 'bertscore_f1']),
                        'local_bleu_4_percent': 100 * float(our_means.loc[model, 'bleu']),
                        'local_gleu_4_percent': 100 * float(our_means.loc[model, 'gleu'])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / 'comparison.csv', index=False)
    intervals = comparison.confidence_intervals(metrics=['bertscore_f1', 'bleu', 'gleu'],
                                                 n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
    ngram_metrics = intervals['metric'].isin(['bleu', 'gleu'])
    intervals.loc[ngram_metrics, ['estimate', 'ci_low', 'ci_high']] *= 100
    intervals['metric'] = intervals['metric'].replace({'bertscore_f1': 'bertscore_f1_rescaled',
                                                         'bleu': 'bleu_4_percent', 'gleu': 'gleu_4_percent'})
    intervals.to_csv(output_dir / 'confidence_intervals.csv', index=False)
    print(f'Saved {story_count}-story paper comparison to {output_dir}. Prompting and GPT4Eval were not measured.', flush=True)


if __name__ == '__main__':
    run_evaluation(build_parser().parse_args())
