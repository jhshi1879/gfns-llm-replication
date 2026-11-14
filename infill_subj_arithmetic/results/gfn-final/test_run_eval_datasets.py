import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DatasetLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / 'project'
        self.script_dir = self.project / 'results' / 'gfn-final'
        self.script_dir.mkdir(parents=True)
        self.data = self.project / 'data'
        self.data.mkdir()
        shutil.copy2(Path(__file__).with_name('run_eval.sh'), self.script_dir)
        (self.script_dir / 'eval.py').write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['ARGS_FILE']).write_text(json.dumps(sys.argv[1:]))\n"
        )
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        curl = self.bin / 'curl'
        curl.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\n" "$@" >> "$CURL_LOG"\n'
            'output=""\n'
            'while (($#)); do\n'
            '  if [[ "$1" == "--output" ]]; then output="$2"; shift 2; else shift; fi\n'
            'done\n'
            'cp "$CURL_FIXTURE" "$output"\n'
        )
        curl.chmod(0o755)
        ssh = self.bin / 'ssh'
        ssh.write_text(
            '#!/usr/bin/env bash\n'
            'if [[ "$*" == *--current-pod-id* ]]; then\n'
            '  printf "%s\\n" "test-pod"\n'
            '  exit 0\n'
            'fi\n'
            'if [[ "$*" == *--receive-current* ]]; then\n'
            '  cat > "$SSH_CREDENTIAL_INPUT"\n'
            '  printf "%s\\n" "/tmp/gfn-runpod-fixture"\n'
            '  exit 0\n'
            'fi\n'
            'printf "%s\\n" "$@" > "$SSH_LOG"\n'
            'cat > "$SSH_INPUT"\n'
            'exit "${SSH_EXIT:-0}"\n'
        )
        ssh.chmod(0o755)
        self.args_file = self.root / 'args.json'
        self.curl_log = self.root / 'curl.log'
        self.ssh_log = self.root / 'ssh.log'
        self.ssh_input = self.root / 'ssh-input'
        self.ssh_credentials = self.root / 'ssh-credentials'
        self.fixture = self.root / 'fixture.csv'
        self.fixture.write_text(
            'sentence1,sentence2,sentence3,sentence4,sentence5\n'
            'one,two,three,four,five\n'
        )
        self.env = os.environ.copy()
        for name in ('GFN_ADAPTER_DIR', 'EVAL_BERT_MODEL_DIR', 'EVAL_BERT_NUM_LAYERS'):
            self.env.pop(name, None)
        for name in ('RUNPOD_API_KEY', 'RUNPOD_POD_ID', 'RUNPOD_CREDENTIALS_FILE',
                     'STOP_RUNPOD_ON_EXIT'):
            self.env.pop(name, None)
        self.env.update({
            'EVAL_PYTHON': sys.executable,
            'PATH': str(self.bin) + os.pathsep + self.env['PATH'],
            'ARGS_FILE': str(self.args_file),
            'CURL_LOG': str(self.curl_log),
            'CURL_FIXTURE': str(self.fixture),
            'SSH_LOG': str(self.ssh_log),
            'SSH_INPUT': str(self.ssh_input),
            'SSH_CREDENTIAL_INPUT': str(self.ssh_credentials),
        })

    def tearDown(self):
        self.temporary.cleanup()

    def run_launcher(self, *args):
        return subprocess.run(
            ['bash', str(self.script_dir / 'run_eval.sh'), *args],
            env=self.env, text=True, capture_output=True,
        )

    def test_reuses_default_validation_and_existing_training_stories(self):
        validation = self.data / 'story-cloze-winter2018-val.csv'
        training = self.data / 'stories.csv'
        validation.write_text('existing validation')
        training.write_text('existing training')
        (self.data / 'rocstories-eval-winter2017.csv').write_text('other training file')
        result = self.run_launcher('--check-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.args_file.read_text()), [
            '--check-only', '--validation-csv', str(validation), '--training-csv', str(training),
        ])
        self.assertFalse(self.curl_log.exists())

    def test_downloads_and_validates_missing_defaults(self):
        result = self.run_launcher('--check-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ('story-cloze-winter2018-val.csv', 'rocstories-eval-winter2017.csv'):
            self.assertEqual((self.data / name).read_text(), self.fixture.read_text())
        curl_args = self.curl_log.read_text()
        self.assertIn('https://goo.gl/XWjas1', curl_args)
        self.assertIn('https://goo.gl/0OYkPK', curl_args)
        self.assertEqual(curl_args.count('--proto-redir\n=https\n'), 2)

    def test_story_cloze_download_rejects_invalid_labels(self):
        self.fixture.write_text(
            'InputSentence1,InputSentence2,InputSentence3,InputSentence4,'
            'RandomFifthSentenceQuiz1,RandomFifthSentenceQuiz2,AnswerRightEnding\n'
            'one,two,three,four,five,six,3\n'
        )
        result = self.run_launcher('--check-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid label', result.stderr)
        self.assertEqual(list(self.data.iterdir()), [])

    def test_detects_pod_models_and_respects_explicit_overrides(self):
        adapter = self.project / 'output' / 'stories_gfn' / 'final_adapter'
        bert = self.project / 'models' / 'deberta-xlarge-mnli'
        adapter.mkdir(parents=True)
        bert.mkdir(parents=True)
        (adapter / 'adapter_config.json').touch()
        (bert / 'config.json').touch()
        paths = ('--validation-csv', 'validation', '--training-csv', 'training')
        result = self.run_launcher('--check-only', *paths)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.args_file.read_text()), [
            '--check-only', *paths, '--adapter-dir', str(adapter),
            '--bert-model-dir', str(bert), '--bert-num-layers', '40',
        ])
        result = self.run_launcher(
            '--check-only', *paths, '--adapter-dir=custom-adapter',
            '--bert-model-dir', 'custom-model', '--bert-num-layers', '12',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.args_file.read_text()), [
            '--check-only', *paths, '--adapter-dir=custom-adapter',
            '--bert-model-dir', 'custom-model', '--bert-num-layers', '12',
        ])
        self.env.update({'GFN_ADAPTER_DIR': 'env-adapter', 'EVAL_BERT_MODEL_DIR': 'env-model'})
        result = self.run_launcher('--check-only', *paths)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.args_file.read_text()), ['--check-only', *paths])

    def test_explicit_dataset_paths_and_help_do_not_download(self):
        result = self.run_launcher(
            '--check-only', '--validation-csv=custom-validation', '--training-csv', 'custom-training',
            '--general-stories', '4', '--paper-stories', '7',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.args_file.read_text()), [
            '--check-only', '--validation-csv=custom-validation', '--training-csv', 'custom-training',
            '--general-stories', '4', '--paper-stories', '7',
        ])
        result = self.run_launcher('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.curl_log.exists())

    def test_invalid_download_fails_without_installing_file(self):
        self.fixture.write_text('<html>not a dataset</html>\n')
        result = self.run_launcher('--check-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing CSV columns', result.stderr)
        self.assertFalse((self.data / 'story-cloze-winter2018-val.csv').exists())
        self.assertEqual(list(self.data.iterdir()), [])
        self.assertFalse(self.args_file.exists())

    def test_ssh_forwards_flags_as_data_without_running_local_evaluation(self):
        suspicious = "name with spaces; $(touch /tmp/not-executed)"
        result = self.run_launcher(
            '--general-stories', '2', '--ssh-host', 'user@pod.example',
            '--paper-stories', '2', '--ssh-port', '17020', '--output-dir', suspicious,
            '--no-stop-pod-on-exit',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        ssh_args = self.ssh_log.read_text().splitlines()
        self.assertEqual(ssh_args[:3], ['-p', '17020', 'user@pod.example'])
        self.assertNotIn(suspicious, ssh_args[3])
        self.assertEqual(self.ssh_input.read_bytes().split(b'\0'), [
            b'launch', b'--no-stop-pod-on-exit', b'--general-stories', b'2', b'--paper-stories', b'2',
            b'--output-dir', suspicious.encode(), b'',
        ])
        self.assertFalse(self.args_file.exists())
        self.assertFalse(self.curl_log.exists())

    def test_ssh_requires_valid_connection_and_propagates_errors(self):
        for flags in [('--ssh-port', '22'), ('--ssh-host=',),
                      ('--ssh-host', '-invalid'), ('--ssh-host', 'pod', '--ssh-port', '0')]:
            with self.subTest(flags=flags):
                result = self.run_launcher(*flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Error:', result.stderr)
                self.assertFalse(self.ssh_log.exists())
        self.env['SSH_EXIT'] = '17'
        result = self.run_launcher('--ssh-host=pod', '--no-stop-pod-on-exit')
        self.assertEqual(result.returncode, 17)
        self.assertEqual(self.ssh_log.read_text().splitlines()[:3], ['-p', '22', 'pod'])
        self.assertEqual(self.ssh_input.read_bytes(), b'launch\0--no-stop-pod-on-exit\0')

    def test_ssh_shutdown_is_default_and_credentials_are_not_in_command(self):
        (self.project / 'runpod_util.py').write_text(
            "import json\n"
            "print(json.dumps(['test-pod', 'dummy-test-key']))\n"
        )
        result = self.run_launcher('--ssh-host', 'pod', '--general-stories', '2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.ssh_credentials.read_text()), ['test-pod', 'dummy-test-key'])
        self.assertNotIn('dummy-test-key', self.ssh_log.read_text() + result.stdout + result.stderr)
        self.assertEqual(self.ssh_input.read_bytes().split(b'\0'), [
            b'launch', b'--shutdown-credentials-file', b'/tmp/gfn-runpod-fixture',
            b'--general-stories', b'2', b'',
        ])

    def test_ssh_check_only_does_not_prepare_shutdown(self):
        result = self.run_launcher('--ssh-host', 'pod', '--check-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.ssh_credentials.exists())
        self.assertEqual(self.ssh_input.read_bytes(), b'launch\0--check-only\0')


if __name__ == '__main__':
    unittest.main()
