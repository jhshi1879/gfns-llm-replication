import json
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("shutdown_helper", Path(__file__).with_name("runpod_util.py"))
shutdown_helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shutdown_helper)


class PodShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runpod-shutdown-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.helper = self.root / "runpod_util.py"
        shutil.copy2(Path(__file__).with_name("runpod_util.py"), self.helper)
        self.pod_environ = self.root / "pid1-environ"
        self.pod_environ.write_bytes(b"RUNPOD_POD_ID=test-pod\0")
        self.helper.write_text(self.helper.read_text().replace('"/proc/1/environ"', repr(str(self.pod_environ))))
        sdk = self.root / ".venv-runpod-sdk/bin"
        sdk.mkdir(parents=True)
        (sdk / "python").symlink_to(sys.executable)
        (self.root / "runpod.py").write_text(
            "import os\n"
            "from pathlib import Path\n"
            "api_key = None\n"
            "def stop_pod(pod_id):\n"
            "    assert api_key == 'dummy-test-key'\n"
            "    if os.environ.get('SDK_FAIL'):\n"
            "        raise RuntimeError('dummy-test-key must not reach logs')\n"
            "    if os.environ.get('TEST_FINISHED_DIR'):\n"
            "        runs = list(Path(os.environ['TEST_FINISHED_DIR']).glob('run-*/exit-status.txt'))\n"
            "        assert len(runs) == 1 and runs[0].read_text().strip() == os.environ['TEST_EXPECTED_STATUS']\n"
            "        assert not Path(os.environ['RUNPOD_CREDENTIALS_FILE']).exists()\n"
            "    Path(os.environ['TEST_STOP_FILE']).write_text(pod_id)\n"
        )
        self.stopped = self.root / "stopped"
        self.env = dict(os.environ, PYTHONPATH=str(self.root),
                        RUNPOD_ENV_FILE=str(self.root / "absent.env"),
                        TEST_STOP_FILE=str(self.stopped))
        for name in ("RUNPOD_API_KEY", "RUNPOD_POD_ID", "STOP_RUNPOD_ON_EXIT",
                     "RUNPOD_CREDENTIALS_FILE", "RANK"):
            self.env.pop(name, None)

    def receive(self):
        result = subprocess.run(
            [sys.executable, str(self.helper), "--receive"],
            input=json.dumps(["test-pod", "dummy-test-key"]), text=True,
            capture_output=True, env=self.env, check=True,
        )
        path = Path(result.stdout.strip())
        self.addCleanup(path.unlink, missing_ok=True)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        return path

    def run_training(self, code="", **overrides):
        credentials = self.receive()
        env = dict(self.env, STOP_RUNPOD_ON_EXIT="1",
                   RUNPOD_CREDENTIALS_FILE=str(credentials), **overrides)
        result = subprocess.run(
            [sys.executable, "-c",
             "from runpod_util import register_pod_shutdown; register_pod_shutdown(); " + code],
            env=env, cwd=self.root, text=True, capture_output=True, timeout=15,
        )
        self.assertNotIn("dummy-test-key", result.stdout + result.stderr)
        return result, credentials

    def test_success_stops_and_removes_credentials(self):
        result, credentials = self.run_training("print('training complete')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.stopped.read_text(), "test-pod")
        self.assertFalse(credentials.exists())
        self.assertIn("accepted the stop request", result.stdout)

    def test_unhandled_training_exception_still_stops(self):
        result, credentials = self.run_training("raise RuntimeError('training failed')")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.stopped.read_text(), "test-pod")
        self.assertFalse(credentials.exists())

    def test_supervisor_stops_after_training_process_exits(self):
        cases = (
            ("success", "print('training complete')", 0),
            ("exception", "raise RuntimeError('training failed')", 1),
            ("killed", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)", 128 + signal.SIGKILL),
        )
        for name, code, expected_status in cases:
            with self.subTest(name=name):
                self.stopped.unlink(missing_ok=True)
                credentials = self.receive()
                env = dict(self.env, STOP_RUNPOD_ON_EXIT="1",
                           RUNPOD_CREDENTIALS_FILE=str(credentials))
                result = subprocess.run(
                    [sys.executable, str(self.helper), "--supervise",
                     sys.executable, "-c", code],
                    env=env, cwd=self.root, text=True, capture_output=True, timeout=15,
                )
                self.assertEqual(result.returncode, expected_status, result.stderr)
                self.assertEqual(self.stopped.read_text(), "test-pod")
                self.assertFalse(credentials.exists())
                self.assertNotIn("dummy-test-key", result.stdout + result.stderr)

    def test_gfn_worker_supervises_a_killed_training_process(self):
        shutil.copy2(Path(__file__).with_name("run_gfn.sh"), self.root / "run_gfn.sh")
        for name in ("data.py", "rewards.py", "replay_buffer.py", "utils.py", "requirements.txt"):
            (self.root / name).write_text("")
        (self.root / "main.py").write_text(
            "import os, signal\n"
            "assert os.environ['STOP_RUNPOD_ON_EXIT'] == '0'\n"
            "os.kill(os.getpid(), signal.SIGKILL)\n"
        )
        (self.root / "configs").mkdir()
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "pytorch_model.bin").touch()
        credentials = self.receive()
        result = subprocess.run(
            ["bash", str(self.root / "run_gfn.sh"), "--worker",
             sys.executable, str(credentials)],
            env=dict(self.env, GFN_BASE_MODEL=str(model)), cwd=self.root,
            stdin=subprocess.DEVNULL, text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "output/stories_gfn/exit-status.txt").read_text().strip(),
                         str(128 + signal.SIGKILL))
        self.assertEqual(self.stopped.read_text(), "test-pod")
        self.assertFalse(credentials.exists())

    def test_sdk_failure_is_reported_without_key(self):
        result, _ = self.run_training("print('training complete')", SDK_FAIL="1")
        self.assertIn("RunPod SDK stop failed", result.stderr)
        self.assertFalse(self.stopped.exists())

    def test_mismatched_pod_is_not_stopped(self):
        result, credentials = self.run_training(RUNPOD_POD_ID="different-pod")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match", result.stderr)
        self.assertFalse(self.stopped.exists())
        self.assertFalse(credentials.exists())

    def test_current_pod_identity_is_verified(self):
        with patch.object(shutdown_helper.Path, 'read_bytes', return_value=b'RUNPOD_POD_ID=actual-pod\0'):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(shutdown_helper.current_runpod_id(), 'actual-pod')
            with patch.dict(os.environ, {'RUNPOD_POD_ID': 'other-pod'}, clear=True):
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    shutdown_helper.current_runpod_id()
        with patch.object(shutdown_helper.Path, 'read_bytes', return_value=b'OTHER=value\0'):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, 'Cannot verify'):
                    shutdown_helper.current_runpod_id()

    def test_default_credentials_file_is_at_repository_root(self):
        project = self.root / "infill_subj_arithmetic"
        project.mkdir()
        (self.root / ".env").write_text("RUNPOD_POD_ID=test-pod\nRUNPOD_API_KEY=test-key\n")
        with patch.object(shutdown_helper, "__file__", str(project / "runpod_util.py")):
            with patch.dict(os.environ, {}, clear=True):
                shutdown_helper.load_runpod_env()
                self.assertEqual(os.environ["RUNPOD_POD_ID"], "test-pod")
                self.assertEqual(os.environ["RUNPOD_API_KEY"], "test-key")

    def test_receive_current_rejects_wrong_pod_before_creating_credentials(self):
        result = subprocess.run(
            [sys.executable, str(self.helper), '--receive-current'],
            input=json.dumps(['different-pod', 'dummy-test-key']), text=True,
            capture_output=True, env=self.env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not match', result.stderr)
        self.assertNotIn('dummy-test-key', result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), '')

    def test_credentials_for_pod_rejects_configured_mismatch(self):
        result = subprocess.run(
            [sys.executable, str(self.helper), '--credentials-for-pod', 'test-pod'],
            env=dict(self.env, RUNPOD_POD_ID='other-pod', RUNPOD_API_KEY='dummy-test-key'),
            text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not match', result.stderr)
        self.assertNotIn('dummy-test-key', result.stdout + result.stderr)

    def test_evaluation_shutdown_runs_after_status_is_saved(self):
        script_dir = self.root / 'results' / 'gfn-final'
        script_dir.mkdir(parents=True)
        shutil.copy2(Path(__file__).parent / 'results/gfn-final/run_eval.sh', script_dir)
        (script_dir / 'eval.py').write_text(
            "import os, sys\n"
            "print('fake evaluation')\n"
            "if '--check-only' in sys.argv and os.environ.get('TEST_PREFLIGHT_FAIL'):\n"
            "    sys.exit(1)\n"
            "if '--check-only' not in sys.argv:\n"
            "    sys.exit(int(os.environ['TEST_EXPECTED_STATUS']))\n"
        )
        data = self.root / 'data'
        data.mkdir()
        (data / 'story-cloze-winter2018-val.csv').touch()
        (data / 'stories.csv').touch()
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        tmux = bin_dir / 'tmux'
        tmux.write_text(
            '#!/usr/bin/env bash\n'
            'if [[ "$1" == has-session ]]; then [[ -f "$TEST_TMUX_STATE" ]]; exit; fi\n'
            'if [[ "$1" == new-session ]]; then\n'
            '  for arg in "$@"; do worker="$arg"; done\n'
            '  bash -c "$worker" || true\n'
            '  touch "$TEST_TMUX_STATE"\n'
            '  exit 0\n'
            'fi\n'
            'exit 1\n'
        )
        tmux.chmod(0o755)
        state = self.root / 'tmux-state'
        evaluations = script_dir / 'evaluations'
        for expected in (0, 1):
            with self.subTest(expected=expected):
                self.stopped.unlink(missing_ok=True)
                state.unlink(missing_ok=True)
                env = dict(self.env, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                           EVAL_PYTHON=sys.executable, RUNPOD_POD_ID='test-pod',
                           RUNPOD_API_KEY='dummy-test-key', TEST_TMUX_STATE=str(state),
                           TEST_FINISHED_DIR=str(evaluations), TEST_EXPECTED_STATUS=str(expected))
                result = subprocess.run(
                    ['bash', str(script_dir / 'run_eval.sh')], env=env,
                    text=True, capture_output=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                run_dir = Path((evaluations / 'active-run.txt').read_text().strip())
                self.assertEqual((run_dir / 'exit-status.txt').read_text().strip(), str(expected))
                self.assertEqual(self.stopped.read_text(), 'test-pod')
                self.assertNotIn('dummy-test-key', result.stdout + result.stderr)
                shutil.rmtree(run_dir)
        self.stopped.unlink(missing_ok=True)
        state.unlink(missing_ok=True)
        env = dict(self.env, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                   EVAL_PYTHON=sys.executable, TEST_TMUX_STATE=str(state), TEST_EXPECTED_STATUS='0')
        result = subprocess.run(
            ['bash', str(script_dir / 'run_eval.sh'), '--no-stop-pod-on-exit'],
            env=env, text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.stopped.exists())
        state.unlink(missing_ok=True)
        result = subprocess.run(
            ['bash', str(script_dir / 'run_eval.sh')],
            env=env, text=True, capture_output=True, timeout=15,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('RUNPOD_API_KEY', result.stderr)
        self.assertFalse(state.exists())
        self.assertFalse(self.stopped.exists())
        credentials = self.receive()
        result = subprocess.run(
            ['bash', str(script_dir / 'run_eval.sh'), '--shutdown-credentials-file', str(credentials)],
            env=dict(env, TEST_PREFLIGHT_FAIL='1'), text=True, capture_output=True, timeout=15,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(credentials.exists())
        self.assertFalse(self.stopped.exists())

    def test_disabled_callback_does_not_stop(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "from runpod_util import register_pod_shutdown; register_pod_shutdown()"],
            env=self.env, cwd=self.root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.stopped.exists())


if __name__ == "__main__":
    unittest.main()
