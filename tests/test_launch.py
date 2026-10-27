"""Verify every launcher without allocating GPUs or starting a trainer."""
import json
import os
from pathlib import Path
import subprocess  # nosec B404 -- local launchers and a test stub only
import sys
import tempfile
import unittest

import yaml


class LauncherTests(unittest.TestCase):
    def test_stage_configs_and_separate_arguments(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp)/'capture.json'
            fake = Path(tmp)/'accelerate'
            fake.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                            'json.dump({"argv":sys.argv[1:],"cwd":os.getcwd()},'
                            'open(os.environ["CAPTURE"],"w"))\n')
            fake.chmod(0o700)
            env = dict(os.environ, PATH=tmp+os.pathsep+os.environ['PATH'], CAPTURE=str(capture))
            for stage in ('tokenizer','alignment','pt','sft'):
                config = 'tokenizer_g16' if stage == 'tokenizer' else stage
                cfg = yaml.safe_load((root/'configs'/f'{config}.yaml').read_text())
                self.assertTrue(all(cfg.get(key) for key in ('checkpoint','data','output')))
                if stage != 'tokenizer':
                    self.assertEqual(cfg['train_llm'], stage != 'alignment')
                    self.assertEqual(cfg['data_mode'], 'sft' if stage == 'sft' else 'pretrain')
                command = ['bash', str(root/'scripts'/f'train_{stage}.sh'),
                           '--data', 'data with spaces.jsonl', '--output', 'output with spaces']
                subprocess.run(command, cwd=tmp, env=env, check=True)  # nosec B603 B607 -- fixed local test script
                actual = json.loads(capture.read_text())
                self.assertEqual(actual['cwd'], str(root))
                distributed = 'accelerate_ddp' if stage == 'tokenizer' else 'accelerate_zero2'
                trainer = 'train_tokenizer' if stage == 'tokenizer' else 'train_vlm'
                self.assertEqual(actual['argv'], ['launch','--config_file',f'configs/{distributed}.yaml',
                    f'training/{trainer}.py','--config',f'configs/{config}.yaml',
                    '--data','data with spaces.jsonl','--output','output with spaces'])
            env['ACCELERATE_CONFIG'] = 'my launch config.yaml'
            subprocess.run(['bash',str(root/'scripts/train_sft.sh')], cwd=tmp, env=env, check=True)  # nosec B603 B607
            self.assertEqual(json.loads(capture.read_text())['argv'][2], 'my launch config.yaml')


if __name__ == '__main__':
    unittest.main()
