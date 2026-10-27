import copy
import tempfile
import unittest
from pathlib import Path
import sys

import torch
from PIL import Image
from fvq.quantizer import FactorizedVQ
from fvq.processing import ImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"training"))
from common import ModelEMA, scheduler, training_paths
from data import JSONLData
from fvq.feature_decoder import FeatureDecoder
from fvq import FVQTokenizer, FVQVLM
from transformers import Qwen3Config, Qwen3ForCausalLM


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def make_quantizer(self):
        return FactorizedVQ(dim=8, groups=4, group_dim=4, size=16, output_dim=8, restart=False)

    def test_frame_and_tie(self):
        q = self.make_quantizer()
        frame = q.R.flatten(0, 1)
        torch.testing.assert_close(frame.T@frame, 2*torch.eye(8), atol=1e-6, rtol=1e-6)
        self.assertEqual(q.nearest(torch.zeros(1,4), torch.zeros(16,4)).item(), 0)

    def test_eval_read_only_and_precision(self):
        q = self.make_quantizer().eval()
        before = copy.deepcopy(q.state_dict())
        q(torch.randn(9,8))
        for key, value in q.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        q.to(torch.bfloat16)
        self.assertTrue(torch.equal(q.R, before['R']))
        for cb in q.codebooks:
            self.assertEqual(cb.embedding.weight.dtype, torch.float32)
            self.assertEqual(cb.embed_ema.dtype, torch.float32)

    def test_training_and_save_resume(self):
        q = self.make_quantizer()
        optimizer = torch.optim.AdamW([p for p in q.parameters() if p.requires_grad], lr=1e-4)
        ema = ModelEMA(q)
        x = torch.randn(9,8)
        out, result = q(x)
        (out.square().mean()+result['quantizer_loss']).backward()
        optimizer.step()
        optimizer.zero_grad()
        ema.update(q)
        clone = self.make_quantizer()
        clone.load_state_dict(q.state_dict(), strict=True)
        optimizer2 = torch.optim.AdamW([p for p in clone.parameters() if p.requires_grad], lr=1e-4)
        optimizer2.load_state_dict(optimizer.state_dict())
        out1, result1 = q(x)
        out2, result2 = clone(x)
        self.assertTrue(torch.equal(result1['ids'], result2['ids']))
        self.assertTrue(torch.equal(out1, out2))
        for key,value in q.state_dict().items():
            self.assertTrue(torch.equal(value, clone.state_dict()[key]), key)

    def test_decoder_backward_and_image_grouping(self):
        decoder = FeatureDecoder(dim=32, output_dim=8, depth=1, heads=4, base_grid=2)
        x = torch.randn(12,32,requires_grad=True)
        y = decoder(x, torch.tensor([[1,2,2],[1,2,4]]))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        batch = ImageProcessor(min_pixels=1024,max_pixels=1024)([Image.new('RGB',(32,32),'white')])
        self.assertEqual(tuple(batch['pixel_values'].shape), (4,1536))
        self.assertTrue(torch.equal(batch['grid_thw'], torch.tensor([[1,2,2]])))
        self.assertTrue((batch['pixel_values']==1).all())

    def test_tiny_vlm_training_and_reload(self):
        config = {'vision': {'depth':1, 'hidden_size':32, 'intermediate_size':64,
                  'num_heads':4, 'patch_size':16, 'temporal_patch_size':2,
                  'spatial_merge_size':2, 'num_position_embeddings':16,
                  'out_hidden_size':32, 'deepstack_visual_indexes':[]},
                  'quantizer': {'groups':4,'group_dim':16,'size':16,'output_dim':None}, 'decoder':None}
        tokenizer = FVQTokenizer(config)
        llm = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64,
                 num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                 head_dim=8, vocab_size=64, max_position_embeddings=128,
                 eos_token_id=2, pad_token_id=0))
        # Tiny test vocabulary ID, not a credential.
        model = FVQVLM(tokenizer, llm, {'image_pad_token_id':1})  # nosec B105
        batch = ImageProcessor(min_pixels=1024,max_pixels=1024)([Image.new('RGB',(32,32),'white')])
        ids = torch.tensor([[4,1,5,6]])
        inputs = {'input_ids':ids, **batch}
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        labels=ids.clone();labels[labels==1]=-100
        result=model(**inputs,labels=labels)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(all(p.grad is None for p in model.tokenizer.parameters()))
        self.assertIsNotNone(model.embeddings[0].weight.grad)
        optimizer.step()
        model.eval()
        before=model(**inputs,use_cache=False).logits.detach()
        generated=model.generate(**inputs,max_new_tokens=4,do_sample=False)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            restored=FVQVLM.from_pretrained(tmp)
            after=restored(**inputs,use_cache=False).logits.detach()
            self.assertTrue(torch.equal(before,after))
            self.assertTrue(torch.equal(generated,restored.generate(**inputs,max_new_tokens=4,do_sample=False)))
        # Packed SFT is isolated, including when checkpointing recomputes attention.
        packed_ids = torch.tensor([[4,5,6,7,8,9]])
        positions = torch.tensor([[0,1,2,0,1,2]])
        model.llm.config.use_cache = False
        model.eval()
        with torch.no_grad():
            packed = model(input_ids=packed_ids, position_ids=positions, sequence_lengths=(3,3), use_cache=False).logits
            separate = model(input_ids=packed_ids[:,3:], use_cache=False).logits
            torch.testing.assert_close(packed[:,3:], separate, atol=1e-6, rtol=1e-6)
            changed = packed_ids.clone(); changed[:,:3] = 11
            new = model(input_ids=changed, position_ids=positions, sequence_lengths=(3,3), use_cache=False).logits
            self.assertTrue(torch.equal(packed[:,3:], new[:,3:]))
        model.train()
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        result = model(input_ids=packed_ids, labels=packed_ids, position_ids=positions,
                       sequence_lengths=(3,3), use_cache=False)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))

    def test_jsonl_sharding_and_invalid_record(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.jsonl'
            path.write_text('\n'.join(json.dumps({'images':['a.png'],'index':i}) for i in range(8)))
            rows = [list(JSONLData(path,rank=r,world=2)) for r in range(2)]
            self.assertEqual(sorted(x['index'] for group in rows for x in group), list(range(8)))
            self.assertTrue(all(x['images']==[str(Path(tmp)/'a.png')] for group in rows for x in group))
            path.write_text('{bad json}\n')
            with self.assertRaisesRegex(ValueError,'line 1'):
                list(JSONLData(path))

    def test_scheduler(self):
        p=torch.nn.Parameter(torch.ones(1))
        optimizer=torch.optim.AdamW([p],lr=1.)
        schedule=scheduler(optimizer,{'lr':1.,'min_lr':0.1,'steps':4,'warmup_steps':2})
        self.assertEqual(schedule.get_last_lr(),[0.])
        optimizer.step();schedule.step()
        self.assertEqual(schedule.get_last_lr(),[0.5])
        optimizer.step();schedule.step()
        self.assertEqual(schedule.get_last_lr(),[1.])

    def test_training_paths_and_legacy_cli(self):
        import argparse
        import contextlib
        import io
        parser = argparse.ArgumentParser()
        cfg = {'checkpoint':'weights', 'data':'data.jsonl', 'output':'outputs', 'cache':'cache/stage'}
        args = argparse.Namespace(checkpoint=None, data=None, output=None, cache=None)
        training_paths(parser, args, cfg)
        self.assertEqual(vars(args), cfg)
        args = argparse.Namespace(checkpoint='other weights', data=None, output=None, cache='other cache')
        training_paths(parser, args, cfg)
        self.assertEqual(args.checkpoint, 'other weights')
        self.assertEqual(args.cache, 'other cache')
        self.assertEqual(args.data, 'data.jsonl')
        legacy = argparse.Namespace(checkpoint='weights', data='data.jsonl', output='outputs', cache=None)
        training_paths(parser, legacy, {})
        self.assertEqual(legacy.cache, 'cache')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            training_paths(parser, argparse.Namespace(checkpoint=None, data=None, output=None), {})


if __name__ == '__main__':
    unittest.main()
