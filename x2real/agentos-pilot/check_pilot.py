"""Exercise data boundaries, geometry parity, action gates and episode isolation."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from prepare_training import relative_actions
from agentos_model import Model, validate_actions, state20, bound_predicted_grippers
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from wall_x._vendor.x2robot_utils.geometry import so3_to_euler_zyx_batch_nb


class PilotChecks(unittest.TestCase):
    def test_gripper_saturation_matches_simulator_without_changing_arm(self):
        state = np.zeros(14,np.float32)
        actions = np.zeros((33,14),np.float32)
        actions[1:,0] = .15
        actions[1:,6] = -.02
        actions[1:,13] = 1.03
        bounded = validate_actions(bound_predicted_grippers(actions),state)
        np.testing.assert_array_equal(bounded[0],state)
        np.testing.assert_array_equal(bounded[:,0:6],actions[:,0:6])
        np.testing.assert_array_equal(bounded[1:,6],0)
        np.testing.assert_array_equal(bounded[1:,13],1)
        self.assertLess(actions[1,6],0)
        actions[1,6]=np.nan
        with self.assertRaises(ValueError): validate_actions(bound_predicted_grippers(actions),state)

    def test_upstream_relative_rotation_parity(self):
        import yaml
        import pandas as pd
        from wall_x.data.backends.lerobot.loader import PreprocessedDataset
        cfg = yaml.safe_load(Path('/workspace/x2real-training/train.yaml').read_text())
        adapter = PreprocessedDataset.__new__(PreprocessedDataset)
        adapter._dof_config = cfg['task']['dof_config']
        table = pd.read_parquet('/workspace/x2real-training/data/bottle-v3/data/chunk-000/file-000.parquet')
        frames = table[table.episode_index==0]
        state = np.stack(frames['observation.state'])[20]
        actions = np.stack(frames['action'])[20:52]
        ours = relative_actions(actions[None],state[None])[0]
        theirs = adapter._to_relative_action(actions,state)
        np.testing.assert_allclose(ours,theirs,atol=1e-5)

    def test_euler_roundtrip_and_relative_rotation(self):
        state = np.array([.1,.2,.3,.2,-.1,.3,.4, -.1,.2,.4,-.3,.2,.1,.8], dtype=np.float32)
        encoded = state20(state)
        decoded = np.concatenate([np.r_[encoded[o:o+3], so3_to_euler_zyx_batch_nb(encoded[None,o+3:o+9])[0], encoded[o+9]] for o in (0,10)])
        np.testing.assert_allclose(decoded, state, atol=1e-6)
        delta = relative_actions(np.repeat(encoded[None,None],32,axis=1), encoded[None])[0]
        for offset in (0,10):
            np.testing.assert_allclose(delta[:,offset:offset+3], 0, atol=1e-6)
            np.testing.assert_allclose(delta[:,offset+3:offset+9], np.tile([1,0,0,0,1,0],(32,1)), atol=1e-6)
            np.testing.assert_allclose(delta[:,offset+9], encoded[offset+9])

    def test_action_gate_rejects_invalid_output(self):
        state = np.zeros(14, np.float32)
        valid = np.zeros((33,14), np.float32)
        validate_actions(valid, state)
        for index, value in ((1,float('nan')), (6,1.1)):
            bad = valid.copy(); bad[1,index] = value
            with self.assertRaises(ValueError): validate_actions(bad,state)
        bad = valid.copy(); bad[0,0] = .1
        with self.assertRaises(ValueError): validate_actions(bad,state)

    def test_episode_memory_clears(self):
        model = Model.__new__(Model)
        model._clear()
        model.memory.append({'event':'old'})
        model.pending = (48,np.ones(1556))
        model.cached = np.ones((33,14)); model.fallback_count=2
        model._clear()
        self.assertFalse(model.memory)
        self.assertIsNone(model.pending)
        self.assertIsNone(model.cached)
        self.assertEqual(model.fallback_count,0)

    def test_bounded_fallback_and_duplicate_step(self):
        class InvalidBase:
            calls=0
            def ingest_observation(self, obs): pass
            def infer_actions(self):
                self.calls+=1
                return np.full((33,14), np.nan)
        model=Model.__new__(Model)
        model.base=InvalidBase()
        model._clear()
        with tempfile.TemporaryDirectory() as directory:
            model.audit_path=Path(directory)/'events.jsonl'
            obs={'state':{'follow1_pos':np.zeros(7),'follow2_pos':np.zeros(7)},'step':0}
            model.ingest_observation(obs)
            first=model.infer_actions()
            np.testing.assert_array_equal(first,model.infer_actions())
            self.assertEqual(model.base.calls,1)
            model.ingest_observation({**obs,'step':48})
            model.infer_actions()
            model.ingest_observation({**obs,'step':96})
            with self.assertRaises(RuntimeError): model.infer_actions()

    def test_lerobot_action_horizon_stays_inside_episode(self):
        d = LeRobotDataset('x2real/bottle-v3', root='/workspace/x2real-training/data/bottle-v3',
                           episodes=[0,1], video_backend='pyav',
                           delta_timestamps={'action':[i/20 for i in range(32)]})
        last = int(d.meta.episodes[0]['length'])-1
        frame = d[last]
        self.assertEqual(tuple(frame['action'].shape),(32,20))
        np.testing.assert_allclose(frame['action'].numpy(),np.repeat(frame['action'][0:1].numpy(),32,axis=0))


if __name__ == '__main__':
    unittest.main()
