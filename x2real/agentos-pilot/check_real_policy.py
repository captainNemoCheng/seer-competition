"""One real policy inference on a held-out HF observation (not a task rollout)."""
import json
import sys
from pathlib import Path
import time
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from policy_space.runtime.server import load_policy_model
from wall_x._vendor.x2robot_utils.geometry import so3_to_euler_zyx_batch_nb

ROOT=Path('/workspace/x2real-training')


def observation():
    cached=ROOT/'outputs/heldout-observation.npz'
    if cached.exists():
        sample=np.load(cached)
        return {'images':{k:sample[k] for k in ('camera_front','camera_left','camera_right')},
                'state':{k:sample[k] for k in ('follow1_pos','follow2_pos')},
                'instruction':str(sample['instruction']),'step':0,'episode_id':'heldout-interface-check'}
    data=LeRobotDataset('x2real/bottle-v3',root=str(ROOT/'data/bottle-v3'),episodes=[270],video_backend='pyav')
    sample=data[0]
    vector=sample['observation.state'].numpy()
    state={key:np.r_[vector[offset:offset+3],so3_to_euler_zyx_batch_nb(vector[None,offset+3:offset+9])[0],vector[offset+9]].astype(np.float32)
           for key,offset in [('follow1_pos',0),('follow2_pos',10)]}
    images={key:np.rint(sample[f'observation.images.{camera}'].numpy().transpose(1,2,0)*255).astype(np.uint8)
            for key,camera in [('camera_front','faceImg'),('camera_left','leftImg'),('camera_right','rightImg')]}
    return {'images':images,'state':state,'instruction':sample['task'],'step':0,'episode_id':'heldout-interface-check'}


def main():
    model,metadata=load_policy_model(ROOT/'deploy.yaml')
    obs=observation()
    original_infer=model.base.infer_actions
    def record_candidate():
        candidate=original_infer()
        np.save(ROOT/'outputs/raw-policy-candidate.npy',candidate)
        print('RAW_GRIPPER_RANGE',float(candidate[:,[6,13]].min()),float(candidate[:,[6,13]].max()),flush=True)
        return candidate
    model.base.infer_actions=record_candidate
    model.initialize_episode({'episode_id':obs['episode_id'],'instruction':obs['instruction']})
    model.ingest_observation(obs)
    start=time.monotonic()
    action=model.infer_actions()
    latency=time.monotonic()-start
    assert action.shape==(33,14) and np.isfinite(action).all()
    assert model.fallback_count==0,'Fallback output must not count as genuine model inference'
    state=np.concatenate(list(obs['state'].values()))
    assert np.max(np.abs(action[1:]-state))>1e-5,'Unexpected all-hold policy output'
    report={'shape':list(action.shape),'finite':True,'fallback_count':model.fallback_count,
            'latency_seconds':latency,'metadata':metadata,'task_success_rate':None,
            'validation':'one held-out observation, no simulator execution'}
    (ROOT/'outputs/interface-check.json').write_text(json.dumps(report,indent=2))
    print('REAL_INFERENCE_OK',json.dumps(report),flush=True)
    model.finalize_episode()
    model.close()


if __name__=='__main__':
    if sys.argv[1:]==['--export-observation']:
        obs=observation()
        np.savez_compressed(ROOT/'outputs/heldout-observation.npz',**obs['images'],**obs['state'],instruction=obs['instruction'])
        print('OBSERVATION_EXPORTED')
    else:
        main()
