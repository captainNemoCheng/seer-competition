"""Check a real remote Policy Space lifecycle without exposing the bearer URL in logs."""
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np
from websockets.sync.client import connect
from policy_space.runtime.server import _safe_packb, _safe_unpackb, _pack_array, _unpack_array

parser=argparse.ArgumentParser()
parser.add_argument('--url-file',required=True,type=Path)
parser.add_argument('--observation',required=True,type=Path)
parser.add_argument('--report',required=True,type=Path)
parser.add_argument('--system-trust',action='store_true',help='Use OS certificate trust via optional truststore package')
args=parser.parse_args()
if args.system_trust:
    import truststore
    truststore.inject_into_ssl()
url=args.url_file.read_text().strip()
sample=np.load(args.observation)
episode='external-check-'+uuid.uuid4().hex
observation={'images':{key:sample[key] for key in ('camera_front','camera_left','camera_right')},
             'state':{key:sample[key] for key in ('follow1_pos','follow2_pos')},
             'instruction':str(sample['instruction']),'step':0,'episode_id':episode}
with connect(url,open_timeout=30,max_size=64*1024*1024,compression=None,ping_interval=None) as ws:
    metadata=_safe_unpackb(ws.recv(timeout=30),object_hook=_unpack_array)
    assert metadata.get('protocol_version')=='2' and metadata.get('policy_name')=='wallx',metadata
    def request(endpoint, **payload):
        ws.send(_safe_packb({'endpoint':endpoint,**payload},default=_pack_array))
        response=_safe_unpackb(ws.recv(timeout=120),object_hook=_unpack_array)
        if 'error' in response: raise RuntimeError(response)
        return response
    request('initialize_episode',episode_id=episode,instruction=observation['instruction'],
            observation_schema='bimanual_rgb_ee@1',action_schema='bimanual_ee_absolute@1',embodiment_schema='ex001_6r@1')
    request('ingest_observation',**observation)
    start=time.monotonic()
    result=request('infer_actions',episode_id=episode)
    latency=time.monotonic()-start
    action=np.asarray(result['actions'])
    assert action.shape==(33,14) and np.isfinite(action).all()
    state=np.concatenate(list(observation['state'].values()))
    assert np.max(np.abs(action[1:]-state))>1e-5,'Hold-only output does not pass the real-model check'
    request('finalize_episode',episode_id=episode)
    report={'external_lifecycle':'passed','shape':list(action.shape),'finite':True,
            'inference_roundtrip_seconds':latency,'metadata':metadata,'platform_task_score':None}
    args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
