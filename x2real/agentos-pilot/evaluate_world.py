"""Re-evaluate saved ensemble; include the stronger last-command baseline."""
import json
import numpy as np
import torch
from latent_world_model import ROOT, HORIZON, Dynamics, metrics

torch.set_num_threads(2)
output = ROOT/'outputs/world-pilot'
checkpoint = torch.load(output/'dynamics.pt', map_location='cpu', weights_only=True)
data = np.load(ROOT/'world_features.npz')
test = data['episode'] >= 270
current, future, actions = (data[key][test] for key in ('current', 'next', 'actions'))
zmean, zstd, amean, astd = (checkpoint[key].numpy() for key in ('zmean','zstd','amean','astd'))
x = np.concatenate(((current-zmean)/zstd, (actions-amean)/astd), -1).astype(np.float32)
predictions = []
for state in checkpoint['members']:
    model = Dynamics().eval()
    model.load_state_dict(state)
    with torch.inference_mode():
        predictions.append(model(torch.from_numpy(x)).numpy())
predicted = current + np.mean(predictions, axis=0)*zstd
command = current.copy()
last = actions.reshape(-1,HORIZON,20)[:,-1]
for offset in (0,10):
    command[:,1536+offset:1539+offset] += last[:,offset:offset+3]
    command[:,1545+offset] = last[:,offset+9]
report = checkpoint['report']
report['learned_predictor'] = metrics(predicted, future, current, zstd)
report['last_command_state_static_visual_baseline'] = metrics(command, future, current, zstd)
(output/'metrics.json').write_text(json.dumps(report, indent=2))
checkpoint['report'] = report
torch.save(checkpoint, output/'dynamics.pt')
print(json.dumps(report, indent=2))
