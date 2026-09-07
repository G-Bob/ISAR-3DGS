import tempfile
from pathlib import Path
import torch
from radar_gs.model import ScatteringNetwork, RadarGaussianScene
from radar_gs.io import save_checkpoint, load_scene

torch.manual_seed(42)
legacy=ScatteringNetwork()
torch.manual_seed(42)
explicit=ScatteringNetwork(direction_encoding_levels=4)
assert all(torch.equal(v,explicit.state_dict()[k]) for k,v in legacy.state_dict().items())
low=ScatteringNetwork(direction_encoding_levels=1)
assert low.net[0].in_features == legacy.net[0].in_features-36
assert low.encoding_levels==4
inputs=[torch.randn(7,3),torch.nn.functional.normalize(torch.randn(7,3),dim=-1),torch.randn(7,16),torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])]
assert torch.equal(legacy(*inputs),explicit(*inputs))
low(*inputs).sum().backward()
assert all(torch.isfinite(p.grad).all() for p in low.parameters())
for levels in (1,4):
    scene=RadarGaussianScene(inputs[0],inputs[1],.04,.012,5.,direction_encoding_levels=levels)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'scene.pt'
        save_checkpoint(path,scene,{'final_support_temperature':1.},0)
        reloaded,_=load_scene(path,torch.device('cpu'))
        assert reloaded.scattering.direction_encoding_levels==levels
        assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in scene.state_dict().items())
        if levels==4:
            payload=torch.load(path,weights_only=False)
            del payload['model_shape']['direction_encoding_levels']
            torch.save(payload,path)
            historical,_=load_scene(path,torch.device('cpu'))
            assert historical.scattering.direction_encoding_levels==4
print('PASS: legacy equivalence, spatial encoding retained, gradients, both checkpoint roundtrips, historical fallback')
