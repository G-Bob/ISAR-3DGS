import torch
from radar_gs.radar_vggt import RadarVGGTInitializer,radar_vggt_surface
class View:
 def __init__(self):
  self.image=torch.zeros(32,32); self.image[12:20,12:20]=1
  self.kr=torch.tensor([1.,0,0]); self.kd=torch.tensor([0.,1,0]); self.normal=torch.tensor([0.,0,1.]); self.dr=self.dd=.05
def test_shapes():
 d,c=RadarVGGTInitializer(dim=32,heads=4,layers=1)(torch.randn(3,16,14)); assert d.shape==c.shape==(3,16)
def test_cpu_initializer():
 p,n,i=radar_vggt_surface(40,2.,42,[View(),View()],view_count=2,patch_grid=4,steps=2,device='cpu'); assert p.shape==n.shape==(40,3) and i.selected==40
