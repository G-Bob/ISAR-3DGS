"""M19 controlled Pei-module ablations; install before training AND evaluation."""
import os, runpy, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pei_bsat_wide import sh3,rotation


def install(mode):
    if mode == 'sh3':
        from radar_gs.model import RadarGaussianScene
        # Reuse 16 per-Gaussian features as coefficients so topology/Adam are native.
        # This is bounded SH, not Pei's clamp(.5+SH); the amplitude range stays M19's.
        def amplitude(self,kr,kd):
            return self.scattering.maximum_amplitude*torch.sigmoid((self.features*sh3(kr/kr.norm())).sum(-1))
        RadarGaussianScene.scattering_amplitude=amplitude
    elif mode == 'full_covariance':
        import ast,inspect,textwrap
        import radar_gs.model as model
        from scipy.spatial.transform import Rotation
        original_init=model.RadarGaussianScene.__init__
        original_groups=model.RadarGaussianScene.parameter_groups
        def initialize(self,*args,**kwargs):
            original_init(self,*args,**kwargs)
            basis=model.build_surface_basis(self.normals()).detach().cpu().numpy()
            xyzw=Rotation.from_matrix(basis).as_quat()
            self.pei_quaternions=torch.nn.Parameter(torch.tensor(xyzw[:,[3,0,1,2]],device=self.means.device,dtype=self.means.dtype))
        def covariance(self):
            r=rotation(self.pei_quaternions);return r@torch.diag_embed(self.scales().square())@r.transpose(-1,-2)
        def groups(self,lr):return original_groups(self,lr)+[{'params':[self.pei_quaternions],'lr':lr*.1}]
        source=textwrap.dedent(inspect.getsource(model.RadarGaussianScene.adaptive_control))
        source=source.replace('normals = self.normal_parameters.detach()[source_indices].clone()', 'normals = self.normal_parameters.detach()[source_indices].clone()\n        pei_quaternions = self.pei_quaternions.detach()[source_indices].clone()')
        source=source.replace('normals[relocation_candidates] = self.normal_parameters.detach()[donors]', 'normals[relocation_candidates] = self.normal_parameters.detach()[donors]\n                pei_quaternions[relocation_candidates] = self.pei_quaternions.detach()[donors]')
        source=source.replace('self.normal_parameters = nn.Parameter(normals)','self.normal_parameters = nn.Parameter(normals)\n        self.pei_quaternions = nn.Parameter(pei_quaternions)')
        ast.parse(source);namespace=dict(model.__dict__);exec(compile(source,model.__file__,'exec'),namespace)
        model.RadarGaussianScene.__init__=initialize;model.RadarGaussianScene.covariance=covariance;model.RadarGaussianScene.parameter_groups=groups;model.RadarGaussianScene.adaptive_control=namespace['adaptive_control']
    elif mode != 'no_visibility':
        raise ValueError(mode)


if __name__=='__main__':
    install(os.environ['HYBRID_MODE'])
    os.environ.update(R6_MASK='1',R6_TRACK='1',R6_GUARD='1')
    runpy.run_path(str(Path(__file__).resolve().parent/'round6_fast_worker.py'),run_name='__main__')
