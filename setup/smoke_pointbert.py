import json, sys, argparse, time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from pointnet2_ops import pointnet2_utils
from models.pointbert.point_encoder import PointTransformer_Colored
from utils.utils import cfg_from_yaml_file
config=cfg_from_yaml_file('models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml')
args=argparse.Namespace()
model=PointTransformer_Colored(config.model,args=args)
path=next(Path('../checkpoints').glob('*10k-xyzrgb*.pt'))
ckpt=torch.load(path,map_location='cpu')
state={(k[7:] if k.startswith('module.') else k):v for k,v in ckpt['state_dict'].items()}
point_state={k[len('point_encoder.'):]:v for k,v in state.items() if k.startswith('point_encoder.')}
result=model.load_state_dict(point_state,strict=True)
model=model.eval().cuda()
projection=state['pc_projection'].cuda()
points=torch.randn(2,10000,6,device='cuda')
points[:,:,3:]=torch.rand(2,10000,3,device='cuda')
start=time.time()
with torch.no_grad():
    features=model(points)
    embedding=features @ projection
    assert embedding.shape==(2,1280),embedding.shape
    assert torch.isfinite(embedding).all()
torch.cuda.synchronize()
report={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'checkpoint':str(path),'point_encoder_strict_load':str(result),'feature_shape':list(features.shape),'embedding_shape':list(embedding.shape),'seconds':time.time()-start,'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3}
Path('../outputs/pointbert-smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
