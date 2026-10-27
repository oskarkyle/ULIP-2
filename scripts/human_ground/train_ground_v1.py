from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from ground_v1 import POLICY, transform

ROOT = Path(os.environ.get('ULIP_ROOT', '/workspace/ULIP-2'))
RUNNER = ROOT/'outputs/ulip_supcon_v3_runner'
sys.path.insert(0,str(RUNNER))
from config import digest, atomic_json, read_rows, require_identity

def main():
    import torch
    torch.set_num_threads(4)
    old = ROOT/'outputs/ulip_trimodal_supcon_v3'
    exp = ROOT/'outputs/ulip_human_ground_v1'
    out = exp/'human_text'; out.mkdir(parents=True,exist_ok=True)
    if not (out/'data').exists(): shutil.copytree(old/'human_text/data',out/'data')
    for name in ('records','train','validation','test'):
        assert digest(out/'data'/f'{name}.jsonl')==digest(old/'human_text/data'/f'{name}.jsonl')
    if not (exp/'split_groups.json').exists(): shutil.copy2(old/'split_groups.json',exp/'split_groups.json')
    c=json.loads((RUNNER/'configs/human_text.json').read_text());c['experiment']='ulip_human_ground_v1'
    config_file=exp/'human_text.json';require_identity(config_file,c)
    rows=read_rows(out/'data/records.jsonl')
    source=ROOT/'outputs/ulip_only_human_projection_v1/cache'
    ids=json.loads((source/'sample_ids.json').read_text());lookup={s:i for i,s in enumerate(ids)}
    offsets=np.load(source/'offsets.npy');xyz=np.memmap(source/'xyz.f32',mode='r',dtype='<f4').reshape(-1,3)
    assert len(ids)+1==len(offsets) and offsets[-1]==len(xyz)
    cache=out/'features';cache.mkdir(exist_ok=True)
    chunks=out/'input_chunks';chunks.mkdir(exist_ok=True)
    ck=ROOT/c['checkpoint']
    print('Hashing source cache and checkpoint',flush=True)
    identity=dict(mode='human_text',records_hash=digest(out/'data/records.jsonl'),preprocess=POLICY,
        checkpoint_sha256=digest(ck),source_xyz_sha256=digest(source/'xyz.f32'),
        source_offsets_sha256=digest(source/'offsets.npy'),source_ids_sha256=digest(source/'sample_ids.json'),
        text_hash=digest(old/'human_text/features/text.npy'))
    require_identity(cache/'identity.json',identity)
    start=time.monotonic()
    if not (cache/'complete.json').exists():
        sys.path.insert(0,str(ROOT/'repo'))
        from models.pointbert.point_encoder import PointTransformer_Colored
        from utils.utils import cfg_from_yaml_file
        state=torch.load(ck,map_location='cpu')['state_dict']
        state={(k[7:] if k.startswith('module.') else k):v for k,v in state.items()}
        model=PointTransformer_Colored(cfg_from_yaml_file(str(ROOT/'repo/models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml')).model,args=argparse.Namespace())
        model.load_state_dict({k[len('point_encoder.'):]:v for k,v in state.items() if k.startswith('point_encoder.')},strict=True)
        del state
        model.eval().requires_grad_(False).cuda()
        print('Frozen encoder loaded; generating new ground inputs and features',flush=True)
        for pos in range(0,len(rows),32):
            target=chunks/f'chunk_{pos:08d}.npz'; receipt=chunks/f'chunk_{pos:08d}.json'
            batch=rows[pos:pos+32];wanted=[r['record_id'] for r in batch]
            if target.exists() and receipt.exists():
                m=json.loads(receipt.read_text())
                if m['record_ids']!=wanted or m['sha256']!=digest(target):raise ValueError('Invalid chunk '+str(target))
                continue
            pcs=[];raw=[];meta=[]
            for r in batch:
                i=lookup[r['sample_id']];points=np.asarray(xyz[offsets[i]:offsets[i+1]])
                assert len(points)==r['points']
                raw_pc,pc,info=transform(points,r['sample_id']);pcs.append(pc);raw.append(raw_pc);meta.append(info)
            feats=[]
            with torch.inference_mode():
                for j in range(0,len(pcs),8):
                    feats.append(model(torch.from_numpy(np.stack(pcs[j:j+8])).cuda()).cpu().numpy())
            features=np.concatenate(feats).astype(np.float32)
            assert features.shape==(len(batch),768) and np.isfinite(features).all()
            tmp=target.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,input_xyz_m=np.stack(raw),point_features=features)
            tmp.replace(target)
            atomic_json(receipt,dict(record_ids=wanted,sha256=digest(target),inputs=meta,policy=POLICY))
            elapsed=time.monotonic()-start
            progress=dict(stage='encoding',processed=pos+len(batch),total=len(rows),wall_s=elapsed,poses_per_s=(pos+len(batch))/elapsed)
            atomic_json(exp/'progress.json',progress)
            if pos%320==0:print(json.dumps(progress),flush=True)
        all_features=[]
        for pos in range(0,len(rows),32):
            with np.load(chunks/f'chunk_{pos:08d}.npz') as f:all_features.append(f['point_features'])
        np.save(cache/'point.npy',np.concatenate(all_features))
        shutil.copy2(old/'human_text/features/text.npy',cache/'text.npy')
        atomic_json(cache/'record_ids.json',[r['record_id'] for r in rows])
        atomic_json(cache/'complete.json',dict(identity,files={n:digest(cache/n) for n in ['point.npy','text.npy','record_ids.json']},encoding_wall_s=time.monotonic()-start))
        del model;torch.cuda.empty_cache()
    from train import run
    for arm in ['baseline','supcon']:
        if not (out/arm/'complete.json').exists():
            run(argparse.Namespace(config=str(config_file),stage='train',arm=arm,device='cuda:0',resume=(out/arm/'last.pt').exists(),dry_run=False))
    run(argparse.Namespace(config=str(config_file),stage='evaluate',arm='supcon',device='cuda:0',resume=False,dry_run=False))
    atomic_json(exp/'progress.json',dict(stage='complete',status='evaluated',total=len(rows),wall_s=time.monotonic()-start))
    print('GROUND EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':main()
