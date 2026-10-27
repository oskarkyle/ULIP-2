from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from ground_v1 import transform, POLICY

ROOT=Path('/workspace/ULIP-2')
BASE=ROOT/'outputs'
EXP=BASE/'ulip_human_ground_v1'
sys.path.insert(0,str(BASE/'ulip_supcon_v3_runner'))
sys.path.insert(0,str(ROOT/'repo'))
from config import LABELS, atomic_json, digest, write_rows
from preprocess import ply_arrays
from train import metrics

def main():
    torch.set_num_threads(4)
    start=time.monotonic()
    out=EXP/'human_text/real50_evaluation'
    out.mkdir(exist_ok=False)
    real=BASE/'ulip_real_human_manual50_v1'
    rows=list(csv.DictReader((real/'manual_labels.csv').open(encoding='utf-8-sig')))
    assert len(rows)==50 and len({r['sample_id'] for r in rows})==50
    pcs=[];raws=[];metas=[]
    for r in rows:
        path=real/r['ply_path']
        xyz,_=ply_arrays(path.read_bytes())
        raw,pc,meta=transform(xyz,r['sample_id'])
        meta.update(ply_sha256=digest(path),label=r['final_label'])
        pcs.append(pc);raws.append(raw);metas.append(meta)
    np.savez_compressed(out/'inputs.npz',input_xyz_m=np.stack(raws),sample_ids=np.array([r['sample_id'] for r in rows]))
    atomic_json(out/'input_metadata.json',metas)
    from models.pointbert.point_encoder import PointTransformer_Colored
    from utils.utils import cfg_from_yaml_file
    ck=ROOT/'checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt'
    identity=json.loads((EXP/'human_text/features/identity.json').read_text())
    assert digest(ck)==identity['checkpoint_sha256'] and identity['preprocess']==POLICY
    state=torch.load(ck,map_location='cpu')['state_dict']
    state={(k[7:] if k.startswith('module.') else k):v for k,v in state.items()}
    model=PointTransformer_Colored(cfg_from_yaml_file(str(ROOT/'repo/models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml')).model,args=argparse.Namespace())
    model.load_state_dict({k[len('point_encoder.'):]:v for k,v in state.items() if k.startswith('point_encoder.')},strict=True)
    del state
    model.eval().requires_grad_(False).cuda()
    chunks=[];encoding_start=time.monotonic()
    with torch.inference_mode():
        for i in range(0,len(rows),8):
            chunks.append(model(torch.from_numpy(np.stack(pcs[i:i+8])).cuda()).cpu())
            print('Encoded',min(i+8,len(rows)),'/50',flush=True)
    x=torch.cat(chunks)
    encoding_s=time.monotonic()-encoding_start
    assert x.shape==(50,768) and torch.isfinite(x).all()
    np.save(out/'point_features.npy',x.numpy())
    t=torch.nn.functional.normalize(torch.from_numpy(np.load(EXP/'human_text/features/text.npy')),dim=1)
    truth=[LABELS.index(r['final_label']) for r in rows]
    assert np.bincount(truth,minlength=6).tolist()==[10,10,10,10,10,0]
    results={};predictions=[dict(sample_id=r['sample_id'],label=r['final_label'],ply_path=r['ply_path']) for r in rows]
    for arm in ['baseline','supcon']:
        path=EXP/'human_text'/arm/'best_projection.pt'
        w=torch.load(path,map_location='cpu')
        scores=(torch.nn.functional.normalize(x@w['pc_projection'],dim=1)@t.T).numpy()
        pred=scores.argmax(1);result=metrics(truth,pred)
        result['macro_f1_present5']=float(np.mean(result.pop('f1')[:5]))
        result.pop('macro_f1')
        result['recall'][-1]=None
        result.update(best_epoch=w['epoch'],projection_sha256=digest(path))
        results[arm]=result
        for i in range(50):
            predictions[i][arm]=dict(prediction=LABELS[pred[i]],correct=bool(pred[i]==truth[i]),cosine_scores=dict(zip(LABELS,map(float,scores[i]))))
    old=json.loads((BASE/'ulip_trimodal_supcon_v3/human_text/real50_evaluation/summary.json').read_text())
    assert old['labels_sha256']==digest(real/'manual_labels.csv')
    summary=dict(status='complete',samples=50,labels=LABELS,arms=results,old_no_ground=old['arms'],
        policy=POLICY,labels_sha256=digest(real/'manual_labels.csv'),encoding_s=encoding_s,wall_s=time.monotonic()-start,
        scope='Real diagnostic set already inspected; no training/tuning. No supported ground truth. Ground+sink+clip changes applied jointly.')
    atomic_json(out/'summary.json',summary);write_rows(out/'predictions.jsonl',predictions)
    lines=['# 加地面后的真实50帧评估','', '人体下移0.1m、裁掉Z<0、圆盘半径1.5m、地面Z=0；8000人体点+2000地面点，联合归一化。重新计算冻结编码器特征，只做推理。','',
        '| 指标 | 原baseline | 原supcon | 地面baseline | 地面supcon |','|---|---:|---:|---:|---:|']
    four=[old['arms']['baseline'],old['arms']['supcon'],results['baseline'],results['supcon']]
    lines.append('| Accuracy | '+' | '.join(f'{r["accuracy"]:.2%}' for r in four)+' |')
    lines.append('| Macro-F1（五类） | '+' | '.join(f'{r["macro_f1_present5"]:.2%}' for r in four)+' |')
    for k,label in enumerate(LABELS[:5]):
        lines.append('| '+label+' 正确数/10 | '+' | '.join(str(r['confusion'][k][k]) for r in four)+' |')
    for arm,r in results.items():
        lines+=['',f'## {arm} 混淆矩阵','', '行真值、列预测：'+', '.join(LABELS),'','```text']
        lines += [f'{LABELS[i]:20} '+ ' '.join(f'{v:3}' for v in row) for i,row in enumerate(r['confusion'])]
        lines+=['```']
    lines+=['','## 限制','', '- 只有50帧、每个真实类别10帧；一个样本影响总体准确率2个百分点。','- 没有真实supported真值，不能评价撑姿召回率。','- 此集合已被多次查看，仅作诊断，不是新的盲测；本次没有训练或调整参数。','- 地面、下沉和裁剪是联合改变，不能将变化单独归因于地面。',f'- 编码耗时 {encoding_s:.2f}s；脚本至指标落盘前耗时 {summary["wall_s"]:.2f}s，不含容器启动。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
