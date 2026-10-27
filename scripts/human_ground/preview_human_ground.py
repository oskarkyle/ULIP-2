from __future__ import annotations

import csv
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path('/media/robosense/DataCenterNew/Kyle/ULIP-2/outputs')
OUT = ROOT / 'ulip_human_ground_v1/input_preview'
sys.path.insert(0, str(ROOT / 'ulip_supcon_v3_runner'))
from preprocess import ply_arrays

LABELS = ['standing', 'crouching_shallow', 'crouching_deep', 'kneeling', 'lying', 'supported']


def augment(xyz, sample_id):
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid points')
    seed = int(hashlib.sha256(('ground-v1:' + sample_id).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    shifted = xyz.copy()
    shifted[:, 2] -= 0.1
    retained = shifted[shifted[:, 2] >= 0]
    n = len(retained)
    if n == 0:
        raise ValueError('No human points remain above ground')
    indices = (rng.choice(n, 8000, replace=False) if n >= 8000 else
               rng.permutation(np.concatenate([np.arange(n), rng.choice(n, 8000-n)])))
    human = retained[indices].copy()
    xy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
    # sqrt(radius draw) gives uniform density over disk area.
    radius = 1.5 * np.sqrt(rng.random(2000))
    theta = rng.uniform(0, 2*np.pi, 2000)
    ground = np.column_stack([xy[0]+radius*np.cos(theta), xy[1]+radius*np.sin(theta), np.zeros(2000)]).astype(np.float32)
    combined = np.concatenate([human, ground]).astype(np.float32)
    center = combined.mean(0)
    scale = float(np.linalg.norm(combined-center, axis=1).max())
    normalized = (combined-center)/scale
    assert np.all(ground[:, 2] == 0)
    assert np.all(human[:, 2] >= 0)
    assert np.max(np.linalg.norm(ground[:, :2]-xy, axis=1)) <= 1.500001
    meta = dict(seed=seed, radius_m=1.5, ground_z_m=0, ground_center_xy_m=xy.tolist(),
                human_translation_m=[0, 0, -0.1],
                shifted_min_z_m=float(human[:, 2].min()),
                original_points=len(xyz), retained_points=n, cropped_points=len(xyz)-n,
                crop_policy='after -0.1m translation, retain Z>=0 before sampling',
                human_input_points=8000, ground_input_points=2000,
                original_min_z_m=float(xyz[:, 2].min()), original_max_z_m=float(xyz[:, 2].max()),
                original_below_ground_points=int((xyz[:, 2]<0).sum()),
                normalization_center_m=center.tolist(), normalization_scale_m=scale,
                normalized_ground_z=float(-center[2]/scale), rgb_constant=0.4)
    return combined, normalized, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT/'ulip_only_human_projection_v1/cache'
    ids = json.loads((cache/'sample_ids.json').read_text())
    lookup = {sid:i for i,sid in enumerate(ids)}
    offsets = np.load(cache/'offsets.npy')
    xyz = np.memmap(cache/'xyz.f32', mode='r', dtype='<f4').reshape(-1, 3)
    assert offsets[-1] == len(xyz)
    rows = [json.loads(s) for s in (ROOT/'ulip_trimodal_supcon_v3/human_text/data/train.jsonl').read_text().splitlines()]
    selected = []
    for label in LABELS:
        row = min((r for r in rows if r['label']==label), key=lambda r: hashlib.sha256(r['sample_id'].encode()).hexdigest())
        assert 'reference floor Z=0' in row['coordinate_systems']['ply']
        i = lookup[row['sample_id']]
        points = np.array(xyz[offsets[i]:offsets[i+1]])
        assert len(points) == row['points']
        selected.append(('sim', label, row['sample_id'], points))
    real = ROOT/'ulip_real_human_manual50_v1'
    manual = list(csv.DictReader((real/'manual_labels.csv').open(encoding='utf-8-sig')))
    for label in LABELS[:-1]:
        row = min((r for r in manual if r['final_label']==label), key=lambda r:r['sample_id'])
        points, _ = ply_arrays((real/row['ply_path']).read_bytes())
        selected.append(('real', label, row['sample_id'], points))
    entries = []
    for source, label, sid, points in selected:
        combined, normalized, meta = augment(points, sid)
        name = f'{source}_{label}_{sid}'
        meta.update(source=source, label=label, sample_id=sid,
                    note='Colors are visualization only. Model input uses RGB=.4 for both human and disk. No ray occlusion or floor fitting.')
        np.savez_compressed(OUT/f'{name}.npz', original_xyz=points, input_xyz_m=combined,
                            normalized_xyz=normalized, is_ground=np.arange(10000)>=8000)
        (OUT/f'{name}.json').write_text(json.dumps(meta, indent=2)+'\n')
        fig = make_subplots(rows=1, cols=3, specs=[[{'type':'scene'}]*3], subplot_titles=['原始人体（米）','人体8000点＋地面2000点（米）','联合归一化后的输入'])
        def scatter(p, col, color, legend):
            fig.add_trace(go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2], mode='markers',
                          marker=dict(size=2, color=color), name=legend, showlegend=(col==2)),row=1,col=col)
        scatter(points,1,'#2684ff','人体')
        for col, p in [(2,combined),(3,normalized)]:
            scatter(p[:8000],col,'#2684ff','人体')
            scatter(p[8000:],col,'#e99b30','人工地面')
        # Identical metric axes in the before/after panels for honest scale comparison.
        lo=combined.min(0)-0.1; hi=combined.max(0)+0.1
        metric=dict(aspectmode='data',xaxis=dict(title='X (m)',range=[lo[0],hi[0]]),yaxis=dict(title='Y (m)',range=[lo[1],hi[1]]),zaxis=dict(title='Z (m)',range=[lo[2],hi[2]]))
        fig.update_layout(title=f'{source} | {label} | {sid} | R=1.5m, ground Z=0, human Z−0.1m',height=720,
                          scene=metric,scene2=metric,scene3=dict(aspectmode='data'),margin=dict(l=0,r=0,b=0))
        fig.write_html(OUT/f'{name}.html', include_plotlyjs='directory')
        entries.append(dict(file=f'{name}.html',**meta))
    (OUT/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    links='\n'.join(f'<tr><td>{e["source"]}</td><td>{e["label"]}</td><td><a href="{html.escape(e["file"])}">{e["sample_id"]}</a></td><td>{e["original_points"]}</td><td>{e["original_min_z_m"]:.3f}</td><td>{e["original_below_ground_points"]}</td></tr>' for e in entries)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>人体＋圆盘地面预览</title><style>body{font:16px sans-serif;margin:32px}td,th{padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>人体＋圆盘地面：输入预览</h1><p>半径1.5米；人体XY包围盒中心；地面Z=0。原始人体不抬高、不重新贴地。蓝色人体，橙色地面；颜色仅为展示，模型输入统一RGB=0.4。</p><p>每页从左至右：原始人体、加地面、联合归一化。鼠标旋转、滚轮缩放，可点击图例隐藏地面。地面是人工均匀圆盘，不是LiDAR模拟扫描；可能穿过人体底部，不做遮挡模拟。</p><p>6个仿真训练样本＋5个真实诊断样本，每类首个/固定哈希抽样，不按效果挑选。真实没有撑姿真值。地面下的原始噪点保留，供检查；本次未启动训练。</p><table><tr><th>来源</th><th>类别</th><th>打开预览</th><th>原始点数</th><th>最低Z(m)</th><th>Z&lt;0点数</th></tr>'+links+'</table>')
    print(json.dumps([dict(source=e['source'],label=e['label'],min_z=e['original_min_z_m'],below=e['original_below_ground_points']) for e in entries],indent=2))
    index_path = OUT/'index.html'
    index_path.write_text(index_path.read_text().replace('原始人体不抬高、不重新贴地。', '人体统一下移0.1米后裁掉Z＜0的点，再采样8000个人体点；地面保持Z=0。左图保留原始人体作为对照，右两图使用裁剪后的数据。').replace('地面下的原始噪点保留，供检查；', '地下人体点已裁掉；'))
    print(OUT/'index.html')


if __name__=='__main__':
    main()
