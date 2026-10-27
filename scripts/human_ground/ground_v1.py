from __future__ import annotations
import hashlib
import numpy as np

POLICY = 'sink10cm-cropZnegative-diskR1.5-Z0-human8000-ground2000-rgb04-v1'

def transform(xyz, sample_id):
    xyz = np.asarray(xyz, dtype=np.float32)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid input: ' + sample_id)
    shifted = xyz.copy(); shifted[:, 2] -= .1
    retained = shifted[shifted[:, 2] >= 0]
    n = len(retained)
    if not n:
        raise ValueError('No human points after clipping: ' + sample_id)
    seed = int(hashlib.sha256(('ground-v1:' + sample_id).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    ix = (rng.choice(n, 8000, replace=False) if n >= 8000 else
          rng.permutation(np.concatenate([np.arange(n), rng.choice(n, 8000-n)])))
    xy = (xyz[:, :2].min(0)+xyz[:, :2].max(0))/2
    radius = 1.5*np.sqrt(rng.random(2000)); theta = rng.uniform(0, 2*np.pi, 2000)
    ground = np.column_stack([xy[0]+radius*np.cos(theta), xy[1]+radius*np.sin(theta), np.zeros(2000)]).astype(np.float32)
    combined = np.concatenate([retained[ix], ground]).astype(np.float32)
    center = combined.mean(0); scale = float(np.linalg.norm(combined-center, axis=1).max())
    normalized = (combined-center)/scale
    pc = np.concatenate([normalized, np.full((10000,3), .4, np.float32)], axis=1)
    assert (combined[:,2]>=0).all() and (combined[8000:,2]==0).all()
    return combined, pc, dict(sample_id=sample_id, seed=seed, original_points=len(xyz),
        retained_points=n, cropped_points=len(xyz)-n, center_m=center.tolist(),scale_m=scale,
        disk_center_xy_m=xy.tolist(), human_points=8000, ground_points=2000)
