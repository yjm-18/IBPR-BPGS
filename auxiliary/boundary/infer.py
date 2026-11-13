"""Frozen geometry-only predictor; identical core/halo policy to the reported experiments."""
import argparse
import numpy as np
import torch
from train_boundary_only import BoundaryOnlyModel

def infer(pos, checkpoint):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = BoundaryOnlyModel(boundary_cfg=ck['boundary_cfg']).cuda().eval()
    model.load_state_dict(ck['state_dict'], strict=True)
    del ck
    def predict(ids):
        p = pos[ids].copy()
        p -= p.min(0)
        cells = (p * 10).astype(np.int64)
        cells, inv = np.unique(cells, axis=0, return_inverse=True)
        batch = dict(voxel_coords=torch.from_numpy(np.column_stack([np.zeros(len(cells), np.int32), cells])).int().cuda(), coords_float=torch.from_numpy(p).cuda(), v2p_map=torch.from_numpy(inv).long().cuda(), feats=torch.zeros((len(cells), 3), device='cuda'), feats_layout='voxel', spatial_shape=np.maximum(cells.max(0) + 1, 512).tolist(), batch_size=1)
        with torch.no_grad():
            out = model(batch, mode='predict')['boundary_probs'].cpu().numpy()
        del batch
        torch.cuda.empty_cache()
        return out
    leaves = []

    def split(ids):
        if len(ids) <= 800000:
            leaves.append(ids)
            return
        axis = int(np.argmax(np.ptp(pos[ids, :2], axis=0)))
        order = ids[np.argsort(pos[ids, axis], kind='stable')]
        mid = len(order) // 2
        split(order[:mid])
        split(order[mid:])
    split(np.arange(len(pos)))
    out = np.empty(len(pos), np.float32)
    coverage = np.zeros(len(pos), np.uint8)
    for core in leaves:
        if len(leaves) == 1:
            context = core
        else:
            lo = pos[core].min(0) - 2
            hi = pos[core].max(0) + 2
            context = np.flatnonzero(np.all((pos >= lo) & (pos <= hi), axis=1))
        vals = predict(context)
        out[core] = vals[np.searchsorted(context, core)] if len(leaves) > 1 else vals
        coverage[core] += 1
    assert np.all(coverage == 1) and np.isfinite(out).all() and np.all((out >= 0) & (out <= 1))
    return out

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--points',required=True,help='NPZ with points array in metres')
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with np.load(a.points) as z: points=z['points'].copy()
    np.save(a.output,infer(points,a.checkpoint))
