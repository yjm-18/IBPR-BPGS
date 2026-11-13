"""FOR-instance raw LAS -> class3 removal -> 0.1m -> fixed domain-K binary labels.
Uses the fixed domain-specific K values below.
"""
import argparse
from pathlib import Path
import laspy
import numpy as np
import torch
from scipy.spatial import cKDTree
DOMAIN_K={'CULS':128,'NIBIO':192,'RMIT':128,'SCION':256,'TUWIEN':160}
ROOM={k:i for i,k in enumerate(DOMAIN_K)}
def voxelize(xyz: np.ndarray, gt: np.ndarray, voxel: float=0.1):
    origin = xyz.min(0)
    cell = np.floor((xyz - origin) / voxel).astype(np.int32)
    _, inverse, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    n = len(counts)
    centers = np.column_stack([np.bincount(inverse, weights=xyz[:, axis], minlength=n) / counts for axis in range(3)]).astype(np.float32)
    order = np.lexsort((gt, inverse))
    si, sg = (inverse[order], gt[order])
    starts = np.r_[0, np.flatnonzero((si[1:] != si[:-1]) | (sg[1:] != sg[:-1])) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    pair_voxel, pair_gt = (si[starts], sg[starts])
    ranked = np.lexsort((pair_gt, -lengths, pair_voxel))
    rv = pair_voxel[ranked]
    first = np.r_[True, rv[1:] != rv[:-1]]
    chosen = ranked[first]
    voxel_gt = np.empty(n, np.int32)
    voxel_gt[pair_voxel[chosen]] = pair_gt[chosen]
    return (centers, voxel_gt, origin.astype(np.float32))

def boundary_knn(xyz: np.ndarray, gt: np.ndarray, neighbors: int) -> np.ndarray:
    valid_ids = np.flatnonzero(gt > 0)
    points, labels = (xyz[valid_ids], gt[valid_ids])
    query_k = min(neighbors + 1, len(valid_ids))
    tree = cKDTree(points)
    output = np.zeros(len(gt), np.uint8)
    for start in range(0, len(valid_ids), 50000):
        local = np.arange(start, min(start + 50000, len(valid_ids)))
        _, indices = tree.query(points[local], k=query_k, workers=8)
        if query_k == 1:
            indices = indices[:, None]
        nonself = indices != local[:, None]
        rank = np.cumsum(nonself, axis=1)
        use = nonself & (rank <= min(neighbors, len(valid_ids) - 1))
        different = use & (labels[indices] != labels[local, None])
        output[valid_ids[local]] = different.any(axis=1)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--las',required=True);p.add_argument('--domain',choices=list(DOMAIN_K),required=True)
    p.add_argument('--output',required=True,help='Path ending _merged_inst_nostuff.pth under train/ or val/')
    a=p.parse_args();cloud=laspy.read(a.las);keep=np.asarray(cloud.classification)!=3
    xyz=np.column_stack([cloud.x,cloud.y,cloud.z])[keep];gt=np.asarray(cloud['treeID'],np.int32)[keep]
    vx,vg,origin=voxelize(xyz,gt);boundary=boundary_knn(vx,vg,DOMAIN_K[a.domain]);valid=vg>0
    instance=np.full(len(vg),-100,np.int64);instance[valid]=np.searchsorted(np.unique(vg[valid]),vg[valid])
    values=[vx,np.zeros((len(vg),3),np.float32),valid.astype(np.int64),instance,np.zeros(len(vg),np.float32),boundary.astype(np.float32)]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(tuple(torch.from_numpy(v) for v in values)+(torch.tensor([ROOM[a.domain]]),Path(a.las).stem),out)
