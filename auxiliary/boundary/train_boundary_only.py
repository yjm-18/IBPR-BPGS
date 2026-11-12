#!/usr/bin/env python3
"""
边界预测专项训练脚本 (轻量版)
==============================

功能：
1. 只训练边界预测网络
2. 使用体素化特征 + 稀疏卷积，避免内存溢出
3. 详细评估边界预测准确性（Precision, Recall, F1, AUC等）

使用方法:
    python train_boundary_only.py --config ginkgo_boundary_segment.yaml
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import numpy as np
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import functools

# ============ 添加当前目录到路径 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from boundary_integrity import (
    BoundaryIntegrityError,
    point_to_voxel_map,
    require_point_shapes,
    validate_metric_arrays,
    validate_voxel_feature_domain,
    validate_voxel_point_table,
)

# 导入稀疏卷积
import spconv.pytorch as spconv


# ==============================================================================
# 边界预测评估指标
# ==============================================================================

class BoundaryMetrics:
    """边界预测评估指标"""

    def __init__(self, threshold=0.5, hard_threshold=0.999):
        self.threshold = threshold
        self.hard_threshold = hard_threshold
        self.reset()

    def reset(self):
        """重置所有计数器"""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.all_probs = []
        self.all_labels = []
        self.threshold_grid = np.linspace(0.05, 0.95, 19)
        self.grid_tp = np.zeros(len(self.threshold_grid), dtype=np.int64)
        self.grid_fp = np.zeros(len(self.threshold_grid), dtype=np.int64)
        self.grid_fn = np.zeros(len(self.threshold_grid), dtype=np.int64)

        # 硬边界专项统计
        self.hard_tp = 0
        self.hard_fn = 0
        self.hard_total = 0

    def update(self, pred_probs, gt_labels, valid_mask=None):
        """
        更新指标

        Args:
            pred_probs: (N,) 边界预测概率
            gt_labels: (N,) 边界标签 [0, 1] 连续值
        """
        pred_probs, gt_labels, valid_mask = validate_metric_arrays(
            pred_probs, gt_labels, valid_mask
        )
        if valid_mask is not None:
            pred_probs = pred_probs[valid_mask]
            gt_labels = gt_labels[valid_mask]

        pred_binary = (pred_probs >= self.threshold).astype(int)
        # Classification quality must be measured against the actual interface,
        # not against the deliberately widened fractional halo.
        gt_binary = (gt_labels >= self.hard_threshold).astype(int)
        gt_hard = gt_binary

        self.tp += np.sum((pred_binary == 1) & (gt_binary == 1))
        self.fp += np.sum((pred_binary == 1) & (gt_binary == 0))
        self.tn += np.sum((pred_binary == 0) & (gt_binary == 0))
        self.fn += np.sum((pred_binary == 0) & (gt_binary == 1))

        self.hard_total += np.sum(gt_hard)
        self.hard_tp += np.sum((pred_binary == 1) & (gt_hard == 1))
        self.hard_fn += np.sum((pred_binary == 0) & (gt_hard == 1))

        labels_bool = gt_binary.astype(bool)
        for grid_index, threshold in enumerate(self.threshold_grid):
            predicted = pred_probs >= threshold
            self.grid_tp[grid_index] += np.count_nonzero(predicted & labels_bool)
            self.grid_fp[grid_index] += np.count_nonzero(predicted & ~labels_bool)
            self.grid_fn[grid_index] += np.count_nonzero(~predicted & labels_bool)

        # Uniformly sample every validation block instead of filling the
        # budget from only the first few blocks/sites.
        sample_count = min(5000, len(pred_probs))
        if sample_count:
            sample_indices = np.linspace(
                0, len(pred_probs) - 1, sample_count, dtype=np.int64
            )
            self.all_probs.extend(pred_probs[sample_indices].tolist())
            self.all_labels.extend(gt_binary[sample_indices].tolist())

    def compute_auc(self):
        """计算 AUC-ROC"""
        if len(self.all_probs) == 0:
            return 0.0

        probs = np.array(self.all_probs)
        labels = np.array(self.all_labels)

        sorted_idx = np.argsort(probs)[::-1]
        labels_sorted = labels[sorted_idx]

        n_pos = labels.sum()
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp, fp = 0, 0
        auc = 0
        tpr_prev, fpr_prev = 0, 0

        for label in labels_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr

        return auc

    def compute_average_precision(self):
        if len(self.all_probs) == 0:
            return 0.0
        probs = np.asarray(self.all_probs)
        labels = np.asarray(self.all_labels, dtype=np.int64)
        positives = int(labels.sum())
        if positives == 0:
            return 0.0
        order = np.argsort(-probs, kind='mergesort')
        labels = labels[order]
        true_positives = np.cumsum(labels)
        ranks = np.arange(1, len(labels) + 1)
        precision = true_positives / ranks
        return float(precision[labels == 1].sum() / positives)

    def compute_best_threshold(self):
        if not np.any(self.grid_tp + self.grid_fn):
            return 0.5, 0.0, 0.0, 0.0
        best = (0.5, 0.0, 0.0, 0.0)
        for grid_index, threshold in enumerate(self.threshold_grid):
            tp = int(self.grid_tp[grid_index])
            fp = int(self.grid_fp[grid_index])
            fn = int(self.grid_fn[grid_index])
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            if f1 > best[1]:
                best = (float(threshold), f1, precision, recall)
        return best

    def get_metrics(self):
        """获取所有指标"""
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) if (
                                                                                                self.tp + self.tn + self.fp + self.fn) > 0 else 0

        hard_recall = self.hard_tp / self.hard_total if self.hard_total > 0 else 0

        auc = self.compute_auc()
        average_precision = self.compute_average_precision()
        best_threshold, best_threshold_f1, best_precision, best_recall = \
            self.compute_best_threshold()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'average_precision': average_precision,
            'best_threshold': best_threshold,
            'best_threshold_f1': best_threshold_f1,
            'best_threshold_precision': best_precision,
            'best_threshold_recall': best_recall,
            'hard_recall': hard_recall,
            'hard_total': self.hard_total,
            'hard_tp': self.hard_tp,
        }


# ==============================================================================
# 轻量级边界预测网络 (使用稀疏卷积)
# ==============================================================================

class SparseBoundaryNet(nn.Module):
    """
    使用稀疏卷积的边界预测网络

    基于体素化点云，内存效率高
    """

    def __init__(self, in_channels=6, channels=32, num_classes=2):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # 输入层
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels),
            nn.ReLU(inplace=True)
        )

        # 编码器 - 下采样
        self.enc1 = spconv.SparseSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm2'),
            norm_fn(channels),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm2'),
            norm_fn(channels),
            nn.ReLU(inplace=True)
        )

        self.down1 = spconv.SparseSequential(
            spconv.SparseConv3d(channels, channels * 2, kernel_size=2, stride=2, bias=False, indice_key='spconv1'),
            norm_fn(channels * 2),
            nn.ReLU(inplace=True)
        )

        self.enc2 = spconv.SparseSequential(
            spconv.SubMConv3d(channels * 2, channels * 2, kernel_size=3, padding=1, bias=False, indice_key='subm3'),
            norm_fn(channels * 2),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(channels * 2, channels * 2, kernel_size=3, padding=1, bias=False, indice_key='subm3'),
            norm_fn(channels * 2),
            nn.ReLU(inplace=True)
        )

        self.down2 = spconv.SparseSequential(
            spconv.SparseConv3d(channels * 2, channels * 4, kernel_size=2, stride=2, bias=False, indice_key='spconv2'),
            norm_fn(channels * 4),
            nn.ReLU(inplace=True)
        )

        self.enc3 = spconv.SparseSequential(
            spconv.SubMConv3d(channels * 4, channels * 4, kernel_size=3, padding=1, bias=False, indice_key='subm4'),
            norm_fn(channels * 4),
            nn.ReLU(inplace=True),
        )

        # 解码器 - 上采样
        self.up2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(channels * 4, channels * 2, kernel_size=2, bias=False, indice_key='spconv2'),
            norm_fn(channels * 2),
            nn.ReLU(inplace=True)
        )

        self.dec2 = spconv.SparseSequential(
            spconv.SubMConv3d(channels * 4, channels * 2, kernel_size=3, padding=1, bias=False, indice_key='subm3_dec'),
            norm_fn(channels * 2),
            nn.ReLU(inplace=True)
        )

        self.up1 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(channels * 2, channels, kernel_size=2, bias=False, indice_key='spconv1'),
            norm_fn(channels),
            nn.ReLU(inplace=True)
        )

        self.dec1 = spconv.SparseSequential(
            spconv.SubMConv3d(channels * 2, channels, kernel_size=3, padding=1, bias=False, indice_key='subm2_dec'),
            norm_fn(channels),
            nn.ReLU(inplace=True)
        )

        # 边界预测头
        self.boundary_head = nn.Sequential(
            nn.Linear(channels, channels),
            norm_fn(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_classes)
        )
        # Fractional target is kept as an auxiliary distance-to-interface
        # signal. It no longer changes which points count as hard positives.
        self.distance_head = nn.Sequential(
            nn.Linear(channels, channels),
            norm_fn(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def forward(self, voxel_feats):
        """
        前向传播

        Args:
            voxel_feats: SparseConvTensor

        Returns:
            boundary_logits: (M, 2) 体素级边界预测
        """
        # 编码
        x = self.input_conv(voxel_feats)
        enc1 = self.enc1(x)

        x = self.down1(enc1)
        enc2 = self.enc2(x)

        x = self.down2(enc2)
        x = self.enc3(x)

        # 解码
        x = self.up2(x)
        x = x.replace_feature(torch.cat([x.features, enc2.features], dim=1))
        x = self.dec2(x)

        x = self.up1(x)
        x = x.replace_feature(torch.cat([x.features, enc1.features], dim=1))
        x = self.dec1(x)

        # 边界预测
        boundary_logits = self.boundary_head(x.features)
        boundary_distance_logits = self.distance_head(x.features).squeeze(1)

        return boundary_logits, boundary_distance_logits


class BoundaryOnlyModel(nn.Module):
    """
    边界预测专用模型
    """

    def __init__(self, boundary_cfg=None, **kwargs):
        super().__init__()

        if boundary_cfg is None:
            boundary_cfg = {}

        self.channels = boundary_cfg.get('width', 32)
        self.in_channels = boundary_cfg.get('in_channels', 6)
        self.threshold = boundary_cfg.get('threshold', 0.5)
        self.loss_weight = boundary_cfg.get('loss_weight', 1.0)
        self.hard_boundary_threshold = float(
            boundary_cfg.get('hard_boundary_threshold', 0.999)
        )
        self.hard_loss_weight = float(
            boundary_cfg.get('hard_loss_weight', 1.0)
        )
        self.soft_distance_weight = float(
            boundary_cfg.get('soft_distance_weight', 0.25)
        )
        self.hard_pos_weight = float(
            boundary_cfg.get('hard_pos_weight', 6.0)
        )
        self.focal_gamma = float(boundary_cfg.get('focal_gamma', 2.0))
        self.tree_only_loss = bool(boundary_cfg.get('tree_only_loss', True))

        self.boundary_net = SparseBoundaryNet(
            in_channels=self.in_channels,
            channels=self.channels,
            num_classes=2
        )

        self._print_config()

    def _print_config(self):
        print(f"\n{'=' * 60}")
        print(f"边界预测专项训练模型 (稀疏卷积版)")
        print(f"{'=' * 60}")
        print(f"通道数: {self.channels}")
        print(f"输入通道数: {self.in_channels}")
        print(f"边界阈值: {self.threshold}")
        print(f"{'=' * 60}\n")

    def forward(self, input_dict, epoch=0, mode='loss'):
        """前向传播"""
        # 准备体素特征
        voxel_feats = self._prepare_voxel_feats(input_dict)

        # 边界预测
        boundary_logits, distance_logits = self.boundary_net(voxel_feats)
        num_voxels = voxel_feats.features.shape[0]
        require_point_shapes(
            num_voxels,
            voxel_boundary_logits=(boundary_logits, 2),
            voxel_distance_logits=(distance_logits, 1),
        )

        # 映射回点
        p2v_map = self._get_p2v_map(input_dict, voxel_feats)
        point_boundary_logits = boundary_logits[p2v_map]  # (N, 2)
        point_distance_logits = distance_logits[p2v_map]

        boundary_probs = F.softmax(point_boundary_logits, dim=1)[:, 1]

        ret = {
            'boundary_logits': point_boundary_logits,
            'boundary_probs': boundary_probs,
            'boundary_distance': torch.sigmoid(point_distance_logits),
        }

        if mode == 'loss':
            losses = self._compute_loss(ret, input_dict)
            return losses.get('loss', torch.tensor(0.0)), losses
        else:
            return ret

    def _prepare_voxel_feats(self, input_dict):
        """准备体素特征"""
        for name in ('voxel_coords', 'feats'):
            if input_dict.get(name) is None or not hasattr(input_dict[name], 'shape'):
                raise BoundaryIntegrityError(f"Batch requires an array named {name}.")
        voxel_coords = input_dict['voxel_coords']
        feats = input_dict['feats']
        p2v_map = input_dict.get('p2v_map')
        if len(voxel_coords.shape) != 2 or voxel_coords.shape[1] != 4:
            raise BoundaryIntegrityError("voxel_coords must have shape (M, 4).")
        num_voxels = voxel_coords.shape[0]
        point_map = point_to_voxel_map(input_dict, num_voxels)
        num_points = input_dict['coords_float'].shape[0]

        # 获取体素特征
        if p2v_map is not None and len(p2v_map.shape) == 2:
            require_point_shapes(num_points, feats=(feats, 2))
            validate_voxel_point_table(p2v_map, point_map, num_points, num_voxels)
            M, K = p2v_map.shape
            C = feats.shape[1]

            valid_mask = p2v_map >= 0
            valid_indices = p2v_map[valid_mask].long()
            counts = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)

            valid_feats = feats[valid_indices]
            voxel_indices = torch.arange(M, device=feats.device).unsqueeze(1).expand(-1, K)[valid_mask]

            voxel_feats_dense = torch.zeros(M, C, device=feats.device, dtype=feats.dtype)
            voxel_feats_dense.scatter_add_(0, voxel_indices.unsqueeze(1).expand(-1, C), valid_feats)
            voxel_feats_dense = voxel_feats_dense / counts
        else:
            # Without the voxel-to-point table this path requires already
            # aggregated voxel features; slicing point features is not valid.
            validate_voxel_feature_domain(
                feats, point_map, num_points, num_voxels,
                layout=input_dict.get('feats_layout'),
            )
            voxel_feats_dense = feats

        # 添加归一化坐标
        C = voxel_feats_dense.shape[1]
        if C == 3:  # RGB
            voxel_xyz = voxel_coords[:, 1:].float()
            if voxel_xyz.numel() > 0:
                voxel_xyz_min = voxel_xyz.min(0, keepdim=True)[0]
                voxel_xyz_max = voxel_xyz.max(0, keepdim=True)[0]
                voxel_xyz_range = (voxel_xyz_max - voxel_xyz_min).clamp(min=1)
                voxel_xyz_norm = (voxel_xyz - voxel_xyz_min) / voxel_xyz_range
                voxel_feats_dense = torch.cat([voxel_feats_dense, voxel_xyz_norm], dim=1)

        spatial_shape = input_dict.get('spatial_shape', [512, 512, 512])
        if isinstance(spatial_shape, np.ndarray):
            spatial_shape = spatial_shape.tolist()

        batch_size = input_dict.get('batch_size', int(voxel_coords[:, 0].max()) + 1)

        voxel_feats = spconv.SparseConvTensor(
            voxel_feats_dense,
            voxel_coords.int(),
            spatial_shape,
            batch_size
        )

        return voxel_feats

    def _get_p2v_map(self, input_dict, voxel_feats):
        """Use a verified point-to-voxel vector; never invent or clip indices."""
        return point_to_voxel_map(
            input_dict, voxel_feats.features.shape[0]
        ).long()

    def _compute_loss(self, ret, input_dict):
        """计算边界损失"""
        losses = {}

        boundary_logits = ret['boundary_logits']
        boundary_labels = input_dict.get('boundary_labels', None)

        if boundary_labels is not None:
            boundary_distance = ret['boundary_distance']
            num_points = input_dict['coords_float'].shape[0]
            require_point_shapes(
                num_points,
                boundary_logits=(boundary_logits, 2),
                boundary_distance=(boundary_distance, 1),
                boundary_labels=(boundary_labels, 1),
            )
            if self.tree_only_loss:
                instance_labels = input_dict.get('instance_labels')
                require_point_shapes(num_points, instance_labels=(instance_labels, 1))
            boundary_labels = boundary_labels.clamp(0, 1)
            boundary_targets = (
                boundary_labels >= self.hard_boundary_threshold
            ).long()
            valid_mask = torch.ones_like(boundary_targets, dtype=torch.bool)
            if self.tree_only_loss:
                valid_mask = instance_labels >= 0

            if not valid_mask.any():
                losses['loss'] = boundary_logits.sum() * 0.0
                return losses

            logits_valid = boundary_logits[valid_mask]
            targets_valid = boundary_targets[valid_mask]
            class_weight = torch.tensor(
                [1.0, self.hard_pos_weight],
                device=boundary_logits.device,
                dtype=boundary_logits.dtype,
            )
            ce = F.cross_entropy(
                logits_valid,
                targets_valid,
                weight=class_weight,
                reduction='none',
            )
            if self.focal_gamma > 0:
                target_prob = F.softmax(logits_valid, dim=1).gather(
                    1, targets_valid[:, None]
                ).squeeze(1)
                ce = ce * (1.0 - target_prob).pow(self.focal_gamma)
            hard_loss = ce.mean()
            distance_loss = F.smooth_l1_loss(
                boundary_distance[valid_mask],
                boundary_labels[valid_mask],
            )
            boundary_loss = (
                self.hard_loss_weight * hard_loss
                + self.soft_distance_weight * distance_loss
            )
            losses['hard_boundary_loss'] = hard_loss
            losses['soft_distance_loss'] = distance_loss
            losses['boundary_loss'] = boundary_loss * self.loss_weight
            losses['loss'] = losses['boundary_loss']

            # 统计信息
            with torch.no_grad():
                pred = logits_valid.argmax(1)
                acc = (pred == targets_valid).float().mean()
                losses['accuracy'] = acc
        else:
            raise BoundaryIntegrityError("Loss mode requires point-aligned boundary_labels.")

        return losses


# ==============================================================================
# 日志记录器
# ==============================================================================

class TrainingLogger:
    def __init__(self, work_dir, name='boundary_train'):
        import logging
        self.work_dir = work_dir

        self.logger = logging.getLogger(f'BoundaryOnly_{name}')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)

        log_file = os.path.join(work_dir, f'{name}.log')
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(file_handler)

        self.csv_file = os.path.join(work_dir, 'boundary_training_history.csv')
        self.csv_initialized = False

        self.info(f"日志文件: {log_file}")

    def info(self, msg):
        self.logger.info(msg)

    def write_csv_row(self, row_dict):
        fieldnames = list(row_dict.keys())
        if not os.path.exists(self.csv_file):
            self.csv_initialized = False
        if not self.csv_initialized:
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            self.csv_initialized = True
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_dict)


# ==============================================================================
# 训练和验证
# ==============================================================================

def prepare_batch(batch, device='cuda'):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def train_epoch(model, dataloader, optimizer, epoch, logger, scaler=None, use_amp=False):
    model.train()
    total_loss = 0
    total_acc = 0
    valid_batches = 0

    for i, batch in enumerate(dataloader):
        if batch is None:
            continue

        try:
            batch = prepare_batch(batch)

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                with autocast():
                    loss, log_vars = model(batch, epoch=epoch, mode='loss')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, log_vars = model(batch, epoch=epoch, mode='loss')
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            if 'accuracy' in log_vars:
                total_acc += log_vars['accuracy'].item()
            valid_batches += 1

            if (i + 1) % 5 == 0:
                avg_loss = total_loss / valid_batches
                avg_acc = total_acc / valid_batches if valid_batches > 0 else 0
                logger.info(f"  [{i + 1}/{len(dataloader)}] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

            # 清理显存
            torch.cuda.empty_cache()

        except BoundaryIntegrityError as e:
            raise BoundaryIntegrityError(f"Training batch {i}: {e}") from e
        except Exception as e:
            logger.info(f"⚠️ Training error at batch {i}: {e}")
            torch.cuda.empty_cache()
            continue

    if valid_batches == 0:
        raise RuntimeError(
            "No boundary-training batch completed successfully. "
            "Inspect the preceding batch errors instead of saving an empty run."
        )
    return total_loss / valid_batches


def validate(model, dataloader, epoch, logger, threshold=0.5):
    model.eval()
    metrics_calc = BoundaryMetrics(
        threshold=threshold,
        hard_threshold=model.hard_boundary_threshold,
    )
    domain_metrics = {}
    total_loss = 0
    valid_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue

            try:
                batch = prepare_batch(batch)
                loss, log_vars = model(batch, epoch=epoch, mode='loss')
                ret = model(batch, epoch=epoch, mode='predict')

                boundary_probs = ret['boundary_probs']
                boundary_labels = batch['boundary_labels']
                num_points = batch['coords_float'].shape[0]
                require_point_shapes(
                    num_points,
                    boundary_probs=(boundary_probs, 1),
                    boundary_labels=(boundary_labels, 1),
                )

                valid_mask = None
                if model.tree_only_loss:
                    instance_labels = batch.get('instance_labels')
                    require_point_shapes(num_points, instance_labels=(instance_labels, 1))
                    valid_mask = instance_labels >= 0
                metrics_calc.update(
                    boundary_probs,
                    boundary_labels,
                    valid_mask=valid_mask,
                )
                scan_name = str(batch.get('scan_ids', ['unknown'])[0]).lower()
                domain = next(
                    (name for name in ('culs', 'nibio', 'rmit', 'scion', 'tuwien')
                     if name in scan_name),
                    'unknown',
                )
                if domain not in domain_metrics:
                    domain_metrics[domain] = BoundaryMetrics(
                        threshold=threshold,
                        hard_threshold=model.hard_boundary_threshold,
                    )
                domain_metrics[domain].update(
                    boundary_probs,
                    boundary_labels,
                    valid_mask=valid_mask,
                )
                total_loss += loss.item()
                valid_batches += 1

                torch.cuda.empty_cache()

            except BoundaryIntegrityError as e:
                raise BoundaryIntegrityError(f"Validation batch {i}: {e}") from e
            except Exception as e:
                logger.info(f"⚠️ Validation error at batch {i}: {e}")
                torch.cuda.empty_cache()
                continue

    if valid_batches == 0:
        raise RuntimeError(
            "No boundary-validation batch completed successfully. "
            "Check the data mapping, CUDA memory and spconv installation."
        )
    avg_loss = total_loss / valid_batches
    metrics = metrics_calc.get_metrics()
    metrics['per_domain'] = {
        domain: calculator.get_metrics()
        for domain, calculator in sorted(domain_metrics.items())
    }
    domain_aps = [
        values['average_precision']
        for domain, values in metrics['per_domain'].items()
        if domain != 'unknown'
    ]
    metrics['macro_average_precision'] = (
        float(np.mean(domain_aps)) if domain_aps else 0.0
    )

    return avg_loss, metrics


def print_metrics(metrics, logger):
    logger.info(f"\n{'=' * 50}")
    logger.info(f"📊 验证指标")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:    {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:     {metrics['auc']:.4f}")
    logger.info(f"  Average Prec:{metrics['average_precision']:.4f}")
    logger.info(
        f"  Best hard-F1:{metrics['best_threshold_f1']:.4f} "
        f"@ threshold={metrics['best_threshold']:.2f} "
        f"(P={metrics['best_threshold_precision']:.4f}, "
        f"R={metrics['best_threshold_recall']:.4f})"
    )
    logger.info(f"  Macro AP:    {metrics['macro_average_precision']:.4f}")
    for domain, values in metrics.get('per_domain', {}).items():
        logger.info(
            f"    {domain:7s}: AP={values['average_precision']:.4f}, "
            f"AUC={values['auc']:.4f}, F1={values['f1']:.4f}, "
            f"R={values['recall']:.4f}"
        )
    logger.info(f"  Hard Recall: {metrics['hard_recall']:.4f} ({metrics['hard_tp']}/{metrics['hard_total']})")
    logger.info(f"{'=' * 50}\n")


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='边界预测专项训练')
    parser.add_argument('--config', type=str, default='ginkgo_boundary_segment.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument(
        '--init', type=str, default='',
        help='Load compatible model weights only and start a new optimizer/run',
    )
    args = parser.parse_args()
    if args.resume and args.init:
        parser.error('--resume and --init are mutually exclusive')

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print("\n" + "🎯" * 30)
    print("边界预测专项训练 (稀疏卷积版)")
    print("🎯" * 30 + "\n")

    work_dir = Path(cfg['work_dir']) / 'boundary_only'
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(work_dir)
    logger.info(f"✓ Work directory: {work_dir}")
    logger.info(f"✓ Epochs: {args.epochs}, LR: {args.lr}")

    # 构建模型
    model = BoundaryOnlyModel(
        boundary_cfg=cfg['model'].get('boundary_cfg', {})
    ).cuda()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ 模型参数量: {n_params / 1e6:.2f}M")

    # 构建数据集
    from ginkgo_dataset import GinkgoDataset, build_ginkgo_dataloader

    train_cfg = {k: v for k, v in cfg['data']['train'].items() if k != 'type'}
    train_set = GinkgoDataset(**train_cfg, logger=logger)
    train_loader = build_ginkgo_dataloader(
        train_set,
        batch_size=cfg['dataloader']['train']['batch_size'],
        num_workers=cfg['dataloader']['train']['num_workers'],
        training=True
    )

    val_cfg = {k: v for k, v in cfg['data']['test'].items() if k != 'type'}
    val_set = GinkgoDataset(**val_cfg, logger=logger)
    val_loader = build_ginkgo_dataloader(
        val_set,
        batch_size=cfg['dataloader']['test']['batch_size'],
        num_workers=cfg['dataloader']['test']['num_workers'],
        training=False
    )

    logger.info(f"✓ 训练集: {len(train_set)} samples")
    logger.info(f"✓ 验证集: {len(val_set)} samples")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    use_amp = cfg.get('fp16', False)
    scaler = GradScaler() if use_amp else None

    # 加载检查点
    start_epoch = 0
    selection_metric = cfg['model'].get('boundary_cfg', {}).get(
        'selection_metric', 'average_precision'
    )
    best_score = -1.0
    if args.init:
        logger.info(f"初始化模型权重: {args.init}")
        try:
            checkpoint = torch.load(
                args.init, map_location='cpu', weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(args.init, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"初始化完成: missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    if args.resume:
        logger.info(f"加载检查点: {args.resume}")
        try:
            checkpoint = torch.load(
                args.resume, map_location='cpu', weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(f"Missing checkpoint keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.info(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
        if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if isinstance(checkpoint, dict) and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if isinstance(checkpoint, dict):
            best_score = float(checkpoint.get(
                'best_score', checkpoint.get('metrics', {}).get(selection_metric, -1.0)
            ))

    # 训练循环

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch [{epoch + 1}/{args.epochs}]")
        logger.info(f"{'=' * 50}")

        train_loss = train_epoch(model, train_loader, optimizer, epoch, logger, scaler, use_amp)
        logger.info(f"训练 Loss: {train_loss:.4f}")

        scheduler.step()

        val_loss, metrics = validate(model, val_loader, epoch, logger, args.threshold)
        logger.info(f"验证 Loss: {val_loss:.4f}")
        print_metrics(metrics, logger)

        # 记录
        csv_row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'average_precision': metrics['average_precision'],
            'macro_average_precision': metrics['macro_average_precision'],
            'best_threshold': metrics['best_threshold'],
            'best_threshold_f1': metrics['best_threshold_f1'],
            'hard_recall': metrics['hard_recall'],
        }
        logger.write_csv_row(csv_row)

        # 保存最佳模型
        current_score = float(metrics[selection_metric])
        if current_score > best_score:
            best_score = current_score
            save_path = work_dir / 'best_boundary_model.pth'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_f1': metrics['f1'],
                'best_score': best_score,
                'selection_metric': selection_metric,
                'metrics': metrics,
                'boundary_cfg': cfg['model'].get('boundary_cfg', {}),
            }, save_path)
            logger.info(
                f"🎉 新最佳模型! {selection_metric}={best_score:.4f}, "
                f"hard-F1={metrics['f1']:.4f}"
            )

        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_path = work_dir / f'boundary_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_f1': metrics['f1'],
                'best_score': best_score,
                'selection_metric': selection_metric,
                'metrics': metrics,
                'boundary_cfg': cfg['model'].get('boundary_cfg', {}),
            }, save_path)

    logger.info("\n" + "🎉" * 20)
    logger.info(
        f"训练完成! 最佳 {selection_metric}: {best_score:.4f}"
    )
    logger.info("🎉" * 20 + "\n")


if __name__ == '__main__':
    main()
