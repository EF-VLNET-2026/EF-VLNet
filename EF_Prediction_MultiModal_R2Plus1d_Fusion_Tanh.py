#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
import time
from tqdm import tqdm
import argparse
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
import random
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from biobert_extractor import BioBERTFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def latexify():
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.usetex': False,
        'figure.figsize': (3, 3),
        'axes.linewidth': 0.5,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.major.size': 3,
        'ytick.minor.size': 1.5,
        'legend.frameon': False,
        'legend.numpoints': 1
    })


class EchoNetStyleVisualizer:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        latexify()

    def create_scatter_plot(self, predictions, targets, title, filename, phase='test'):
        fig = plt.figure(figsize=(3, 3))
        lower = min(targets.min(), predictions.min())
        upper = max(targets.max(), predictions.max())

        plt.scatter(targets, predictions, color="k", s=1, edgecolor=None, zorder=2)
        plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
        plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("Actual EF (%)")
        plt.ylabel("Predicted EF (%)")
        plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
        plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
        plt.tight_layout()

        pdf_path = self.output_dir / filename
        plt.savefig(pdf_path)
        plt.close(fig)
        logger.info(f"Scatter plot saved: {pdf_path}")

        correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0
        r2 = r2_score(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))

        return correlation, r2, mae, rmse

    def create_roc_plot(self, predictions, targets, title, filename, phase='test'):
        fig = plt.figure(figsize=(3, 3))
        plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")

        for thresh in [35, 40, 45, 50]:
            binary_targets = (targets > thresh).astype(int)
            try:
                fpr, tpr, _ = roc_curve(binary_targets, predictions)
                auc = roc_auc_score(binary_targets, predictions)
                plt.plot(fpr, tpr, label=f"EF > {thresh} (AUC: {auc:.3f})")
            except:
                continue

        plt.axis([-0.01, 1.01, -0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()

        pdf_path = self.output_dir / filename
        plt.savefig(pdf_path)
        plt.close(fig)
        logger.info(f"ROC curve saved: {pdf_path}")


class EFNormalizer:
    def __init__(self):
        self.ef_min = None
        self.ef_max = None
        self.ef_mean = None
        self.ef_std = None

    def fit(self, ef_values: np.ndarray):
        self.ef_min = float(ef_values.min())
        self.ef_max = float(ef_values.max())
        self.ef_mean = float(ef_values.mean())
        self.ef_std = float(ef_values.std())
        logger.info(f"  EF normalization parameters:")
        logger.info(f"    Range: [{self.ef_min:.1f}, {self.ef_max:.1f}]")
        logger.info(f"    Mean +/- Std: {self.ef_mean:.1f}±{self.ef_std:.1f}")

    def normalize(self, ef_value: float) -> float:
        return (ef_value - self.ef_min) / (self.ef_max - self.ef_min + 1e-8)

    def denormalize(self, normalized_value: float) -> float:
        return normalized_value * (self.ef_max - self.ef_min) + self.ef_min


class RobustTemporalFeatureExtractor:

    def __init__(self):
        self.blood_flow_encoder = LabelEncoder()
        self.encoder_fitted = False

    def fit_encoder(self, all_blood_flows: List[str]):
        unique_flows = sorted(set(all_blood_flows))
        self.blood_flow_encoder.fit(unique_flows)
        self.encoder_fitted = True
        logger.info(f"  Blood_Flow encoder fitted: {len(unique_flows)} categories")

    def extract_features(self, chamber_data: pd.DataFrame) -> np.ndarray:
        features_list = []

        area = chamber_data['Area_pixels'].values.astype(float)
        phase = chamber_data['Phase'].values
        blood_flow = chamber_data['Blood_Flow'].values

        area_mean = area.mean()
        area_std = area.std() + 1e-8
        area_normalized = (area - area_mean) / area_std
        features_list.append(area_normalized.reshape(-1, 1))

        phase_encoded = (phase == 'systole').astype(float)
        features_list.append(phase_encoded.reshape(-1, 1))

        if self.encoder_fitted:
            try:
                blood_flow_encoded = self.blood_flow_encoder.transform(blood_flow)
            except ValueError:
                blood_flow_encoded = np.zeros(len(blood_flow))
        else:
            blood_flow_encoded = np.zeros(len(blood_flow))

        n_classes = len(self.blood_flow_encoder.classes_) if self.encoder_fitted else 1
        blood_flow_normalized = blood_flow_encoded.astype(float) / max(n_classes - 1, 1)
        features_list.append(blood_flow_normalized.reshape(-1, 1))

        area_velocity = np.gradient(area)
        area_velocity = (area_velocity - area_velocity.mean()) / (area_velocity.std() + 1e-8)
        features_list.append(area_velocity.reshape(-1, 1))

        area_acceleration = np.gradient(area_velocity)
        area_acceleration = np.clip(area_acceleration, -5, 5)
        area_acceleration = (area_acceleration - area_acceleration.mean()) / (area_acceleration.std() + 1e-8)
        features_list.append(area_acceleration.reshape(-1, 1))

        area_change_rate = np.zeros_like(area)
        for i in range(1, len(area)):
            if abs(area[i-1]) > 1e-3:
                area_change_rate[i] = (area[i] - area[i-1]) / (abs(area[i-1]) + 1e-8)
        area_change_rate = np.clip(area_change_rate, -2, 2)
        features_list.append(area_change_rate.reshape(-1, 1))

        area_cumsum = np.cumsum(area - area_mean)
        area_cumsum = (area_cumsum - area_cumsum.mean()) / (area_cumsum.std() + 1e-8)
        features_list.append(area_cumsum.reshape(-1, 1))

        window = 5
        area_range = np.zeros_like(area)
        for i in range(len(area)):
            start = max(0, i - window)
            end = min(len(area), i + window + 1)
            area_range[i] = area[start:end].max() - area[start:end].min()
        area_range = (area_range - area_range.mean()) / (area_range.std() + 1e-8)
        features_list.append(area_range.reshape(-1, 1))

        features_list.append(phase_encoded.reshape(-1, 1))

        phase_change = np.zeros(len(phase))
        phase_change[1:] = (phase_encoded[1:] != phase_encoded[:-1]).astype(float)
        features_list.append(phase_change.reshape(-1, 1))

        phase_position = np.zeros(len(phase))
        current_phase_start = 0
        for i in range(1, len(phase)):
            if phase_encoded[i] != phase_encoded[i-1]:
                phase_length = i - current_phase_start
                if phase_length > 0:
                    phase_position[current_phase_start:i] = np.arange(phase_length) / phase_length
                current_phase_start = i
        phase_length = len(phase) - current_phase_start
        if phase_length > 0:
            phase_position[current_phase_start:] = np.arange(phase_length) / phase_length
        features_list.append(phase_position.reshape(-1, 1))

        features_list.append(blood_flow_normalized.reshape(-1, 1))
        blood_flow_phase = blood_flow_normalized * phase_encoded
        features_list.append(blood_flow_phase.reshape(-1, 1))

        area_rolling_mean = np.convolve(area, np.ones(window)/window, mode='same')
        area_rolling_mean = (area_rolling_mean - area_rolling_mean.mean()) / (area_rolling_mean.std() + 1e-8)
        features_list.append(area_rolling_mean.reshape(-1, 1))

        area_rolling_std = np.zeros_like(area)
        for i in range(len(area)):
            start = max(0, i - window)
            end = min(len(area), i + window + 1)
            area_rolling_std[i] = area[start:end].std()
        area_rolling_std = (area_rolling_std - area_rolling_std.mean()) / (area_rolling_std.std() + 1e-8)
        features_list.append(area_rolling_std.reshape(-1, 1))

        area_cv = np.zeros_like(area)
        for i in range(len(area)):
            start = max(0, i - window)
            end = min(len(area), i + window + 1)
            local_mean = area[start:end].mean()
            local_std = area[start:end].std()
            if abs(local_mean) > 1e-3:
                area_cv[i] = local_std / abs(local_mean)
        area_cv = np.clip(area_cv, 0, 5)
        area_cv = (area_cv - area_cv.mean()) / (area_cv.std() + 1e-8)
        features_list.append(area_cv.reshape(-1, 1))

        all_features = np.concatenate(features_list, axis=1)
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

        return all_features


class SimpleFeatureAugmenter:

    @staticmethod
    def add_gaussian_noise(features: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(features) * noise_std
            return features + noise
        return features


class OptimizedFourLayerDataset(Dataset):

    CHAMBERS = ['LA', 'LV', 'RA', 'RV']
    PHASES = ['t0_t1', 't1_t2']

    def __init__(
        self,
        base_dir: str,
        split: str = "TRAIN",
        max_frames: int = 100,
        use_ai_diagnosis: bool = True,
        use_phase_info: bool = True,
        use_temporal: bool = True,
        use_video: bool = True,
        model_type: str = "r2plus1d",
        biobert_model_path: Optional[str] = None,
        feature_scalers: Optional[Dict] = None,
        feature_extractor: Optional[RobustTemporalFeatureExtractor] = None,
        ef_normalizer: Optional[EFNormalizer] = None,
        augment: bool = False,
    ):
        super().__init__()

        self.base_dir = Path(base_dir)
        self.split = split.upper()
        self.max_frames = max_frames
        self.use_ai_diagnosis = use_ai_diagnosis
        self.use_phase_info = use_phase_info
        self.use_temporal = use_temporal
        self.use_video = use_video
        self.model_type = model_type.lower()
        self.feature_scalers = feature_scalers or {}
        self.feature_extractor = feature_extractor
        self.ef_normalizer = ef_normalizer
        self.augment = augment and (split == "TRAIN")

        logger.info("=" * 80)
        logger.info(f"Initializing dataset: split={self.split}, model_type={self.model_type}")
        logger.info("=" * 80)

        self.loading_stats = {
            'ai_features': {'total': 0, 'loaded': 0, 'missing': 0},
            'phase_info': {'total': 0, 'loaded': 0, 'missing': 0},
            'temporal': {'total': 0, 'loaded': 0, 'missing': 0},
            self.model_type: {'total': 0, 'loaded': 0, 'missing': 0, 'all_zero': 0}
        }

        self.biobert_extractor = None
        self.ai_features = {}
        if use_ai_diagnosis:
            try:
                logger.info("\nInitializing BioBERT...")
                if biobert_model_path and Path(biobert_model_path).exists():
                    self.biobert_extractor = BioBERTFeatureExtractor(model_path=biobert_model_path)
                else:
                    self.biobert_extractor = BioBERTFeatureExtractor()
                self.ai_features = self._extract_ai_diagnosis()
            except Exception as e:
                logger.warning(f"BioBERT initialization failed: {e}")

        self.phase_data = self._load_phase_data() if use_phase_info else {}
        self.temporal_df = self._load_temporal_features() if use_temporal else pd.DataFrame()
        self.temporal_dict = self._preprocess_temporal_df()
        self.video_features = self._load_video_features() if use_video else {}

        self._load_ef_labels()
        self._print_loading_stats()

        logger.info(f"\nDataset loaded: {len(self.valid_samples)} samples")

    def _extract_ai_diagnosis(self) -> Dict[str, np.ndarray]:
        ai_diagnosis_file = self.base_dir / "ai_reports.json"
        if not ai_diagnosis_file.exists():
            self.loading_stats['ai_features']['total'] = 0
            return {}

        with open(ai_diagnosis_file, 'r', encoding='utf-8') as f:
            ai_reports_data = json.load(f)

        self.loading_stats['ai_features']['total'] = len(ai_reports_data)

        ai_features = {}
        video_ids = []
        texts = []

        for video_id, report in ai_reports_data.items():
            video_id = str(video_id).replace('.avi', '')
            if not report or pd.isna(report):
                self.loading_stats['ai_features']['missing'] += 1
                continue

            if isinstance(report, str):
                text = " ".join([p.strip() for p in report.split('.') if p.strip()])
            else:
                text = str(report)

            if text.strip():
                video_ids.append(video_id)
                texts.append(text)

        if texts:
            features = self.biobert_extractor.extract_features_batch(texts, batch_size=32)

            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()

            for video_id, feat in zip(video_ids, features):
                ai_features[video_id] = feat
                self.loading_stats['ai_features']['loaded'] += 1

        return ai_features

    def _load_phase_data(self) -> Dict[str, Dict]:
        phase_data = {}
        for phase in self.PHASES:
            csv_file = self.base_dir / f"{phase}.csv"
            if not csv_file.exists():
                continue

            df = pd.read_csv(csv_file)
            self.loading_stats['phase_info']['total'] += len(df)

            if phase == 't0_t1':
                required_cols = ['FileName', 'T0', 'T1']
                t_start_col, t_end_col = 'T0', 'T1'
            else:
                required_cols = ['FileName', 'T1', 'T2']
                t_start_col, t_end_col = 'T1', 'T2'

            df = df.dropna(subset=required_cols)

            for _, row in df.iterrows():
                try:
                    video_id = str(row['FileName']).replace('.avi', '')
                    if video_id not in phase_data:
                        phase_data[video_id] = {}
                    phase_data[video_id][phase] = {
                        'start': int(row[t_start_col]),
                        'end': int(row[t_end_col])
                    }
                    self.loading_stats['phase_info']['loaded'] += 1
                except:
                    self.loading_stats['phase_info']['missing'] += 1
                    continue

        return phase_data

    def _load_temporal_features(self) -> pd.DataFrame:
        csv_files = sorted(self.base_dir.glob("heart_chamber_flow_analysis_part*.csv"))
        if not csv_files:
            return pd.DataFrame()

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                self.loading_stats['temporal']['total'] += len(df)
            except:
                pass

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            if 'Video_Name' in combined_df.columns:
                combined_df['Video_Name'] = combined_df['Video_Name'].str.replace('.avi', '')
            self.loading_stats['temporal']['loaded'] = len(combined_df)
            return combined_df
        return pd.DataFrame()

    def _preprocess_temporal_df(self) -> Dict:
        if self.temporal_df.empty:
            return {}

        logger.info("\nPreprocessing temporal data...")
        temporal_dict = {}

        grouped = self.temporal_df.groupby(['Video_Name', 'Chamber'])

        for (video_id, chamber), group in tqdm(grouped, desc="  Grouping temporal data", leave=False):
            video_id = str(video_id).replace('.avi', '')
            if video_id not in temporal_dict:
                temporal_dict[video_id] = {}
            temporal_dict[video_id][chamber] = group.sort_values('Frame_Index').reset_index(drop=True)

        logger.info(f"  Preprocessing complete: {len(temporal_dict)} videos")

        del self.temporal_df
        self.temporal_df = pd.DataFrame()

        return temporal_dict

    def _load_video_features(self) -> Dict[str, torch.Tensor]:

        video_features = {}

        logger.info(f"\nLoading {self.model_type.upper()} features...")

        for chamber in self.CHAMBERS:
            for phase in self.PHASES:
                phase_dir = phase.replace('_', '')

                if self.model_type == 'r3d':
                    base_path = f"{chamber.lower()}_r3d_{phase_dir}"
                else:
                    base_path = f"{chamber.lower()}_{self.model_type}_{phase_dir}"

                feature_file = None
                for epoch_num in [0, 1, 2, 3, 4, 5]:
                    candidate = self.base_dir / base_path / "features" / f"features_{self.split}_epoch_{epoch_num}.npz"

                    if candidate.exists():
                        feature_file = candidate
                        logger.info(f"  Found epoch_{epoch_num} file: {chamber}_{phase}")
                        break

                if feature_file and feature_file.exists():
                    try:
                        data = np.load(feature_file, allow_pickle=True)
                        self.loading_stats[self.model_type]['total'] += len(data['filenames'])

                        features_array = data['features']
                        feature_sum = np.abs(features_array).sum()

                        if feature_sum < 1e-6:
                            logger.warning(f"  Warning: {chamber}_{phase}: Features all zero! (sum={feature_sum:.2e})")
                            self.loading_stats[self.model_type]['all_zero'] += len(data['filenames'])
                        else:
                            logger.info(f"  {chamber}_{phase}: Features non-zero (sum={feature_sum:.2e})")

                        for video_id, feat in zip(data['filenames'], data['features']):
                            video_id = str(video_id)
                            if video_id not in video_features:
                                video_features[video_id] = {}
                            video_features[video_id][f"{chamber}_{phase}"] = torch.from_numpy(feat).float()
                            self.loading_stats[self.model_type]['loaded'] += 1

                    except Exception as e:
                        logger.warning(f"  Error: {chamber}_{phase}: Load failed - {e}")
                else:
                    logger.warning(f"  Error: {chamber}_{phase}: File not found (tried epoch_0 to epoch_5)")

        return video_features

    def _load_ef_labels(self):
        file_list_path = self.base_dir / "FileList.csv"
        df = pd.read_csv(file_list_path)
        df = df[df['Split'] == self.split].dropna(subset=['EF'])
        df['video_id'] = df['FileName'].str.replace('.avi', '')

        self.ef_labels = {}
        for _, row in df.iterrows():
            self.ef_labels[row['video_id']] = float(row['EF'])

        self.valid_samples = sorted(self.ef_labels.keys())

    def _print_loading_stats(self):
        logger.info("\n" + "=" * 60)
        logger.info("Data loading statistics:")
        logger.info("=" * 60)

        for feature_type, stats in self.loading_stats.items():
            total = stats['total']
            loaded = stats['loaded']
            missing = stats['missing']
            if total > 0:
                success_rate = (loaded / total) * 100
                logger.info(f"  {feature_type}:")
                logger.info(f"    - Total: {total}")
                logger.info(f"    - Loaded: {loaded} ({success_rate:.1f}%)")
                if missing > 0:
                    logger.info(f"    - Missing/Failed: {missing}")

                if 'all_zero' in stats:
                    all_zero = stats['all_zero']
                    if all_zero > 0:
                        zero_rate = (all_zero / loaded) * 100 if loaded > 0 else 0
                        logger.error(f"    Warning: All-zero features: {all_zero} ({zero_rate:.1f}%)")

        logger.info("=" * 60)

    def _get_temporal_sequence(self, video_id: str, chamber: str) -> Optional[torch.Tensor]:
        if not self.temporal_dict or self.feature_extractor is None:
            return None

        if video_id not in self.temporal_dict:
            return None
        if chamber not in self.temporal_dict[video_id]:
            return None

        chamber_data = self.temporal_dict[video_id][chamber]

        if chamber_data.empty:
            return None

        try:
            features = self.feature_extractor.extract_features(chamber_data)
            seq_tensor = torch.from_numpy(features).float()

            T, F = seq_tensor.shape
            if T > self.max_frames:
                seq_tensor = seq_tensor[:self.max_frames, :]
            elif T < self.max_frames:
                padding = torch.zeros(self.max_frames - T, F)
                seq_tensor = torch.cat([seq_tensor, padding], dim=0)

            scaler_key = f"temporal_{chamber}"
            if scaler_key in self.feature_scalers:
                seq_np = self.feature_scalers[scaler_key].transform(seq_tensor.numpy())
                seq_tensor = torch.from_numpy(seq_np).float()

            return seq_tensor
        except Exception as e:
            return None

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        video_id = self.valid_samples[idx]
        ef = self.ef_labels[video_id]

        if self.ef_normalizer is not None:
            ef = self.ef_normalizer.normalize(ef)

        ai_feat = None
        if self.use_ai_diagnosis and video_id in self.ai_features:
            ai_feat_data = self.ai_features[video_id]
            if isinstance(ai_feat_data, torch.Tensor):
                ai_feat = ai_feat_data.float()
            else:
                ai_feat = torch.from_numpy(ai_feat_data).float()

        phase_feat = None
        if self.use_phase_info and video_id in self.phase_data:
            phase_values = []
            for p in self.PHASES:
                if p in self.phase_data[video_id]:
                    info = self.phase_data[video_id][p]
                    phase_values.extend([
                        info['start'], info['end'],
                        info['end'] - info['start'],
                        (info['start'] + info['end']) / 2.0
                    ])
                else:
                    phase_values.extend([0.0, 0.0, 0.0, 0.0])
            phase_feat = torch.tensor(phase_values, dtype=torch.float32)

        temporal_feats = {}
        if self.use_temporal:
            for chamber in self.CHAMBERS:
                seq = self._get_temporal_sequence(video_id, chamber)
                if seq is not None:
                    temporal_feats[chamber] = seq

        video_feat = None
        if self.use_video and video_id in self.video_features:
            video_list = []
            for chamber in self.CHAMBERS:
                for phase in self.PHASES:
                    key = f"{chamber}_{phase}"
                    if key in self.video_features[video_id]:
                        video_list.append(self.video_features[video_id][key])
                    else:
                        video_list.append(torch.zeros(128))
            video_feat = torch.stack(video_list, dim=0)

        return {
            'video_id': video_id,
            'ai_feat': ai_feat,
            'phase_feat': phase_feat,
            'temporal_feats': temporal_feats,
            'video_feat': video_feat,
            'ef': torch.tensor(ef, dtype=torch.float32)
        }


def collate_fn(batch):
    video_ids = [item['video_id'] for item in batch]
    efs = torch.stack([item['ef'] for item in batch])

    ai_feats = torch.stack([
        item['ai_feat'] if item['ai_feat'] is not None else torch.zeros(768)
        for item in batch
    ])

    phase_feats = torch.stack([
        item['phase_feat'] if item['phase_feat'] is not None else torch.zeros(8)
        for item in batch
    ])

    temporal_feats = {}
    for chamber in ['LA', 'LV', 'RA', 'RV']:
        seqs = []
        for item in batch:
            if chamber in item['temporal_feats']:
                seq = item['temporal_feats'][chamber]
                if seq.shape[1] != 16:
                    if seq.shape[1] < 16:
                        padding = torch.zeros(seq.shape[0], 16 - seq.shape[1])
                        seq = torch.cat([seq, padding], dim=1)
                    else:
                        seq = seq[:, :16]
                seqs.append(seq)
            else:
                seqs.append(torch.zeros(100, 16))
        temporal_feats[chamber] = torch.stack(seqs)

    video_feats = torch.stack([
        item['video_feat'] if item['video_feat'] is not None else torch.zeros(8, 128)
        for item in batch
    ])

    return {
        'video_ids': video_ids,
        'ai_feats': ai_feats,
        'phase_feats': phase_feats,
        'temporal_feats': temporal_feats,
        'video_feats': video_feats,
        'efs': efs
    }


class AttentionFusion(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.15):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        return self.norm(x + attn_out)


class SimplifiedTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.30):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.25 if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.15,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim * 2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout / 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout1(lstm_out)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.norm(lstm_out + attn_out)
        lstm_out = self.dropout2(lstm_out)

        mean_pool = lstm_out.mean(dim=1)
        max_pool, _ = lstm_out.max(dim=1)
        pooled = mean_pool + max_pool

        out = self.fc(pooled)
        return out


class ImprovedFourLayerFusionModel(nn.Module):
    def __init__(
        self,
        ai_dim: int = 768,
        phase_dim: int = 8,
        temporal_dim: int = 16,
        video_chamber_dim: int = 128,
        num_chambers: int = 4,
        num_video_features: int = 8,
        hidden_dim: int = 256,
        lstm_hidden: int = 256,
        dropout: float = 0.30
    ):
        super().__init__()

        self.num_chambers = num_chambers

        self.ai_encoder = nn.Sequential(
            nn.Linear(ai_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )

        self.phase_encoder = nn.Sequential(
            nn.Linear(phase_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )

        self.temporal_encoders = nn.ModuleDict()
        for chamber in ['LA', 'LV', 'RA', 'RV']:
            self.temporal_encoders[chamber] = SimplifiedTemporalEncoder(
                input_dim=temporal_dim,
                hidden_dim=lstm_hidden,
                num_layers=2,
                dropout=dropout
            )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_chamber_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        fusion_dim = (
            hidden_dim // 2 +
            hidden_dim // 4 +
            (lstm_hidden // 2) * num_chambers +
            (hidden_dim // 4) * num_video_features
        )

        self.attention_fusion = AttentionFusion(fusion_dim, num_heads=8, dropout=0.15)
        self.fusion_dropout = nn.Dropout(dropout)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, ai_feats, phase_feats, temporal_feats, video_feats):
        batch_size = ai_feats.size(0)

        ai_encoded = self.ai_encoder(ai_feats)
        phase_encoded = self.phase_encoder(phase_feats)

        temporal_encoded = []
        for chamber in ['LA', 'LV', 'RA', 'RV']:
            encoded = self.temporal_encoders[chamber](temporal_feats[chamber])
            temporal_encoded.append(encoded)
        temporal_encoded = torch.cat(temporal_encoded, dim=1)

        video_feats_reshaped = video_feats.view(batch_size * 8, 128)
        video_encoded = self.video_encoder(video_feats_reshaped)
        video_encoded = video_encoded.view(batch_size, -1)

        fused = torch.cat([
            ai_encoded,
            phase_encoded,
            temporal_encoded,
            video_encoded
        ], dim=1)

        fused = fused.unsqueeze(1)
        fused = self.attention_fusion(fused)
        fused = fused.squeeze(1)
        fused = self.fusion_dropout(fused)

        ef_pred = self.regressor(fused).squeeze(-1)
        ef_pred = (ef_pred + 1) / 2

        return ef_pred


class ConservativeTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        ef_normalizer,
        feature_scalers: Dict = None,
        learning_rate: float = 9e-4,
        weight_decay: float = 3e-4,
        max_epochs: int = 100,
        patience: int = 16,
        save_dir: str = "checkpoints",
        warmup_epochs: int = 3
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.ef_normalizer = ef_normalizer
        self.feature_scalers = feature_scalers or {}
        self.max_epochs = max_epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=True
        )

        self.criterion = nn.HuberLoss(delta=0.1)
        self.augmenter = SimpleFeatureAugmenter()

        self.best_val_r2 = -float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.start_epoch = 1
        self.best_checkpoint_path = None

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_r2': [], 'val_r2': [],
            'train_mae': [], 'val_mae': [],
            'lr': [], 'generalization_gap': []
        }

        self.log_path = self.save_dir / "training_log.csv"
        self._initialize_csv_log()

        self.visualizer = EchoNetStyleVisualizer(self.save_dir)

        logger.info(f"Conservative trainer initialized:")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Weight Decay: {weight_decay}")
        logger.info(f"  Dropout: 0.30")
        logger.info(f"  Patience: {patience}")
        logger.info(f"  Using only slight Gaussian noise augmentation")
        logger.info(f"  Model structure: LSTM 2 layers/256 dim, Attention 8 heads")

    def _initialize_csv_log(self):
        with open(self.log_path, 'w') as f:
            f.write("epoch,phase,loss,r2_score,mae,lr,generalization_gap\n")

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            try:
                ai_feats = batch['ai_feats'].to(self.device)
                phase_feats = batch['phase_feats'].to(self.device)
                temporal_feats = {k: v.to(self.device) for k, v in batch['temporal_feats'].items()}
                video_feats = batch['video_feats'].to(self.device)
                targets = batch['efs'].to(self.device)

                if self.model.training:
                    for chamber in temporal_feats.keys():
                        temporal_feats[chamber] = self.augmenter.add_gaussian_noise(
                            temporal_feats[chamber], noise_std=0.01
                        )
                    ai_feats = self.augmenter.add_gaussian_noise(ai_feats, noise_std=0.008)
                    video_feats = self.augmenter.add_gaussian_noise(video_feats, noise_std=0.01)

                predictions = self.model(ai_feats, phase_feats, temporal_feats, video_feats)
                loss = self.criterion(predictions, targets)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

                preds_denorm = [self.ef_normalizer.denormalize(p.item()) for p in predictions.detach()]
                targets_denorm = [self.ef_normalizer.denormalize(t.item()) for t in targets.detach()]

                all_preds.extend(preds_denorm)
                all_targets.extend(targets_denorm)

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            except Exception as e:
                logger.error(f"Training batch {batch_idx} failed: {str(e)}")
                continue

        avg_loss = total_loss / len(self.train_loader)
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)

        return avg_loss, r2, mae

    def validate(self, epoch):
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                try:
                    ai_feats = batch['ai_feats'].to(self.device)
                    phase_feats = batch['phase_feats'].to(self.device)
                    temporal_feats = {k: v.to(self.device) for k, v in batch['temporal_feats'].items()}
                    video_feats = batch['video_feats'].to(self.device)
                    targets = batch['efs'].to(self.device)

                    predictions = self.model(ai_feats, phase_feats, temporal_feats, video_feats)
                    loss = self.criterion(predictions, targets)

                    total_loss += loss.item()

                    preds_denorm = [self.ef_normalizer.denormalize(p.item()) for p in predictions]
                    targets_denorm = [self.ef_normalizer.denormalize(t.item()) for t in targets]

                    all_preds.extend(preds_denorm)
                    all_targets.extend(targets_denorm)

                except Exception as e:
                    logger.error(f"Validation batch {batch_idx} failed: {str(e)}")
                    continue

        avg_loss = total_loss / len(self.val_loader)
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        correlation = np.corrcoef(all_preds, all_targets)[0, 1]

        return avg_loss, r2, mae, correlation, np.array(all_preds), np.array(all_targets)

    def test(self, test_loader: DataLoader) -> Dict:
        logger.info("\n" + "=" * 80)
        logger.info("Starting test set evaluation...")
        logger.info("=" * 80)

        self.model.eval()

        total_loss = 0
        all_preds = []
        all_targets = []
        all_video_ids = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                try:
                    ai_feats = batch['ai_feats'].to(self.device)
                    phase_feats = batch['phase_feats'].to(self.device)
                    temporal_feats = {k: v.to(self.device) for k, v in batch['temporal_feats'].items()}
                    video_feats = batch['video_feats'].to(self.device)
                    targets = batch['efs'].to(self.device)
                    video_ids = batch['video_ids']

                    predictions = self.model(ai_feats, phase_feats, temporal_feats, video_feats)
                    loss = self.criterion(predictions, targets)
                    total_loss += loss.item()

                    preds_denorm = [self.ef_normalizer.denormalize(p.item()) for p in predictions]
                    targets_denorm = [self.ef_normalizer.denormalize(t.item()) for t in targets]

                    all_preds.extend(preds_denorm)
                    all_targets.extend(targets_denorm)
                    all_video_ids.extend(video_ids)

                except Exception as e:
                    logger.error(f"Test batch {batch_idx} failed: {str(e)}")
                    continue

        avg_loss = total_loss / len(test_loader)
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

        pred_std = np.std(all_preds)
        if pred_std < 1e-6:
            logger.error(f"Error: Prediction std dev near 0: {pred_std:.10f}")
            correlation = float('nan')
        else:
            correlation = np.corrcoef(all_preds, all_targets)[0, 1]

        errors = np.array(all_preds) - np.array(all_targets)
        abs_errors = np.abs(errors)

        within_5 = (abs_errors <= 5).sum() / len(abs_errors) * 100
        within_10 = (abs_errors <= 10).sum() / len(abs_errors) * 100
        within_15 = (abs_errors <= 15).sum() / len(abs_errors) * 100

        logger.info("\nTest set results:")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Correlation: {correlation:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"\nPrediction stats:")
        logger.info(f"  Prediction range: [{np.min(all_preds):.2f}%, {np.max(all_preds):.2f}%]")
        logger.info(f"  Prediction std dev: {pred_std:.6f}")
        logger.info(f"  Error <= 5%: {within_5:.2f}%")
        logger.info(f"  Error <= 10%: {within_10:.2f}%")
        logger.info("=" * 80)

        self.visualizer.create_scatter_plot(
            np.array(all_preds), np.array(all_targets),
            "Test Set Performance", "test_scatter.pdf", phase='test'
        )

        self.visualizer.create_roc_plot(
            np.array(all_preds), np.array(all_targets),
            "Test Set ROC Curves", "test_roc.pdf", phase='test'
        )

        return {
            'loss': avg_loss,
            'r2': r2,
            'correlation': correlation,
            'mae': mae,
            'rmse': rmse,
            'pred_std': float(pred_std),
            'within_5': float(within_5),
            'within_10': float(within_10),
            'within_15': float(within_15),
            'predictions': [
                {
                    'video_id': vid,
                    'predicted_ef': float(pred),
                    'true_ef': float(true),
                    'error': float(pred - true),
                    'abs_error': float(abs(pred - true))
                }
                for vid, pred, true in zip(all_video_ids, all_preds, all_targets)
            ]
        }

    def train(self):
        logger.info("\n" + "=" * 80)
        logger.info("Starting training (Conservative strategy)")
        logger.info("=" * 80)

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.max_epochs}")

            try:
                train_loss, train_r2, train_mae = self.train_epoch(epoch)
                val_loss, val_r2, val_mae, val_corr, val_preds, val_targets = self.validate(epoch)

                self.scheduler.step(val_r2)
                current_lr = self.optimizer.param_groups[0]['lr']

                generalization_gap = train_r2 - val_r2

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_r2'].append(train_r2)
                self.history['val_r2'].append(val_r2)
                self.history['train_mae'].append(train_mae)
                self.history['val_mae'].append(val_mae)
                self.history['lr'].append(current_lr)
                self.history['generalization_gap'].append(generalization_gap)

                with open(self.log_path, 'a') as f:
                    f.write(f"{epoch},train,{train_loss:.6f},{train_r2:.6f},{train_mae:.6f},{current_lr:.8f},{generalization_gap:.6f}\n")
                    f.write(f"{epoch},val,{val_loss:.6f},{val_r2:.6f},{val_mae:.6f},{current_lr:.8f},{generalization_gap:.6f}\n")

                logger.info(f"  Train - Loss: {train_loss:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
                logger.info(f"  Val - Loss: {val_loss:.4f}, R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
                logger.info(f"  Generalization gap: {generalization_gap:.4f} ({generalization_gap*100:.1f}%)")
                logger.info(f"  Learning rate: {current_lr:.2e}")

                if val_r2 > self.best_val_r2:
                    improvement = val_r2 - self.best_val_r2
                    self.best_val_r2 = val_r2
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0

                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'val_r2': val_r2,
                        'val_mae': val_mae,
                        'generalization_gap': generalization_gap,
                        'ef_normalizer': self.ef_normalizer,
                        'feature_scalers': self.feature_scalers,
                        'history': self.history
                    }

                    checkpoint_filename = f"best_model_r2_{val_r2:.4f}.pt"
                    temp_path = self.save_dir / f"temp_{checkpoint_filename}"
                    final_path = self.save_dir / checkpoint_filename

                    try:
                        torch.save(checkpoint, temp_path)

                        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
                            self.best_checkpoint_path.unlink()

                        temp_path.rename(final_path)
                        self.best_checkpoint_path = final_path

                        logger.info(f"  Saved best model (↑{improvement:.4f}): {checkpoint_filename}")

                    except Exception as save_error:
                        logger.error(f"  Failed to save model: {save_error}")
                        if temp_path.exists():
                            temp_path.unlink()

                else:
                    self.epochs_without_improvement += 1
                    logger.info(f"  {self.epochs_without_improvement}/{self.patience} epochs without improvement")

                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"\nEarly stopping at epoch {epoch}")
                    logger.info(f"    Best Val R²: {self.best_val_r2:.4f} (Epoch {self.best_epoch})")
                    break

            except Exception as e:
                logger.error(f"Epoch {epoch} failed: {str(e)}")
                import traceback
                traceback.print_exc()
                break

        logger.info("\n" + "=" * 80)
        logger.info(f"Training completed!")
        logger.info(f"   Best Val R²: {self.best_val_r2:.4f} (Epoch {self.best_epoch})")
        if len(self.history['generalization_gap']) > 0 and self.best_epoch > 0:
            logger.info(f"   Final generalization gap: {self.history['generalization_gap'][self.best_epoch-1]:.4f}")
        logger.info("=" * 80)

        return self.history

    def load_checkpoint(self, checkpoint_path: str):
        logger.info(f"\nLoading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if 'val_r2' in checkpoint:
            self.best_val_r2 = checkpoint['val_r2']

        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_epoch = checkpoint['epoch']

        logger.info(f"  Model loaded successfully")


def fit_all_scalers(dataset):
    logger.info("\nFitting feature scalers...")

    scalers = {}

    if dataset.temporal_dict:
        all_blood_flows = []
        for video_dict in dataset.temporal_dict.values():
            for chamber_data in video_dict.values():
                if 'Blood_Flow' in chamber_data.columns:
                    all_blood_flows.extend(chamber_data['Blood_Flow'].unique().tolist())

        if all_blood_flows:
            dataset.feature_extractor.fit_encoder(all_blood_flows)

        for chamber in dataset.CHAMBERS:
            all_features = []
            sampled_count = 0
            max_samples = 1000

            for video_id, video_dict in dataset.temporal_dict.items():
                if sampled_count >= max_samples:
                    break

                if chamber in video_dict:
                    chamber_data = video_dict[chamber]
                    if len(chamber_data) > 0:
                        try:
                            features = dataset.feature_extractor.extract_features(chamber_data)
                            all_features.append(features)
                            sampled_count += 1
                        except:
                            continue

            if all_features:
                all_features = np.vstack(all_features)
                scaler = RobustScaler()
                scaler.fit(all_features)
                scalers[f"temporal_{chamber}"] = scaler
                logger.info(f"  {chamber} scaler fitted")

    logger.info(f"  Scaler fitting completed: {len(scalers)} scalers")
    return scalers


def main():
    parser = argparse.ArgumentParser(description="EF Prediction Training Script (Conservative Optimization Version)")
    parser.add_argument('--base_dir', type=str, default='/path/to/your/own/data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=9e-4)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default='/path/to/your/own/checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--model_type', type=str, default='r2plus1d', choices=['r2plus1d', 'r3d'])
    parser.add_argument('--dropout', type=float, default=0.30)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Using model type: {args.model_type.upper()}")
    logger.info(f"Conservative optimization version (dropout={args.dropout}, wd={args.weight_decay}, slight noise augmentation)")

    if args.resume == 'auto' or args.test_only:
        checkpoint_dir = Path(args.save_dir)
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('best_model_r2_*.pt'))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                args.resume = str(latest_checkpoint)
                logger.info(f"\nAutomatically found checkpoint: {latest_checkpoint.name}")

    ef_normalizer = EFNormalizer()
    feature_scalers = {}

    temp_df = pd.read_csv(Path(args.base_dir) / "FileList.csv")
    temp_df = temp_df[temp_df['Split'] == 'TRAIN'].dropna(subset=['EF'])
    ef_normalizer.fit(temp_df['EF'].values)

    if args.resume and Path(args.resume).exists():
        try:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if 'ef_normalizer' in checkpoint:
                ef_normalizer = checkpoint['ef_normalizer']
            if 'feature_scalers' in checkpoint:
                feature_scalers = checkpoint['feature_scalers']
        except Exception as e:
            logger.warning(f"Cannot load from checkpoint: {e}")

    feature_extractor = RobustTemporalFeatureExtractor()

    if args.test_only:
        logger.info("\nTest mode")

        if not feature_scalers:
            temp_train_dataset = OptimizedFourLayerDataset(
                base_dir=args.base_dir, split="TRAIN",
                use_ai_diagnosis=False, use_phase_info=False,
                use_temporal=True, use_video=False,
                model_type=args.model_type,
                feature_extractor=feature_extractor,
                ef_normalizer=ef_normalizer
            )
            feature_scalers = fit_all_scalers(temp_train_dataset)

        test_dataset = OptimizedFourLayerDataset(
            base_dir=args.base_dir, split="TEST",
            use_ai_diagnosis=True, use_phase_info=True,
            use_temporal=True, use_video=True,
            model_type=args.model_type,
            feature_scalers=feature_scalers,
            feature_extractor=feature_extractor,
            ef_normalizer=ef_normalizer
        )

        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=2, collate_fn=collate_fn, pin_memory=True
        )

        model = ImprovedFourLayerFusionModel(dropout=args.dropout)
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        trainer = ConservativeTrainer(
            model=model, train_loader=None, val_loader=None,
            device=device, ef_normalizer=ef_normalizer,
            feature_scalers=feature_scalers, save_dir=args.save_dir
        )

        test_results = trainer.test(test_loader)

        results_path = Path(args.save_dir) / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"\nTest results saved: {results_path}")

        return

    if not feature_scalers:
        train_dataset = OptimizedFourLayerDataset(
            base_dir=args.base_dir, split="TRAIN",
            use_ai_diagnosis=True, use_phase_info=True,
            use_temporal=True, use_video=True,
            model_type=args.model_type,
            feature_extractor=feature_extractor,
            ef_normalizer=ef_normalizer, augment=True
        )
        feature_scalers = fit_all_scalers(train_dataset)

    train_dataset = OptimizedFourLayerDataset(
        base_dir=args.base_dir, split="TRAIN",
        use_ai_diagnosis=True, use_phase_info=True,
        use_temporal=True, use_video=True,
        model_type=args.model_type,
        feature_scalers=feature_scalers,
        feature_extractor=feature_extractor,
        ef_normalizer=ef_normalizer, augment=True
    )

    val_dataset = OptimizedFourLayerDataset(
        base_dir=args.base_dir, split="VAL",
        use_ai_diagnosis=True, use_phase_info=True,
        use_temporal=True, use_video=True,
        model_type=args.model_type,
        feature_scalers=feature_scalers,
        feature_extractor=feature_extractor,
        ef_normalizer=ef_normalizer
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    model = ImprovedFourLayerFusionModel(dropout=args.dropout)

    trainer = ConservativeTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, ef_normalizer=ef_normalizer,
        feature_scalers=feature_scalers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        save_dir=args.save_dir
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.train()

    if trainer.best_checkpoint_path:
        test_dataset = OptimizedFourLayerDataset(
            base_dir=args.base_dir, split="TEST",
            use_ai_diagnosis=True, use_phase_info=True,
            use_temporal=True, use_video=True,
            model_type=args.model_type,
            feature_scalers=feature_scalers,
            feature_extractor=feature_extractor,
            ef_normalizer=ef_normalizer
        )

        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=2, collate_fn=collate_fn, pin_memory=True
        )

        checkpoint = torch.load(trainer.best_checkpoint_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])

        test_results = trainer.test(test_loader)

        results_path = Path(args.save_dir) / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)

        logger.info("\nAll done!")


if __name__ == "__main__":
    main()