

import sys
import os

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Iterator, Dict, Any, List
import numpy as np
import copy
from functools import partial
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pyannote.audio.utils.permutation import permutate
import torchaudio

from sklearn.metrics import precision_recall_fscore_support

from semi_pipeline.speaker_diarization_bet import SpeakerDiarization_bet

from semi_pipeline.streaming_mixed_dataloader import create_streaming_mixed_dataloader




class UnlabeledTrainDataset(IterableDataset):

    def __init__(self, task):
        self.task = task
        self.protocol = task.protocol

        print("\n" + "=" * 70)
        print("Unlabeled Dataset Initialization (PyAnnote-style sampling)")
        print("=" * 70)
        print("Preparing unlabeled files...")

        self.files = list(self.protocol.unlabeled_train())
        self.num_files = len(self.files)

        if self.num_files == 0:
            print("Warning: No unlabeled audio found, MeanTeacher will degrade to supervised training")
        else:
            print(f"Loaded {self.num_files} unlabeled audio files")

        self.duration = float(task.duration)
        self.batch_size = int(getattr(task, "batch_size", 1))
        self.rng = np.random.RandomState()

        self.metadata: List[Dict[str, Any]] = []
        for file in self.files:
            audio_path = file.get("audio")
            if not audio_path:
                continue
            try:
                info = torchaudio.info(audio_path)
            except Exception:
                continue

            sample_rate = info.sample_rate
            total_duration = max(0.0, info.num_frames / max(sample_rate, 1))
            if total_duration <= 0.0:
                continue

            self.metadata.append(
                {
                    "audio_path": audio_path,
                    "sample_rate": sample_rate,
                    "duration": total_duration,
                    "uri": file.get("uri", "unknown"),
                }
            )

        self.total_annotated = sum(meta["duration"] for meta in self.metadata)
        print(f"Total unlabeled duration approximately {self.total_annotated:.1f} seconds")
        print("=" * 70 + "\n")

    def __len__(self) -> int:
        if self.total_annotated <= 0:
            return 0

        total_chunks = 0
        for meta in self.metadata:
            total_chunks += max(1, int(np.ceil(meta["duration"] / self.duration)))

        return max(self.batch_size, total_chunks)

    def iterate_once(self) -> Iterator[Dict[str, Any]]:
        if not self.metadata:
            return

        for meta in self.metadata:
            total_duration = meta["duration"]
            num_chunks = max(1, int(np.ceil(total_duration / self.duration)))

            for idx in range(num_chunks):
                start_time = idx * self.duration
                if start_time >= total_duration:
                    break

                chunk_duration = min(self.duration, total_duration - start_time)
                if chunk_duration <= 0.0:
                    continue

                chunk = self._load_chunk(meta, start_time, chunk_duration)
                if chunk is not None:
                    yield chunk

    def iterate_eternally(self) -> Iterator[Dict[str, Any]]:
        while True:
            yield from self.iterate_once()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self.iterate_eternally()

    def _load_chunk(self, meta: Dict[str, Any], start_time: float, chunk_duration: float) -> Optional[Dict[str, Any]]:
        try:
            frame_offset = int(start_time * meta["sample_rate"])
            num_frames = int(chunk_duration * meta["sample_rate"])
            if num_frames <= 0:
                return None

            waveform, sr = torchaudio.load(
                meta["audio_path"],
                frame_offset=frame_offset,
                num_frames=num_frames,
            )

            if waveform.shape[1] < num_frames:
                waveform = torch.nn.functional.pad(
                    waveform, (0, num_frames - waveform.shape[1])
                )
            elif waveform.shape[1] > num_frames:
                waveform = waveform[:, :num_frames]

            sample: Dict[str, Any] = {"X": waveform}
            meta_out = sample.setdefault("meta", {})
            meta_out["uri"] = meta["uri"]
            meta_out["start"] = start_time
            meta_out["duration"] = chunk_duration

            return sample
        except Exception:
            return None



class SpeakerDiarizationMeanTeacher(SpeakerDiarizationV11):
    
    def __init__(
        self,
        protocol,
        duration: float = 2.0,
        max_speakers_per_chunk: Optional[int] = None,
        max_speakers_per_frame: Optional[int] = None,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        batch_size: int = 32,
        num_workers: int = None,
        change_weight: float = 0.3,
        use_mean_teacher: bool = True,
        ema_decay: float = 0.999,
        consistency_weight: float = 1.0,
        consistency_loss_type: str = 'pit_mse',
        rampup_epochs: int = 10,
        labeled_unlabeled_ratio: float = 1.0,
        use_audio_augmentation: bool = True,
        augmentation_volume_range: tuple = (0.95, 1.05),
        augmentation_initial_snr: float = 25.0,
        augmentation_final_snr: float = 10.0,
        augmentation_schedule: str = 'cosine',
        max_epochs: int = 300,
        **kwargs,
    ):
        mean_teacher_keys = {
            'use_robust_augmentation', 'musan_path', 'rir_path', 
            'robust_final_snr_db', 'robust_initial_snr_db', 
            'robust_augmentation_warmup_ratio', 'robust_rir_max_prob',
            'robust_student_volume_db', 'robust_teacher_volume_db',
            'pitch_shift_prob', 'megaphone_prob',
            'uncertainty_gamma', 'uncertainty_min_weight', 'uncertainty_warmup_epochs'
        }
        parent_kwargs = {k: v for k, v in kwargs.items() if k not in mean_teacher_keys}
        
        super().__init__(
            protocol=protocol,
            duration=duration,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            change_weight=change_weight,
            **parent_kwargs,
        )
        
        self.use_mean_teacher = use_mean_teacher
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight
        self.consistency_loss_type = consistency_loss_type
        self.rampup_epochs = rampup_epochs
        self.labeled_unlabeled_ratio = labeled_unlabeled_ratio
        
        self.use_audio_augmentation = use_audio_augmentation
        self.augmentation_volume_range = augmentation_volume_range
        self.augmentation_initial_snr = augmentation_initial_snr
        self.augmentation_final_snr = augmentation_final_snr
        self.augmentation_schedule = augmentation_schedule
        
        self.use_robust_augmentation = kwargs.get('use_robust_augmentation', False)
        self.musan_path = kwargs.get('musan_path', None)
        self.rir_path = kwargs.get('rir_path', None)
        self.robust_final_snr_db = kwargs.get('robust_final_snr_db', 10.0)
        self.robust_initial_snr_db = kwargs.get('robust_initial_snr_db', 25.0)
        self.robust_augmentation_warmup_ratio = kwargs.get('robust_augmentation_warmup_ratio', 0.1)
        self.robust_rir_max_prob = kwargs.get('robust_rir_max_prob', 0.5)
        self.robust_student_volume_db = kwargs.get('robust_student_volume_db', 6.0)
        self.robust_teacher_volume_db = kwargs.get('robust_teacher_volume_db', 2.0)
        self.pitch_shift_prob = kwargs.get('pitch_shift_prob', 0.5)
        self.megaphone_prob = kwargs.get('megaphone_prob', 0.3)
        
        self.uncertainty_gamma = kwargs.get('uncertainty_gamma', 2.0)
        self.uncertainty_min_weight = kwargs.get('uncertainty_min_weight', 0.1)
        self.uncertainty_warmup_epochs = kwargs.get('uncertainty_warmup_epochs', 5)
        
        self.max_epochs = max_epochs
        
        self.has_unlabeled = hasattr(protocol, 'unlabeled_train')
        
        if self.use_mean_teacher and not self.has_unlabeled:
            print("Warning: Protocol does not support unlabeled data, disabling MeanTeacher")
            self.use_mean_teacher = False

        self.teacher_model = None
        self.teacher_validation_metric = None
        self.teacher_validation_metric_name = "DiarizationErrorRate_teacher"

        self.audio_augmentation = None

        self.global_step = 0
        self.current_epoch = 0

    def _ensure_teacher_device(self, reference_device: torch.device) -> None:
        if not (self.use_mean_teacher and self.teacher_model is not None):
            return
        try:
            teacher_param = next(self.teacher_model.parameters())
        except StopIteration:
            return
        teacher_device = teacher_param.device
        if teacher_device != reference_device:
            print(f"[TeacherDevice] MovingTeacher from {teacher_device} to {reference_device}")
            self.teacher_model = self.teacher_model.to(reference_device)
    
    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        
        if self.use_mean_teacher and self.has_unlabeled and stage == "fit":
            print("\n" + "="*70)
            print("[Mean Teacher] Initialization")
            print("="*70)
            print(f"EMA Decay: {self.ema_decay}")
            print(f"Consistency Weight: {self.consistency_weight}")
            print(f"Consistency Loss Type: {self.consistency_loss_type}")
            print(f"Rampup Epochs: {self.rampup_epochs}")
            print(f"Labeled/Unlabeled Ratio: {self.labeled_unlabeled_ratio}")
            
            if not hasattr(self, 'model') or self.model is None:
                print("Warning: self.model does not exist, skipping Teacher model creation")
                print("="*70 + "\n")
                return
            
            print("Creating Teacher model...")
            self.teacher_model = copy.deepcopy(self.model)
            
            try:
                device = next(self.model.parameters()).device
                print(f"Detected Student device: {device}")
                self.teacher_model = self.teacher_model.to(device)
                print(f"Teacher moved to: {device}")
            except StopIteration:
                print("Warning: Cannot get Student device，Teacher remains on CPU")
            
            if hasattr(self.teacher_model, 'build'):
                print(" CallingTeacher model build() method...")
                try:
                    self.teacher_model.build()
                    print("Teacher model build() complete")
                except Exception as e:
                    print(f"Warning:Teacher model build() failed: {e}")
            
            try:
                student_device = next(self.model.parameters()).device
                self._ensure_teacher_device(student_device)
            except StopIteration:
                pass

            print("🔄 [Setup] Synchronizing Student weights toTeacher...")
            self.teacher_model.load_state_dict(self.model.state_dict())
            print("Teacher weight synchronization complete")

            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            self.teacher_model.eval()
            
            checkpoint_state = getattr(self, '_teacher_checkpoint_state_dict', None)
            if checkpoint_state is not None:
                missing = self.teacher_model.load_state_dict(checkpoint_state, strict=False)
                print("Teacher model weights restored from checkpoint")
                if missing.missing_keys or missing.unexpected_keys:
                    print(f"    Missing keys: {missing.missing_keys}")
                    print(f"    Unexpected keys: {missing.unexpected_keys}")
                delattr(self, '_teacher_checkpoint_state_dict')

            print("Teacher model creation complete")
            
            from pyannote.audio.torchmetrics import (
                OptimalDiarizationErrorRate,
                OptimalDiarizationErrorRateThreshold,
            )
            from torchmetrics import MetricCollection
            
            teacher_metrics = {
                self.teacher_validation_metric_name: OptimalDiarizationErrorRate(),
                f"{self.teacher_validation_metric_name}/Threshold": OptimalDiarizationErrorRateThreshold(),
            }
            self.teacher_validation_metric = MetricCollection(teacher_metrics)
            self.teacher_validation_metric.to(self.model.device)
            
            self.model.teacher_validation_metric = self.teacher_validation_metric
            print(f"Teacher validation metric created: {self.teacher_validation_metric_name}")
            
            if self.use_robust_augmentation and self.musan_path and self.rir_path:
                print("\n" + "="*70)
                print("[Audio Augmentation]Initialization Robust MeanTeacher Augmentation")
                print("="*70)
                try:
                    from data_augm.audio_augmentation import RobustMeanTeacherAugmentation
                    
                    self.audio_augmentation = RobustMeanTeacherAugmentation(
                        musan_path=self.musan_path,
                        rir_path=self.rir_path,
                        total_epochs=self.max_epochs,
                        final_snr_db=self.robust_final_snr_db,
                        initial_snr_db=self.robust_initial_snr_db,
                        augmentation_warmup_ratio=self.robust_augmentation_warmup_ratio,
                        rir_max_prob=self.robust_rir_max_prob,
                        student_volume_db=self.robust_student_volume_db,
                        teacher_volume_db=self.robust_teacher_volume_db,
                        pitch_shift_prob=self.pitch_shift_prob,
                        megaphone_prob=self.megaphone_prob,
                    )
                    print(" Audio Augmentationcreated successfully")
                    print(f"   - MUSAN path: {self.musan_path}")
                    print(f"   - RIR path: {self.rir_path}")
                    print(f"   - SNR range: {self.robust_initial_snr_db:.0f} → {self.robust_final_snr_db:.0f} dB")
                    print(f"   - RIR range: 0% → {self.robust_rir_max_prob*100:.0f}%")
                    print(f"   - Pitch Shift probability: {self.pitch_shift_prob:.2f}")
                    print(f"   - Megaphone probability:   {self.megaphone_prob:.2f}")
                    print(f"   -Total epochs: {self.max_epochs}")
                    print("="*70 + "\n")
                except ImportError as e:
                    print(f"Warning: Cannot load RobustMeanTeacherAugmentation: {e}")
                    print("   Please ensure installed: pip install torch-audiomentations")
                    self.audio_augmentation = None
                except FileNotFoundError as e:
                    print(f"Warning: Data path error: {e}")
                    print("   Please check musan_path and rir_path are correct")
                    self.audio_augmentation = None
            elif self.use_audio_augmentation:
                print("Warning: Using legacy simple audio augmentation (Gaussian noise), recommend switching to use_robust_augmentation")
                from data_augm.audio_augmentation import create_mean_teacher_augmentation
                
                self.audio_augmentation = create_mean_teacher_augmentation(
                    use_volume=True,
                    use_noise=True,
                    volume_range=self.augmentation_volume_range,
                    initial_snr=self.augmentation_initial_snr,
                    final_snr=self.augmentation_final_snr,
                    noise_schedule=self.augmentation_schedule,
                )
            else:
                print("Warning: Audio augmentation not enabled")
                self.audio_augmentation = None
            
            print("="*70 + "\n")
    
    
    def train_dataloader(self):
        if self.use_mean_teacher and self.has_unlabeled:
            from pyannote.audio.core.task import TrainDataset
            
            print("\n" + "="*70)
            print("[Streaming Mixed Dataloader]Creating...")
            print("="*70)
            
            labeled_dataset = TrainDataset(self)
            print(" Labeled dataset created (IterableDataset - streaming)")
            
            unlabeled_dataset = UnlabeledTrainDataset(self)
            print(" Unlabeled Datasetcreated (streaming)")
            
            ratio_sum = self.labeled_unlabeled_ratio + 1.0
            labeled_fraction = self.labeled_unlabeled_ratio / ratio_sum
            labeled_batch_size = max(1, int(self.batch_size * labeled_fraction))
            
            print(f"\nBatch configuration:")
            print(f"  -Total batch size: {self.batch_size}")
            print(f"  - Labeled per batch: {labeled_batch_size}")
            print(f"  - Unlabeled per batch: {self.batch_size - labeled_batch_size}")
            print(f"  - Labeled ratio: {labeled_batch_size / self.batch_size * 100:.1f}%")
            
            mixed_dataloader = create_streaming_mixed_dataloader(
                labeled_dataset=labeled_dataset,
                unlabeled_dataset=unlabeled_dataset,
                task=self,
                batch_size=self.batch_size,
                labeled_batch_size=labeled_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                audio_augmentation=self.audio_augmentation,
            )
            
            print("\n Streaming Mixed Dataloadercreated successfully")
            print("="*70 + "\n")
            
            return mixed_dataloader
        else:
            return super().train_dataloader()
    
    def training_step(self, batch, batch_idx: int):
        if self.use_mean_teacher and hasattr(self, 'teacher_model') and self.teacher_model is not None:
            if not hasattr(self, '_teacher_device_synced'):
                device = next(self.model.parameters()).device
                self._ensure_teacher_device(device)
                self._teacher_device_synced = True
        
        if self.use_mean_teacher and self.audio_augmentation is not None:
            if 'X_student' not in batch:
                raw_wav = batch['X']
                
                with torch.no_grad():
                    student_wav, teacher_wav = self.audio_augmentation(raw_wav)
                
                batch['X_student'] = student_wav
                batch['X_teacher'] = teacher_wav
        
        if not self.use_mean_teacher or 'is_labeled' not in batch:
            return self._standard_training_step(batch, batch_idx)
        
        is_labeled = batch['is_labeled']
        
        has_labeled = is_labeled.any().item()
        has_unlabeled = (~is_labeled).any().item()
        
        total_loss = 0.0
        loss_count = 0
        
        if has_labeled:
            labeled_mask = is_labeled
            
            labeled_meta = {}
            for key, val in batch['meta'].items():
                if isinstance(val, (list, tuple)):
                    labeled_meta[key] = [val[i] for i in range(len(val)) if is_labeled[i]]
                else:
                    labeled_meta[key] = val
            
            if 'X_student' in batch:
                X_for_supervised = batch['X_student'][labeled_mask]
            else:
                X_for_supervised = batch['X'][labeled_mask]
            
            labeled_batch = {
                'X': X_for_supervised,
                'y': batch['y'][labeled_mask],
                'meta': labeled_meta
            }
            
            supervised_loss = self._standard_training_step(labeled_batch, batch_idx)
            
            if supervised_loss is not None:
                total_loss += supervised_loss
                loss_count += 1
                
                if hasattr(self, 'model') and hasattr(self.model, 'log'):
                    self.model.log(
                        "train/supervised_loss",
                        supervised_loss.detach(),
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        batch_size=labeled_mask.sum().item(),
                    )
        
        if 'X_student' in batch and 'X_teacher' in batch:
            student_X = batch['X_student']
            teacher_X = batch['X_teacher']
            consistency_loss = self._compute_consistency_loss_augmented(student_X, teacher_X)
        else:
            all_X = batch['X']
            consistency_loss = self._compute_consistency_loss(all_X)
        
        if consistency_loss is not None:
            current_weight = self._get_current_consistency_weight()
            weighted_consistency_loss = consistency_loss * current_weight
            
            total_loss += weighted_consistency_loss
            loss_count += 1
            
            batch_size_for_log = batch['X_student'].size(0) if 'X_student' in batch else batch['X'].size(0)
            if hasattr(self, 'model') and hasattr(self.model, 'log'):
                self.model.log(
                    "train/consistency_loss",
                    consistency_loss.detach(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size_for_log,
                )
                self.model.log(
                    "train/consistency_weight",
                    current_weight,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )
                self.model.log(
                    "train/weighted_consistency_loss",
                    weighted_consistency_loss.detach(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch_size_for_log,
                )
        
        if loss_count > 0:
            
            batch_size_total = batch['X_student'].size(0) if 'X_student' in batch else batch['X'].size(0)
            if hasattr(self, 'model') and hasattr(self.model, 'log'):
                self.model.log("loss/train", total_loss.detach(),
                        on_step=False, on_epoch=True, prog_bar=True,
                        batch_size=batch_size_total)
            
            return total_loss
        else:
            return None
    
    def _standard_training_step(self, batch, batch_idx: int):
        target = batch["y"]
        waveform = batch["X"]
        
        change_point_target, positive_ratio = self.collate_is_change(batch, target)
        num_speakers = torch.sum(torch.any(target, dim=1), dim=1)
        keep = num_speakers <= self.max_speakers_per_chunk
        target = target[keep]
        change_point_target = change_point_target[keep]
        waveform = waveform[keep]
        
        if not keep.any():
            return None

        main_output, change_point_output = self.model(waveform)
        batch_size, num_frames, _ = main_output.shape

        weight_key = getattr(self, "weight", None)
        weight = batch.get(weight_key, torch.ones(batch_size, num_frames, 1, device=self.model.device))
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right:] = 0.0

        permutated_prediction, _ = permutate(target, main_output)
        seg_loss = self.segmentation_loss(permutated_prediction, target, weight=weight)

        vad_loss = 0.0
        if self.vad_loss is not None:
            vad_loss = self.voice_activity_detection_loss(permutated_prediction, target, weight=weight)

        change_loss = self.focal_loss(
            change_point_output[:, :-1, 0],
            change_point_target,
            alpha=positive_ratio if positive_ratio > 0 else 0.25,
            gamma=2.0,
            reduction='mean'
        )

        total_loss = seg_loss + vad_loss + self.change_weight * change_loss


        self.model.log("loss/train/segmentation", seg_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        if self.vad_loss is not None:
            self.model.log("loss/train/vad", vad_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.model.log("loss/train/change", change_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx: int):
        result = super().validation_step(batch, batch_idx)

        if not (self.use_mean_teacher and self.teacher_model is not None):
            return result
        
        if batch_idx == 0:
            first_teacher_param = next(self.teacher_model.parameters())
            first_student_param = next(self.model.parameters())
            print(f"\n[Validation start] Epoch {self.current_epoch}")
            print(f" Teacher param mean: {first_teacher_param.data.mean():.6f}")
            print(f"  Student param mean: {first_student_param.data.mean():.6f}\n")

        target = batch["y"]
        change_point_target, positive_ratio = self.collate_is_change(batch, target)
        waveform = batch["X"]

        self._ensure_teacher_device(waveform.device)
        with torch.no_grad():
            teacher_main_output, teacher_change_output = self.teacher_model(waveform)

        batch_size, num_frames, _ = teacher_main_output.shape

        weight_key = getattr(self, "weight", None)
        if weight_key is not None and weight_key in batch:
            weight = batch[weight_key].to(teacher_main_output.device)
        else:
            weight = torch.ones(batch_size, num_frames, 1, device=teacher_main_output.device)

        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right:] = 0.0

        permutated_prediction, _ = permutate(target, teacher_main_output)
        seg_loss = self.segmentation_loss(permutated_prediction, target, weight=weight)

        vad_loss = 0.0
        if self.vad_loss is not None:
            vad_loss = self.voice_activity_detection_loss(permutated_prediction, target, weight=weight)

        change_loss = self.focal_loss(
            teacher_change_output[:, :-1, 0],
            change_point_target,
            alpha=positive_ratio if positive_ratio > 0 else 0.25,
            gamma=2.0,
            reduction='mean'
        )

        total_loss = seg_loss + vad_loss + self.change_weight * change_loss

        self.model.log("loss_val_teacher", total_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.model.log("loss_val_teacher/segmentation", seg_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        if self.vad_loss is not None:
            self.model.log("loss_val_teacher/vad", vad_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.model.log("loss_val_teacher/change", change_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

        if hasattr(self.model, 'teacher_validation_metric') and self.model.teacher_validation_metric is not None:
            self.model.teacher_validation_metric.to(teacher_main_output.device)
            
            if self.specifications.powerset:
                multilabel = self.model.powerset.to_multilabel(teacher_main_output)
                self.model.teacher_validation_metric(
                    torch.transpose(multilabel[:, warm_up_left:num_frames - warm_up_right], 1, 2),
                    torch.transpose(target[:, warm_up_left:num_frames - warm_up_right], 1, 2),
                )
            else:
                self.model.teacher_validation_metric(
                    torch.transpose(teacher_main_output[:, warm_up_left:num_frames - warm_up_right], 1, 2),
                    torch.transpose(target[:, warm_up_left:num_frames - warm_up_right], 1, 2),
                )
            
            self.model.log_dict(
                self.model.teacher_validation_metric, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True,
                batch_size=batch_size
            )

        return result
    
    def _supervised_step(self, batch, batch_idx: int):
        loss = super().training_step(batch, batch_idx)
        
        if hasattr(self, 'model') and hasattr(self.model, 'log'):
            batch_size = batch['X'].size(0)
            self.model.log(
                "train/supervised_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        
        return loss
    
    def _consistency_step(self, batch, batch_idx: int):
        waveform = batch["X"]
        if not (self.use_mean_teacher and self.teacher_model is not None):
            raise RuntimeError("Teacher model is not initialized for consistency step")
        
        student_main_output, student_change_output = self.model(waveform)

        self._ensure_teacher_device(waveform.device)
        with torch.no_grad():
            teacher_main_output, teacher_change_output = self.teacher_model(waveform)
        
        consistency_loss = self._compute_consistency_loss(
            student_main_output,
            teacher_main_output,
            student_change_output,
            teacher_change_output
        )
        
        current_weight = self._get_current_consistency_weight()
        
        weighted_consistency_loss = current_weight * consistency_loss
        
        batch_size = waveform.size(0)
        if hasattr(self, 'model') and hasattr(self.model, 'log'):
            self.model.log(
                "train/consistency_loss",
                consistency_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.model.log(
                "train/consistency_weight",
                current_weight,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.model.log(
                "train/weighted_consistency_loss",
                weighted_consistency_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        
        return weighted_consistency_loss
    
    def _pit_mse_loss(self, source_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_channels = source_probs.shape
        
        permutations = list(itertools.permutations(range(num_channels)))
        
        losses = []
        
        for p in permutations:
            target_permuted = target_probs[..., p]
            
            mse = F.mse_loss(source_probs, target_permuted, reduction='none').mean(dim=(1, 2))
            losses.append(mse)
            
        losses_stack = torch.stack(losses, dim=1)
        min_loss, _ = torch.min(losses_stack, dim=1)
        
        return min_loss.mean()
    
    def _pit_uncertainty_aware_mse_loss(self, source_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_channels = source_probs.shape
        
        current_epoch = self.trainer.current_epoch if self.trainer else self.current_epoch
        
        if current_epoch < self.uncertainty_warmup_epochs:
            weights = torch.ones_like(target_probs)
        else:
            dist_from_uncertainty = torch.abs(target_probs - 0.5) * 2.0
            
            weights = dist_from_uncertainty ** self.uncertainty_gamma
            
            weights = torch.clamp(weights, min=self.uncertainty_min_weight)
            
            weights = weights.detach()
        
        permutations = list(itertools.permutations(range(num_channels)))
        losses = []
        
        for p in permutations:
            target_permuted = target_probs[..., p]
            
            weights_permuted = weights[..., p]
            
            squared_diff = (source_probs - target_permuted) ** 2
            weighted_mse = weights_permuted * squared_diff
            
            loss_per_sample = weighted_mse.mean(dim=(1, 2))
            losses.append(loss_per_sample)
        
        losses_stack = torch.stack(losses, dim=1)
        min_loss, _ = torch.min(losses_stack, dim=1)
        
        return min_loss.mean()
    
    def _compute_consistency_loss(self, waveform: torch.Tensor) -> torch.Tensor:
        if not (self.use_mean_teacher and self.teacher_model is not None):
            raise RuntimeError("Teacher model is not initialized for consistency loss")

        student_main_output, student_change_output = self.model(waveform)
        
        self._ensure_teacher_device(waveform.device)
        with torch.no_grad():
            teacher_main_output, teacher_change_output = self.teacher_model(waveform)
        
        if self.consistency_loss_type == 'pit_mse':
            main_consistency = self._pit_mse_loss(student_main_output, teacher_main_output)
        elif self.consistency_loss_type == 'uncertainty_pit_mse':
            main_consistency = self._pit_uncertainty_aware_mse_loss(student_main_output, teacher_main_output)
        elif self.consistency_loss_type == 'mse':
            main_consistency = F.mse_loss(student_main_output, teacher_main_output)
        elif self.consistency_loss_type == 'kl':
            main_consistency = F.kl_div(
                F.log_softmax(student_main_output, dim=-1),
                F.softmax(teacher_main_output, dim=-1),
                reduction='batchmean'
            )
        elif self.consistency_loss_type == 'weighted_mse':
            teacher_confidence = teacher_main_output.max(dim=-1, keepdim=True)[0]
            main_consistency = (F.mse_loss(student_main_output, teacher_main_output, reduction='none') * teacher_confidence).mean()
        else:
            main_consistency = self._pit_mse_loss(student_main_output, teacher_main_output)
        
        change_consistency = F.mse_loss(student_change_output, teacher_change_output)
        
        consistency_loss = main_consistency + change_consistency
        
        return consistency_loss
    
    def _compute_consistency_loss_augmented(
        self, 
        student_waveform: torch.Tensor,
        teacher_waveform: torch.Tensor
    ) -> torch.Tensor:
        if not (self.use_mean_teacher and self.teacher_model is not None):
            raise RuntimeError("Teacher model is not initialized for consistency loss")

        student_main_output, student_change_output = self.model(student_waveform)
        
        self._ensure_teacher_device(teacher_waveform.device)
        with torch.no_grad():
            teacher_main_output, teacher_change_output = self.teacher_model(teacher_waveform)
        
        if self.consistency_loss_type == 'pit_mse':
            main_consistency = self._pit_mse_loss(student_main_output, teacher_main_output)
        elif self.consistency_loss_type == 'uncertainty_pit_mse':
            main_consistency = self._pit_uncertainty_aware_mse_loss(student_main_output, teacher_main_output)
        elif self.consistency_loss_type == 'mse':
            main_consistency = F.mse_loss(student_main_output, teacher_main_output)
        elif self.consistency_loss_type == 'kl':
            main_consistency = F.kl_div(
                F.log_softmax(student_main_output, dim=-1),
                F.softmax(teacher_main_output, dim=-1),
                reduction='batchmean'
            )
        elif self.consistency_loss_type == 'weighted_mse':
            teacher_confidence = teacher_main_output.max(dim=-1, keepdim=True)[0]
            main_consistency = (F.mse_loss(student_main_output, teacher_main_output, reduction='none') * teacher_confidence).mean()
        else:
            main_consistency = self._pit_mse_loss(student_main_output, teacher_main_output)
        
        change_consistency = F.mse_loss(student_change_output, teacher_change_output)
        
        consistency_loss = main_consistency + change_consistency
        
        return consistency_loss
    
    def _get_current_consistency_weight(self) -> float:
        current = self.trainer.current_epoch if self.trainer else self.current_epoch
        
        if self.global_step % 100 == 0:
            print(f"\n[Consistency weight] Step {self.global_step}")
            print(f"  - trainer.current_epoch: {self.trainer.current_epoch if self.trainer else 'N/A'}")
            print(f"  - self.current_epoch: {self.current_epoch}")
            print(f"  - Using epoch: {current}")
            print(f"  - rampup_epochs: {self.rampup_epochs}")
        
        if current >= self.rampup_epochs:
            weight = self.consistency_weight
        elif current < self.rampup_epochs:
            phase = 1.0 - current / self.rampup_epochs
            weight = self.consistency_weight * np.exp(-5.0 * phase * phase)
        else:
            weight = self.consistency_weight
        
        if self.global_step % 100 == 0:
            print(f"  - Calculated weight: {weight:.6f}\n")
        
        return weight
    
    def on_train_start(self):
        if self.use_mean_teacher and self.teacher_model is not None:
            device = next(self.model.parameters()).device
            
            self._ensure_teacher_device(device)
            print("\n" + "="*70)
            print("[Mean Teacher] Device synchronization")
            print("="*70)
            print(f"Student device: {device}")
            print(f"Teacher device: {next(self.teacher_model.parameters()).device}")
            print("="*70 + "\n")
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        
        if self.global_step % 50 == 0:
            print(f"[Debug] on_train_batch_end called (step={self.global_step}, use_mean_teacher={self.use_mean_teacher}, teacher_model={'exists' if self.teacher_model is not None else 'does not exist'})")
        
        if self.use_mean_teacher and self.teacher_model is not None:
            self._update_teacher_model()
        
        self.global_step += 1
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.current_epoch += 1
        
        print(f"\n[Epoch end] self.current_epoch updated to: {self.current_epoch}")
        if self.trainer:
            print(f"[Epoch end] trainer.current_epoch: {self.trainer.current_epoch}\n")
        
        import gc
        gc.collect()
        
    
    def _update_teacher_model(self):
        alpha = min(1.0 - 1.0 / (self.global_step + 1), self.ema_decay)
        
        if self.global_step % 100 == 0:
            first_teacher_param = next(self.teacher_model.parameters())
            first_student_param = next(self.model.parameters())
            print(f"\n[EMA update] Step {self.global_step}, alpha={alpha:.6f}")
            print(f" Teacher param mean: {first_teacher_param.data.mean():.6f}")
            print(f"  Student param mean: {first_student_param.data.mean():.6f}")
        
        for teacher_param, student_param in zip(
            self.teacher_model.parameters(),
            self.model.parameters()
        ):
            teacher_param.data.mul_(alpha).add_(
                student_param.data, alpha=1 - alpha
            )
        
        for teacher_buffer, student_buffer in zip(
            self.teacher_model.buffers(),
            self.model.buffers()
        ):
            teacher_buffer.data.copy_(student_buffer.data)
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def on_train_epoch_start(self):
        if hasattr(super(), 'on_train_epoch_start'):
            super().on_train_epoch_start()
        
        if self.audio_augmentation is not None and hasattr(self.audio_augmentation, 'transform'):
            current_epoch = self.trainer.current_epoch if self.trainer else 0
            
            if hasattr(self.audio_augmentation.transform, 'set_epoch'):
                self.audio_augmentation.transform.set_epoch(current_epoch, self.max_epochs)
                
                if hasattr(self.audio_augmentation.transform, 'get_current_snr'):
                    current_snr = self.audio_augmentation.transform.get_current_snr()
                    print(f"[Epoch {current_epoch}] Current SNR: {current_snr:.2f} dB")
                elif hasattr(self.audio_augmentation.transform, 'transforms'):
                    for transform in self.audio_augmentation.transform.transforms:
                        if hasattr(transform, 'get_current_snr'):
                            current_snr = transform.get_current_snr()
                            print(f"[Epoch {current_epoch}] Current SNR: {current_snr:.2f} dB")
                            break
