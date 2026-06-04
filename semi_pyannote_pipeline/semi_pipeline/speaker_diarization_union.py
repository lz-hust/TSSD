import sys
import os
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
from pyannote.core import SlidingWindowFeature
from pyannote.audio.core.task import Specifications, Problem
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
            print("Warning: No unlabeled audio found, Mean Teacher will degrade to supervised training")
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
        print(f"Total unlabeled duration: {self.total_annotated:.1f} seconds")
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

class SpeakerDiarizationUnion(SpeakerDiarizationV11):
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
        consistency_loss_type: str = 'mse',
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
            'uncertainty_gamma', 'uncertainty_min_weight', 'uncertainty_warmup_epochs',
            'augmentation_max_epochs'
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
        self.augmentation_max_epochs = kwargs.get('augmentation_max_epochs', None)
        self.has_unlabeled = hasattr(protocol, 'unlabeled_train')
        
        if self.use_mean_teacher and not self.has_unlabeled:
            print("Warning: Protocol does not support unlabeled data, disabling Mean Teacher")
            self.use_mean_teacher = False

        self.teacher_model = None
        self.teacher_validation_metric = None  
        self.teacher_validation_metric_name = "DiarizationErrorRate_teacher"
        self.audio_augmentation = None
        self.global_step = 0
        self.current_epoch = 0

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        chunk = super().prepare_chunk(file_id, start_time, duration)

        if chunk is None or "y" not in chunk:
            return chunk

        swf = chunk["y"]
        raw_data = swf.data
        raw_labels = swf.labels  
        target_labels = self.specifications.classes

        new_data = np.zeros(
            (raw_data.shape[0], len(target_labels)), 
            dtype=raw_data.dtype
        )
        

        ann = None
        try:
            ann = self._id_to_annotation.get(int(file_id)) if hasattr(self, "_id_to_annotation") else None
        except Exception:
            ann = None

        if ann is not None:
            sw = swf.sliding_window
            num_frames = raw_data.shape[0]
            chunk_start = start_time
            chunk_end = start_time + duration
            step = sw.step

            for segment, _, label in ann.itertracks(yield_label=True):
                overlap_start = max(float(segment.start), float(chunk_start))
                overlap_end = min(float(segment.end), float(chunk_end))
                if overlap_end <= overlap_start:
                    continue
                if label not in target_labels:
                    continue
                target_idx = target_labels.index(label)
                start_idx = int(np.floor((overlap_start - chunk_start) / step))
                end_idx = int(np.ceil((overlap_end - chunk_start) / step))
                start_idx = max(0, start_idx)
                end_idx = min(num_frames, end_idx)
                if end_idx > start_idx:
                    new_data[start_idx:end_idx, target_idx] = 1.0
        else:
            if len(raw_labels) > 0:
                first_label = raw_labels[0]
                
                if isinstance(first_label, str):
                    for raw_idx, label in enumerate(raw_labels):
                        if label in target_labels:
                            target_idx = target_labels.index(label)
                            new_data[:, target_idx] = raw_data[:, raw_idx]
                else:
                    for raw_idx, label_idx in enumerate(raw_labels):
                        target_idx = int(label_idx)
                        if 0 <= target_idx < len(target_labels):
                            new_data[:, target_idx] = raw_data[:, raw_idx]
        

        chunk["y"] = SlidingWindowFeature(
            new_data, 
            swf.sliding_window, 
            labels=target_labels
        )

        meta = chunk.setdefault("meta", {})
        meta["start"] = float(start_time)
        meta["duration"] = float(duration)
        meta["file_id"] = int(file_id)

        return chunk

    def _ensure_teacher_device(self, reference_device: torch.device) -> None:
        if not (self.use_mean_teacher and self.teacher_model is not None):
            return
        try:
            teacher_param = next(self.teacher_model.parameters())
        except StopIteration:
            return
        teacher_device = teacher_param.device
        if teacher_device != reference_device:
            print(f"[TeacherDevice] Moving Teacher from {teacher_device} to {reference_device}")
            self.teacher_model = self.teacher_model.to(reference_device)
    
    def setup(self, stage: Optional[str] = None):
        stage_key = stage or "__none__"
        if not hasattr(self, "_scheme_b_setup_done_stages"):
            self._scheme_b_setup_done_stages = set()
        if stage_key in self._scheme_b_setup_done_stages:
            print(f"\n[Scheme B] Setup already executed (stage={stage_key}), skipping duplicate call")
            return
        print("\n" + "="*70)
        print("[Scheme B] Starting 4-channel model construction strategy")
        print("="*70)
        original_max_speakers = self.max_speakers_per_chunk
        print(f"Original max_speakers_per_chunk: {original_max_speakers}")
        self.max_speakers_per_chunk = 4
        print(f"Temporarily setting max_speakers_per_chunk = 4 (deception mode)")
        print(f"Calling super().setup()...")
        super().setup(stage)
        if hasattr(self, 'specifications'):
            temp_classes = self.specifications.classes
            print(f"Stage 1 complete: specifications.classes = {temp_classes}")
            print(f"   Model will be built with {len(temp_classes)} output channels")
            
            if len(temp_classes) != 4:
                print(f"Warning: Expected 4 classes, got {len(temp_classes)}!")
        print(f"\nStage 2: Building model immediately (specifications has 4 classes)")
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'build'):
                try:
                    self.model.build()
                    print(f"Student model built with 4 channels")
                    if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'linear'):
                        out_features = self.model.classifier.linear.out_features
                        print(f"Verification: classifier.linear.out_features = {out_features}")
                        if out_features == 4:
                            print(f"Classification head confirmed as 4 channels")
                        else:
                            print(f"Warning: Classification head has {out_features} channels, not expected 4!")
                except Exception as e:
                    print(f"Warning: Model build failed: {e}")
            else:
                print(f"Warning: Model has no build() method")
        else:
            print(f"Warning: self.model does not exist, skipping build")
        
        print(f"\nStage 3: Correcting label system to 2 classes (model structure locked)")
        
        if original_max_speakers is None:
            self.max_speakers_per_chunk = 4
            print(f"Original value was None, locking max_speakers_per_chunk = 4")
        else:
            self.max_speakers_per_chunk = original_max_speakers
            print(f"Restoring max_speakers_per_chunk = {self.max_speakers_per_chunk}")
        
        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=self.specifications.resolution,
            duration=self.specifications.duration,
            min_duration=self.specifications.min_duration,
            warm_up=self.specifications.warm_up,
            classes=['student', 'teacher'], 
            powerset_max_classes=None,
            permutation_invariant=False,
        )
        
        print(f"specifications.classes corrected to: {self.specifications.classes}")
        print(f"\n[Scheme B] Strategy complete!")
        print(f"   Model: 4-channel output (can load pretrained weights)")
        print(f"   Labels: 2-class system (student, teacher)")
        print(f"   Loss: Union Loss dynamic mapping 4->2")
        print("="*70 + "\n")

        self._scheme_b_setup_done_stages.add(stage_key)

        try:
            self._id_to_annotation = {idx: f.get("annotation") for idx, f in enumerate(self.protocol.train())}
        except Exception as e:
            print(f"Warning: Cannot rebuild id->annotation mapping: {e}")
            self._id_to_annotation = {}

        if self.use_mean_teacher and self.has_unlabeled and stage == "fit":
            print("\n" + "="*70)
            print("Mean Teacher Initialization")
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
                print(f"Teacher model moved to: {device}")
            except StopIteration:
                print("Warning: Cannot get Student device, Teacher remains on CPU")
   
            if hasattr(self.teacher_model, 'build'):
                print("Calling Teacher model build() method...")
                try:
                    self.teacher_model.build()
                    print("Teacher model build() complete")
                except Exception as e:
                    print(f"Warning: Teacher model build() failed: {e}")
            
            try:
                student_device = next(self.model.parameters()).device
                self._ensure_teacher_device(student_device)
            except StopIteration:
                pass

            for param in self.teacher_model.parameters():
                param.requires_grad = False

            self.teacher_model.eval()

            checkpoint_state = getattr(self, '_teacher_checkpoint_state_dict', None)
            if checkpoint_state is not None:
                missing = self.teacher_model.load_state_dict(checkpoint_state, strict=False)
                print("Teacher model weights restored from checkpoint")
                if missing.missing_keys or missing.unexpected_keys:
                    print(f"   Missing keys: {missing.missing_keys}")
                    print(f"   Unexpected keys: {missing.unexpected_keys}")
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
                print("Audio Augmentation: Initializing Robust Mean Teacher Augmentation")
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
                        augmentation_max_epochs=self.augmentation_max_epochs,
                    )
                    print("Audio augmentation created successfully")
                    print(f"   - MUSAN path: {self.musan_path}")
                    print(f"   - RIR path: {self.rir_path}")
                    print(f"   - SNR range: {self.robust_initial_snr_db:.0f} -> {self.robust_final_snr_db:.0f} dB")
                    print(f"   - RIR range: 0% -> {self.robust_rir_max_prob*100:.0f}%")
                    print(f"   - Pitch Shift prob: {self.pitch_shift_prob:.2f}")
                    print(f"   - Megaphone prob: {self.megaphone_prob:.2f}")
                    print(f"   - Total training epochs: {self.max_epochs}")
                    if self.augmentation_max_epochs is not None:
                        print(f"   - Augmentation max epochs: {self.augmentation_max_epochs} (independent control)")
                        if self.augmentation_max_epochs < self.max_epochs:
                            print(f"   - Augmentation will reach max intensity at epoch {self.augmentation_max_epochs} and remain constant")
                    else:
                        print(f"   - Augmentation max epochs: {self.max_epochs} (synced with training)")
                    print("="*70 + "\n")
                except ImportError as e:
                    print(f"Warning: Cannot load RobustMeanTeacherAugmentation: {e}")
                    print("   Please install: pip install torch-audiomentations")
                    self.audio_augmentation = None
                except FileNotFoundError as e:
                    print(f"Warning: Data path error: {e}")
                    print("   Please check musan_path and rir_path")
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
            print("Creating Streaming Mixed Dataloader...")
            print("="*70)
            
            labeled_dataset = TrainDataset(self)
            print("Labeled dataset created (IterableDataset - streaming)")
            
            unlabeled_dataset = UnlabeledTrainDataset(self)
            print("Unlabeled dataset created (streaming)")
            
            ratio_sum = self.labeled_unlabeled_ratio + 1.0
            labeled_fraction = self.labeled_unlabeled_ratio / ratio_sum
            labeled_batch_size = max(1, int(self.batch_size * labeled_fraction))
            
            print(f"\nBatch configuration:")
            print(f"  - Total batch size: {self.batch_size}")
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
            
            print("\nStreaming mixed dataloader created successfully")
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
    
    def _compute_union_loss(
        self, 
        model_output: torch.Tensor, 
        target: torch.Tensor, 
        teacher_gt_idx: int,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
       
        if teacher_gt_idx == 0:
            target_teacher = target[:, :, 0:1]
            target_student = target[:, :, 1:2]
        else:
            target_teacher = target[:, :, 1:2]
            target_student = target[:, :, 0:1]
        
        probs = model_output
        num_channels = probs.shape[-1]
        
        if num_channels < 2:
            raise ValueError(f"Model output channels {num_channels} < 2, cannot apply Union Loss")
        
        losses_per_hypothesis = []
        
        for teacher_idx in range(num_channels):
            pred_teacher = probs[:, :, teacher_idx:teacher_idx+1]
            
            student_indices = [i for i in range(num_channels) if i != teacher_idx]
            pred_students_raw = probs[:, :, student_indices]
            pred_student_union, _ = torch.max(pred_students_raw, dim=-1, keepdim=True)
            
            eps = 1e-7
            pred_teacher = torch.clamp(pred_teacher, eps, 1 - eps)
            pred_student_union = torch.clamp(pred_student_union, eps, 1 - eps)
            
            loss_teacher = -(
                target_teacher * torch.log(pred_teacher) + 
                (1 - target_teacher) * torch.log(1 - pred_teacher)
            )
            
            loss_student = -(
                target_student * torch.log(pred_student_union) + 
                (1 - target_student) * torch.log(1 - pred_student_union)
            )
            
            if weight is not None:
                loss_teacher = loss_teacher * weight
                loss_student = loss_student * weight
            
            loss_combined = (loss_teacher + loss_student).mean(dim=1).squeeze(-1)
            losses_per_hypothesis.append(loss_combined)
        
        losses_stack = torch.stack(losses_per_hypothesis, dim=1)
        min_loss_per_sample, best_hypothesis_idx = torch.min(losses_stack, dim=1)
        
        return min_loss_per_sample.mean()
    
    def _standard_training_step(self, batch, batch_idx: int):
        target = batch["y"]
        waveform = batch["X"]
        
        real_classes = self.specifications.classes
        num_real_classes = len(real_classes)
        
        if target.shape[-1] > num_real_classes:
            target = target[..., :num_real_classes]
            
        if target.shape[-1] != 2:
            raise ValueError(f"[Training] Target dimension error! Expected 2 (from specifications), got {target.shape[-1]}. Please check if RTTM labels contain classes other than teacher/student.")
        
        change_point_target, positive_ratio = self.collate_is_change(batch, target)
        num_speakers = torch.sum(torch.any(target, dim=1), dim=1)
        keep = num_speakers <= self.max_speakers_per_chunk
        target = target[keep]
        change_point_target = change_point_target[keep]
        waveform = waveform[keep]
        
        if not keep.any():
            return None

        main_output, change_point_output = self.model(waveform)
        batch_size, num_frames, num_model_channels = main_output.shape

        weight_key = getattr(self, "weight", None)
        weight = batch.get(weight_key, torch.ones(batch_size, num_frames, 1, device=self.model.device))
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right:] = 0.0

        teacher_gt_idx = -1
        if hasattr(self, 'specifications') and hasattr(self.specifications, 'classes'):
            classes = self.specifications.classes
            for i, class_name in enumerate(classes):
                if class_name.lower() == 'teacher':
                    teacher_gt_idx = i
                    break
        
        if teacher_gt_idx == -1:
            raise ValueError(
                f"Cannot find 'teacher' label in specifications.classes!\n"
                f"Current labels: {self.specifications.classes if hasattr(self, 'specifications') else 'N/A'}\n"
                f"Please ensure data labels contain 'teacher' (case-insensitive)"
            )
        
        if target.shape[-1] != 2:
            raise ValueError(
                f"Union Loss requires target to have 2 classes, but got {target.shape[-1]}\n"
                f"target shape: {target.shape}"
            )
        
        seg_loss = self._compute_union_loss(main_output, target, teacher_gt_idx, weight=weight)

        vad_loss = 0.0
        if self.vad_loss is not None:
            main_output_for_vad = main_output.max(dim=-1, keepdim=True)[0]
            target_for_vad = target.max(dim=-1, keepdim=True)[0]
            vad_loss = self.voice_activity_detection_loss(main_output_for_vad, target_for_vad, weight=weight)

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
        target_original = batch["y"]
        
            
            
        
        target = target_original
        
        real_classes = self.specifications.classes
        num_real_classes = len(real_classes)
        
        if target.shape[-1] > num_real_classes:
            target = target[..., :num_real_classes]
            
        if target.shape[-1] != 2:
            raise ValueError(f"[Validation] Target dimension error! Expected 2 (from specifications), got {target.shape[-1]}. Please check if RTTM labels contain classes other than teacher/student.")
        
        change_point_target, positive_ratio = self.collate_is_change(batch, target)
        waveform = batch["X"]
        
        main_output, change_point_output = self.model(waveform)
        batch_size, num_frames, _ = main_output.shape

        weight_key = getattr(self, "weight", None)
        weight = batch.get(weight_key, torch.ones(batch_size, num_frames, 1, device=self.model.device))
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right:] = 0.0

        teacher_gt_idx = -1
        if hasattr(self, 'specifications') and hasattr(self.specifications, 'classes'):
            classes = self.specifications.classes
            for i, class_name in enumerate(classes):
                if class_name.lower() == 'teacher':
                    teacher_gt_idx = i
                    break
        
        if teacher_gt_idx == -1:
            raise ValueError(
                f"[Validation] Cannot find 'teacher' label!\n"
                f"Current labels: {self.specifications.classes if hasattr(self, 'specifications') else 'N/A'}"
            )
        
        if target.shape[-1] != 2:
            raise ValueError(
                f"[Validation] Union Loss requires target to have 2  classes,but got {target.shape[-1]}"
            )
        
        seg_loss = self._compute_union_loss(main_output, target, teacher_gt_idx, weight=weight)

        vad_loss = 0.0
        if self.vad_loss is not None:
            student_probs_for_vad = main_output.max(dim=-1, keepdim=True)[0]
            target_for_vad = target.max(dim=-1, keepdim=True)[0]
            vad_loss = self.voice_activity_detection_loss(student_probs_for_vad, target_for_vad, weight=weight)

        change_loss = self.focal_loss(
            change_point_output[:, :-1, 0],
            change_point_target,
            alpha=positive_ratio if positive_ratio > 0 else 0.25,
            gamma=2.0,
            reduction='mean'
        )

        total_loss = seg_loss + vad_loss + self.change_weight * change_loss

        self.model.log("loss/val/segmentation", seg_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        if self.vad_loss is not None:
            self.model.log("loss/val/vad", vad_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.model.log("loss/val/change", change_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.model.log("loss/val", total_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        with torch.no_grad():
            probs = main_output
            num_channels = probs.shape[-1]
            target_teacher = target[..., teacher_gt_idx:teacher_gt_idx+1]
            
            pred_teacher_list = []
            pred_student_list = []
            
            for b in range(batch_size):
                sample_probs = probs[b]
                sample_target = target_teacher[b]
                
                losses = [F.mse_loss(sample_probs[:, ch:ch+1], sample_target) for ch in range(num_channels)]
                best_ch = torch.argmin(torch.stack(losses))
                
                p_teacher = sample_probs[:, best_ch:best_ch+1]
                student_indices = [i for i in range(num_channels) if i != best_ch]
                p_student_union, _ = torch.max(sample_probs[:, student_indices], dim=-1, keepdim=True)
                
                pred_teacher_list.append(p_teacher)
                pred_student_list.append(p_student_union)
            
            pred_teacher_stacked = torch.stack(pred_teacher_list)
            pred_student_stacked = torch.stack(pred_student_list)
            
            if teacher_gt_idx == 0:
                final_pred = torch.cat([pred_teacher_stacked, pred_student_stacked], dim=-1)
            else:
                final_pred = torch.cat([pred_student_stacked, pred_teacher_stacked], dim=-1)
            
                
        
        self.model.validation_metric(
            torch.transpose(final_pred[:, warm_up_left:num_frames - warm_up_right], 1, 2),
            torch.transpose(target[:, warm_up_left:num_frames - warm_up_right], 1, 2),
        )
        self.model.log_dict(self.model.validation_metric, on_step=False, on_epoch=True, prog_bar=True)

        if not (self.use_mean_teacher and self.teacher_model is not None):
            return total_loss
        

        target = batch["y"]
        
        real_classes = self.specifications.classes
        num_real_classes = len(real_classes)
        
        if target.shape[-1] > num_real_classes:
            target = target[..., :num_real_classes]
            
        if target.shape[-1] != 2:
            raise ValueError(f"[Validation - Teacher] Target dimension error！Expected 2 (from specifications), got {target.shape[-1]}。Please check if RTTM labels contain classes other than teacher/student 。")
        
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

        teacher_gt_idx = -1
        if hasattr(self, 'specifications') and hasattr(self.specifications, 'classes'):
            classes = self.specifications.classes
            for i, class_name in enumerate(classes):
                if class_name.lower() == 'teacher':
                    teacher_gt_idx = i
                    break
        
        if teacher_gt_idx == -1:
            raise ValueError(
                f"[Validation] Cannot find 'teacher' label!\n"
                f"Current labels: {self.specifications.classes if hasattr(self, 'specifications') else 'N/A'}"
            )
        
        if target.shape[-1] != 2:
            raise ValueError(
                f"[Validation] Union Loss requires target to have 2  classes,but got {target.shape[-1]}"
            )
        
        seg_loss = self._compute_union_loss(teacher_main_output, target, teacher_gt_idx, weight=weight)

        vad_loss = 0.0
        if self.vad_loss is not None:
            teacher_probs_for_vad = teacher_main_output.max(dim=-1, keepdim=True)[0]
            target_for_vad = target.max(dim=-1, keepdim=True)[0]
            vad_loss = self.voice_activity_detection_loss(teacher_probs_for_vad, target_for_vad, weight=weight)

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
            
            probs = teacher_main_output
            num_channels = probs.shape[-1]
            batch_size = probs.shape[0]
            
            target_teacher = target[..., teacher_gt_idx:teacher_gt_idx+1]
            
            pred_teacher_list = []
            pred_student_list = []
            
            for b in range(batch_size):
                sample_probs = probs[b]
                sample_target = target_teacher[b]
                
                losses = [F.mse_loss(sample_probs[:, ch:ch+1], sample_target) for ch in range(num_channels)]
                best_ch = torch.argmin(torch.stack(losses))
                
                p_teacher = sample_probs[:, best_ch:best_ch+1]
                
                student_indices = [i for i in range(num_channels) if i != best_ch]
                p_student_union, _ = torch.max(sample_probs[:, student_indices], dim=-1, keepdim=True)
                
                pred_teacher_list.append(p_teacher)
                pred_student_list.append(p_student_union)
            
            pred_teacher_stacked = torch.stack(pred_teacher_list)
            pred_student_stacked = torch.stack(pred_student_list)
            
            if teacher_gt_idx == 0:
                final_pred = torch.cat([pred_teacher_stacked, pred_student_stacked], dim=-1)
            else:
                final_pred = torch.cat([pred_student_stacked, pred_teacher_stacked], dim=-1)
            
                
            
            self.model.teacher_validation_metric(
                torch.transpose(final_pred[:, warm_up_left:num_frames - warm_up_right], 1, 2),
                torch.transpose(target[:, warm_up_left:num_frames - warm_up_right], 1, 2),
            )
            
            self.model.log_dict(
                self.model.teacher_validation_metric, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True,
                batch_size=batch_size
            )

        return total_loss
    
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
    
    def _pit_mse_loss(self, source_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        import itertools
        
        batch_size, num_frames, num_channels = source_probs.shape
        
        permutations = list(itertools.permutations(range(num_channels)))
        
        losses = []
        
        for p in permutations:
            target_permuted = target_probs[..., p]
            
            mse = F.mse_loss(source_probs, target_permuted, reduction='none').mean(dim=(1, 2))
            losses.append(mse)
            
        losses_stack = torch.stack(losses, dim=1)
        min_loss, best_perm_idx = torch.min(losses_stack, dim=1)

            
            
        
        return min_loss.mean()
    
    def _pit_uncertainty_aware_mse_loss(self, source_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        import itertools
        
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
        
        
        if current >= self.rampup_epochs:
            weight = self.consistency_weight
        elif current < self.rampup_epochs:
            phase = 1.0 - current / self.rampup_epochs
            weight = self.consistency_weight * np.exp(-5.0 * phase * phase)
        else:
            weight = self.consistency_weight
        
        
        return weight
    
    def on_train_start(self):
        if self.use_mean_teacher and self.teacher_model is not None:
            device = next(self.model.parameters()).device
            
            self._ensure_teacher_device(device)
            print("\n" + "="*70)
            print("Mean Teacher Device Synchronization")
            print("="*70)
            print(f"Student device: {device}")
            print(f"Teacher device: {next(self.teacher_model.parameters()).device}")
            print("="*70 + "\n")
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        
        
        if self.use_mean_teacher and self.teacher_model is not None:
            self._update_teacher_model()
        
        self.global_step += 1
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.current_epoch += 1
        
        
        import gc
        gc.collect()
        
    
    def _update_teacher_model(self):
        alpha = min(1.0 - 1.0 / (self.global_step + 1), self.ema_decay)
        
        
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
