
import torch
import numpy as np
import os
from typing import Optional, Union, Callable, List
import math

try:
    from torch_audiomentations import (
        Compose,
        AddBackgroundNoise,
        ApplyImpulseResponse,
        Gain,
        PitchShift,
    )

    try:
        from torch_audiomentations import HighPassFilter
    except ImportError:
        HighPassFilter = None

    try:
        from torch_audiomentations import SevenBandParametricEQ
    except ImportError:
        SevenBandParametricEQ = None
        print("Warning:  Current torch-audiomentations version missing SevenBandParametricEQ，related filter augmentation will be skipped")

    TORCH_AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    TORCH_AUDIOMENTATIONS_AVAILABLE = False
    Compose = None
    AddBackgroundNoise = None
    ApplyImpulseResponse = None
    Gain = None
    PitchShift = None
    HighPassFilter = None
    SevenBandParametricEQ = None
    print("Warning:  torch_audiomentations not installed, advanced augmentation features unavailable")
    print("   Installation: pip install torch-audiomentations")


class AudioTransformTwice:
    
    def __init__(self, transform: Callable):
        self.transform = transform
    
    def __call__(self, waveform: torch.Tensor) -> tuple:
        aug1 = self.transform(waveform)
        aug2 = self.transform(waveform)
        return aug1, aug2


class VolumeAugmentation:
    
    def __init__(self, min_scale: float = 0.95, max_scale: float = 1.05):
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return waveform * scale


class CurriculumNoiseAugmentation:
    
    def __init__(
        self,
        initial_snr_db: float = 25.0,
        final_snr_db: float = 10.0,
        snr_std: float = 2.0,
        noise_type: str = 'gaussian',
        schedule: str = 'linear'
    ):
        self.initial_snr_db = initial_snr_db
        self.final_snr_db = final_snr_db
        self.snr_std = snr_std
        self.noise_type = noise_type
        self.schedule = schedule
        
        self.current_snr_db = initial_snr_db
    
    def set_epoch(self, current_epoch: int, total_epochs: int):
        if self.schedule == 'linear':
            self.current_snr_db = self._linear_schedule(current_epoch, total_epochs)
        elif self.schedule == 'cosine':
            self.current_snr_db = self._cosine_schedule(current_epoch, total_epochs)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def _linear_schedule(self, current_epoch: int, total_epochs: int) -> float:
        progress = current_epoch / max(total_epochs - 1, 1)
        snr = self.initial_snr_db - (self.initial_snr_db - self.final_snr_db) * progress
        return snr
    
    def _cosine_schedule(self, current_epoch: int, total_epochs: int) -> float:
        progress = current_epoch / max(total_epochs - 1, 1)
        cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
        snr = self.final_snr_db + (self.initial_snr_db - self.final_snr_db) * cosine_progress
        return snr
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        target_snr = self.current_snr_db + np.random.randn() * self.snr_std
        
        if self.noise_type == 'gaussian':
            return self._add_gaussian_noise(waveform, target_snr)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _add_gaussian_noise(self, waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        signal_power = torch.mean(waveform ** 2)
        
        if signal_power < 1e-10:
            return waveform
        
        signal_power_db = 10 * torch.log10(signal_power)
        
        noise_power_db = signal_power_db - snr_db
        
        noise_power = 10 ** (noise_power_db / 10)
        
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        
        return waveform + noise
    
    def get_current_snr(self) -> float:
        return self.current_snr_db


class ComposedAudioAugmentation:
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            waveform = transform(waveform)
        return waveform
    
    def set_epoch(self, current_epoch: int, total_epochs: int):
        for transform in self.transforms:
            if hasattr(transform, 'set_epoch'):
                transform.set_epoch(current_epoch, total_epochs)


def create_mean_teacher_augmentation(
    use_volume: bool = True,
    use_noise: bool = True,
    volume_range: tuple = (0.95, 1.05),
    initial_snr: float = 25.0,
    final_snr: float = 10.0,
    noise_schedule: str = 'cosine'
) -> AudioTransformTwice:
    transforms = []
    
    if use_volume:
        transforms.append(VolumeAugmentation(*volume_range))
    
    if use_noise:
        transforms.append(CurriculumNoiseAugmentation(
            initial_snr_db=initial_snr,
            final_snr_db=final_snr,
            schedule=noise_schedule
        ))
    
    if len(transforms) == 0:
        raise ValueError("At least one augmentation must be enabled!")
    
    if len(transforms) == 1:
        composed = transforms[0]
    else:
        composed = ComposedAudioAugmentation(transforms)
    
    return AudioTransformTwice(composed)



class RobustMeanTeacherAugmentation(torch.nn.Module):
    
    def __init__(
        self,
        musan_path: str,
        rir_path: str,
        total_epochs: int,
        final_snr_db: float = 10.0,
        initial_snr_db: float = 25.0,
        augmentation_warmup_ratio: float = 0.1,
        rir_max_prob: float = 0.5,
        student_volume_db: float = 6.0,
        teacher_volume_db: float = 2.0,
        pitch_shift_prob: float = 0.5,
        megaphone_prob: float = 0.3,
        pitch_min_semitones: float = -4.0,
        pitch_max_semitones: float = 4.0,
        megaphone_min_cutoff: float = 200.0,
        megaphone_max_cutoff: float = 800.0,
        augmentation_max_epochs: int = None,
    ):
        super().__init__()
        
        if not TORCH_AUDIOMENTATIONS_AVAILABLE:
            raise ImportError(
                "RobustMeanTeacherAugmentation  torch-audiomentations！\n"
                "Installation: pip install torch-audiomentations"
            )
        
        self.total_epochs = total_epochs
        self.final_snr_db = final_snr_db
        self.initial_snr_db = initial_snr_db
        self.augmentation_warmup_ratio = augmentation_warmup_ratio
        self.rir_max_prob = rir_max_prob
        self.student_volume_db = student_volume_db
        self.teacher_volume_db = teacher_volume_db
        self.pitch_shift_prob = pitch_shift_prob
        self.megaphone_prob = megaphone_prob
        self.pitch_min_semitones = pitch_min_semitones
        self.pitch_max_semitones = pitch_max_semitones
        self.megaphone_min_cutoff = megaphone_min_cutoff
        self.megaphone_max_cutoff = megaphone_max_cutoff
        
        self.augmentation_max_epochs = augmentation_max_epochs if augmentation_max_epochs is not None else total_epochs
        
        if self.augmentation_max_epochs > self.total_epochs:
            print(f"Warning:  Warning: augmentation_max_epochs ({self.augmentation_max_epochs}) > total_epochs ({self.total_epochs})")
            print(f"   will be automatically adjusted to total_epochs")
            self.augmentation_max_epochs = self.total_epochs
        
        print("\n" + "="*70)
        print("[Audio Augmentation Initialization]Robust Mean Teacher Augmentation")
        print("="*70)
        
        safe_musan_paths = [
            os.path.join(musan_path, "noise"),
        ]
        
        existing_musan = [p for p in safe_musan_paths if os.path.exists(p)]
        if not existing_musan:
            raise FileNotFoundError(
                f"Error: MUSAN Data path does not exist！\n"
                f"   Expected path：{safe_musan_paths}\n"
                f"   Please check musan_path parameter is correct"
            )
        
        all_musan_files = []
        print(f" MUSAN Data source:")
        for path in existing_musan:
            file_count = 0
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith('.wav'):
                        all_musan_files.append(os.path.join(root, f))
                        file_count += 1
            print(f"   - {path} ({file_count} files)")
        
        if not all_musan_files:
            raise FileNotFoundError(
                f"Error: MUSAN No files found in directory .wav files！\n"
                f"   Searched：{existing_musan}"
            )
        
        simulated_rir_path = os.path.join(rir_path, "simulated_rirs")
        if not os.path.exists(simulated_rir_path):
            raise FileNotFoundError(
                f"Error: RIR Data path does not exist！\n"
                f"   Expected path：{simulated_rir_path}\n"
                f"   Please check rir_path parameter is correct"
            )
        
        all_rir_files = []
        for root, dirs, files in os.walk(simulated_rir_path):
            for f in files:
                if f.endswith('.wav'):
                    all_rir_files.append(os.path.join(root, f))
        
        print(f" RIR Data source: {simulated_rir_path}")
        print(f"   - Total: {len(all_rir_files)} files")
        
        if not all_rir_files:
            raise FileNotFoundError(
                f"Error: RIR No files found in directory .wav files！\n"
                f"   Searched：{simulated_rir_path}"
            )
        
        print(f"\n Student Augmentation strategy:")
        print(f"   - PitchShift: 0% (warmup) → {pitch_shift_prob*100:.0f}% (later)")
        print(f"   - HighPass: 0% (warmup) → {megaphone_prob*100:.0f}% (later)")
        print(f"   - RIR: 0% (early) → {rir_max_prob*100:.0f}% (later)")
        print(f"   - Noise: SNR {initial_snr_db:.0f}dB (early) → {final_snr_db:.0f}dB (later)")
        print(f"   - Volume: ±{student_volume_db:.0f}dB (always)")
        print(f"\n Curriculum learning parameters:")
        print(f"   - Total training Epochs: {total_epochs}")
        print(f"   - ⭐ Augmentation max Epochs: {self.augmentation_max_epochs}")
        if self.augmentation_max_epochs < total_epochs:
            print(f"   - 📌 Augmentation will reach max intensity at epoch {self.augmentation_max_epochs} and remain constant")
            print(f"   - 📌 Remaining {total_epochs - self.augmentation_max_epochs} epochs training with stable augmentation")
        print(f"   - warmup (0-{augmentation_warmup_ratio*100:.0f}%):  RIR, SNR={initial_snr_db:.0f}dB")
        print(f"   - Final mode: RIR={rir_max_prob*100:.0f}%, SNR={final_snr_db:.0f}dB")
        
        student_transforms = [
            PitchShift(
                min_transpose_semitones=self.pitch_min_semitones,
                max_transpose_semitones=self.pitch_max_semitones,
                sample_rate=16000,
                p=0.0,
            ),
        ]

        if HighPassFilter is not None:
            student_transforms.append(
                HighPassFilter(
                    min_cutoff_freq=self.megaphone_min_cutoff,
                    max_cutoff_freq=self.megaphone_max_cutoff,
                    sample_rate=16000,
                    p=0.0,
                )
            )
        else:
            print("Warning:  HighPassFilter ，/")

        student_transforms.extend([
            ApplyImpulseResponse(
                ir_paths=all_rir_files,
                p=0.0,
                sample_rate=16000,
            ),
            AddBackgroundNoise(
                background_paths=all_musan_files,
                min_snr_in_db=initial_snr_db,
                max_snr_in_db=initial_snr_db + 5.0,
                p=1.0,
                sample_rate=16000,
            ),
            Gain(
                min_gain_in_db=-student_volume_db,
                max_gain_in_db=student_volume_db,
                p=1.0,
                sample_rate=16000,
            ),
        ])

        self.student_augment = Compose(student_transforms)
        
        print(f"\n Teacher Augmentation strategy:")
        print(f"   - Volume: ±{teacher_volume_db:.0f}dB (keep clean)")
        print(f"   -  RIR /  Noise (stable anchor)")
        
        self.teacher_augment = Compose([
            Gain(
                min_gain_in_db=-teacher_volume_db,
                max_gain_in_db=teacher_volume_db,
                p=1.0,
                sample_rate=16000,
            )
        ])
        
        print(f"\n Curriculum learning parameters:")
        print(f"   - Total Epochs: {total_epochs}")
        print(f"   - warmup (0-{augmentation_warmup_ratio*100:.0f}%):  RIR, SNR={initial_snr_db:.0f}dB")
        print(f"   - Final mode: RIR={rir_max_prob*100:.0f}%, SNR={final_snr_db:.0f}dB")
        print("="*70 + "\n")
    
    def set_epoch(self, current_epoch: int):
        effective_epoch = min(current_epoch, self.augmentation_max_epochs)
        progress = min(1.0, effective_epoch / max(self.augmentation_max_epochs - 1, 1))
        warmup = self.augmentation_warmup_ratio

        if progress < warmup:
            ramp = 0.0
        else:
            ramp = (progress - warmup) / max(1.0 - warmup, 1e-6)
        ramp = float(np.clip(ramp, 0.0, 1.0))

        pitch_p = self.pitch_shift_prob * ramp
        highpass_p = self.megaphone_prob * ramp
        rir_p = self.rir_max_prob * ramp

        target_snr = self.initial_snr_db - (self.initial_snr_db - self.final_snr_db) * ramp
        
        for transform in self.student_augment.transforms:
            if PitchShift is not None and isinstance(transform, PitchShift):
                transform.p = pitch_p
            if HighPassFilter is not None and isinstance(transform, HighPassFilter):
                transform.p = highpass_p
            if ApplyImpulseResponse is not None and isinstance(transform, ApplyImpulseResponse):
                transform.p = rir_p
            if AddBackgroundNoise is not None and isinstance(transform, AddBackgroundNoise):
                transform.min_snr_in_db = target_snr - 2.0
                transform.max_snr_in_db = target_snr + 2.0
        
        plateau_indicator = " [PLATEAU]" if current_epoch >= self.augmentation_max_epochs else ""
        print(
            f" [Epoch {current_epoch:3d}/{self.total_epochs}] "
            f"Augm_Epoch={effective_epoch:3d}/{self.augmentation_max_epochs}{plateau_indicator} | "
            f"Progress={progress*100:5.1f}% | Warmup={progress < self.augmentation_warmup_ratio} | "
            f"Pitch_p={pitch_p:.2f} | HighPass_p={highpass_p:.2f} | "
            f"RIR_p={rir_p:.2f} | Noise_SNR={target_snr:.1f}dB"
        )
    
    def __call__(self, waveform: torch.Tensor) -> tuple:
        original_shape = waveform.shape
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        elif waveform.dim() == 3:
            pass
        else:
            raise ValueError(f"Unsupported audio dimension: {waveform.shape}")
        
        teacher_view = self.teacher_augment(waveform)
        
        prev_det_state = torch.are_deterministic_algorithms_enabled()
        if prev_det_state:
            torch.use_deterministic_algorithms(False)

        try:
            student_view = self.student_augment(waveform)
        finally:
            if prev_det_state:
                torch.use_deterministic_algorithms(True)
        
        if len(original_shape) == 1:
            student_view = student_view.squeeze()
            teacher_view = teacher_view.squeeze()
        elif len(original_shape) == 2:
            student_view = student_view.squeeze(1)
            teacher_view = teacher_view.squeeze(1)
        
        return student_view, teacher_view
    
    def get_augmentation_info(self) -> dict:
        rir_prob = 0.0
        pitch_prob = 0.0
        highpass_prob = 0.0
        noise_min_snr = 25.0
        noise_max_snr = 30.0
        
        for transform in self.student_augment.transforms:
            if PitchShift is not None and isinstance(transform, PitchShift):
                pitch_prob = transform.p
            if HighPassFilter is not None and isinstance(transform, HighPassFilter):
                highpass_prob = transform.p
            if ApplyImpulseResponse is not None and isinstance(transform, ApplyImpulseResponse):
                rir_prob = transform.p
            if AddBackgroundNoise is not None and isinstance(transform, AddBackgroundNoise):
                noise_min_snr = transform.min_snr_in_db
                noise_max_snr = transform.max_snr_in_db
        
        return {
            'total_epochs': self.total_epochs,
            'final_snr_db': self.final_snr_db,
            'current_pitch_prob': pitch_prob,
            'current_highpass_prob': highpass_prob,
            'current_rir_prob': rir_prob,
            'current_noise_snr': (noise_min_snr, noise_max_snr),
            'student_volume_range': (-self.student_volume_db, self.student_volume_db),
            'teacher_volume_range': (-self.teacher_volume_db, self.teacher_volume_db),
        }
