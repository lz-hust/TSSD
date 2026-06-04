import torch
import torch.nn.functional as F
from pyannote.audio.tasks import SpeakerDiarization
from typing import Optional, Sequence, Text, Tuple, Union
import numpy as np
from pyannote.audio.utils.permutation import permutate
from sklearn.metrics import precision_recall_fscore_support

class SpeakerDiarization_bet(SpeakerDiarization):
    """Speaker diarization task with change point detection as an auxiliary task."""
    
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
        **kwargs,
    ):
        super().__init__(
            protocol=protocol,
            duration=duration,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.change_weight = change_weight

    def collate_is_change(self, batch, collated_y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        collated_is_change = []
        
        for y in collated_y:
            is_change = (y[1:] != y[:-1]).any(dim=-1).float()
            collated_is_change.append(is_change)
        
        
        collated_is_change = torch.stack(collated_is_change)
        
       
        total_samples = collated_is_change.numel()
        positive_samples = torch.sum(collated_is_change).item()
        negative_samples = total_samples - positive_samples
        positive_ratio = positive_samples / total_samples if total_samples > 0 else 0.0
        negative_ratio = negative_samples / total_samples if total_samples > 0 else 0.0

        
        return collated_is_change, positive_ratio

    def focal_loss(self, inputs, targets, alpha=0.1, gamma=2.0, reduction='mean'):
        """Compute Focal Loss for change point detection."""
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss
        if reduction == 'mean':
            return F_loss.mean()
        return F_loss.sum()

    def training_step(self, batch, batch_idx: int):
        """Compute main segmentation loss, VAD loss (if enabled), and change point loss."""
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

        if self.specifications.powerset:
            multilabel = self.model.powerset.to_multilabel(main_output)
            permutated_target, _ = permutate(multilabel, target)
            permutated_target_powerset = self.model.powerset.to_powerset(permutated_target.float())
            seg_loss = self.segmentation_loss(main_output, permutated_target_powerset, weight=weight)
        else:
            permutated_prediction, _ = permutate(target, main_output)
            seg_loss = self.segmentation_loss(permutated_prediction, target, weight=weight)

        vad_loss = 0.0
        if self.vad_loss is not None:
            if self.specifications.powerset:
                vad_loss = self.voice_activity_detection_loss(main_output, permutated_target_powerset, weight=weight)
            else:
                vad_loss = self.voice_activity_detection_loss(permutated_prediction, target, weight=weight)

        # Focal Loss
        change_loss = self.focal_loss(
            change_point_output[:, :-1, 0],
            change_point_target,
            alpha=positive_ratio if positive_ratio > 0 else 0.25,
            gamma=2.0,
            reduction='mean'
        )

        
        total_loss = seg_loss + vad_loss + self.change_weight * change_loss

        
        preds = (torch.sigmoid(change_point_output[:, :-1, 0]) > 0.5).float().cpu().numpy()
        targets = change_point_target.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets.flatten(), preds.flatten(), average='binary', zero_division=0
        )
        self.model.log("metrics/train/change_precision", precision, on_step=False, on_epoch=True)
        self.model.log("metrics/train/change_recall", recall, on_step=False, on_epoch=True)
        self.model.log("metrics/train/change_f1", f1, on_step=False, on_epoch=True)

        self.model.log("loss/train/segmentation", seg_loss, on_step=False, on_epoch=True)
        if self.vad_loss is not None:
            self.model.log("loss/train/vad", vad_loss, on_step=False, on_epoch=True)
        self.model.log("loss/train/change", change_loss, on_step=False, on_epoch=True)
        self.model.log("loss/train", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx: int):
        """Validate main task and change point task, log losses."""
        target = batch["y"]
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

        if self.specifications.powerset:
            multilabel = self.model.powerset.to_multilabel(main_output)
            permutated_target, _ = permutate(multilabel, target)
            permutated_target_powerset = self.model.powerset.to_powerset(permutated_target.float())
            seg_loss = self.segmentation_loss(main_output, permutated_target_powerset, weight=weight)
        else:
            permutated_prediction, _ = permutate(target, main_output)
            seg_loss = self.segmentation_loss(permutated_prediction, target, weight=weight)

        vad_loss = 0.0
        if self.vad_loss is not None:
            if self.specifications.powerset:
                vad_loss = self.voice_activity_detection_loss(main_output, permutated_target_powerset, weight=weight)
            else:
                vad_loss = self.voice_activity_detection_loss(permutated_prediction, target, weight=weight)

        
        change_loss = self.focal_loss(
            change_point_output[:, :-1, 0],
            change_point_target,
            alpha=positive_ratio if positive_ratio > 0 else 0.25,
            gamma=2.0,
            reduction='mean'
        )

        
        total_loss = seg_loss + vad_loss + self.change_weight * change_loss

        
        preds = (torch.sigmoid(change_point_output[:, :-1, 0]) > 0.5).float().cpu().numpy()
        targets = change_point_target.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets.flatten(), preds.flatten(), average='binary', zero_division=0
        )
        self.model.log("metrics/val/change_precision", precision, on_step=False, on_epoch=True)
        self.model.log("metrics/val/change_recall", recall, on_step=False, on_epoch=True)
        self.model.log("metrics/val/change_f1", f1, on_step=False, on_epoch=True)

        self.model.log("loss/val/segmentation", seg_loss, on_step=False, on_epoch=True)
        if self.vad_loss is not None:
            self.model.log("loss/val/vad", vad_loss, on_step=False, on_epoch=True)
        self.model.log("loss/val/change", change_loss, on_step=False, on_epoch=True)
        self.model.log("loss/val", total_loss, on_step=False, on_epoch=True)

        if self.specifications.powerset:
            self.model.validation_metric(
                torch.transpose(multilabel[:, warm_up_left:num_frames - warm_up_right], 1, 2),
                torch.transpose(target[:, warm_up_left:num_frames - warm_up_right], 1, 2),
            )
        else:
            self.model.validation_metric(
                torch.transpose(main_output[:, warm_up_left:num_frames - warm_up_right], 1, 2),
                torch.transpose(target[:, warm_up_left:num_frames - warm_up_right], 1, 2),
            )
        self.model.log_dict(self.model.validation_metric, on_step=False, on_epoch=True, prog_bar=True)