import math
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterator, Optional, Callable, Any, Dict, cast
import numpy as np


class StreamingMixedDataset(IterableDataset):
    
    def __init__(
        self,
        labeled_dataset: IterableDataset,
        unlabeled_dataset: IterableDataset,
        labeled_batch_size: int,
        total_batch_size: int,
    ):
        super().__init__()
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = total_batch_size - labeled_batch_size
        self.total_batch_size = total_batch_size
        
        assert self.labeled_batch_size > 0, "labeled_batch_size must be > 0"
        assert self.unlabeled_batch_size > 0, "unlabeled_batch_size must be > 0"
        
        print(f"\n{'='*70}")
        print("[Streaming Mixed Dataset] Initialization (Mean Teacher)")
        print(f"{'='*70}")
        print(f"Following TwoStreamBatchSampler logic:")
        print(f"   - Unlabeled data: iterate_once (iterate once, define epoch)")
        print(f"   - Labeled data: iterate_eternally (iterate eternally)")
        print(f"")
        print(f"Batch Configuration:")
        print(f"   - Total batch size: {self.total_batch_size}")
        print(f"   - Labeled samples/batch: {self.labeled_batch_size}")
        print(f"   - Unlabeled samples/batch: {self.unlabeled_batch_size}")
        print(f"   - Labeled ratio: {self.labeled_batch_size / self.total_batch_size * 100:.1f}%")
        print(f"")
        print(f" Memory advantage:")
        print(f"   - Traditional approach: Load all data to memory (GB)")
        print(f"   - Streaming approach: Only keep current batch (~20MB)")
        print(f"{'='*70}\n")

        def safe_length(dataset):
            if hasattr(dataset, "__len__"):
                try:
                    value = dataset.__len__()
                    return int(value)
                except Exception:
                    return None
            return None

        total_unlabeled = safe_length(unlabeled_dataset)
        total_labeled = safe_length(labeled_dataset)

        self.total_unlabeled_samples = total_unlabeled
        self.total_labeled_samples = total_labeled

        def batches(total, batch_size):
            if total is None or total <= 0:
                return None
            effective = total // batch_size
            if effective <= 0:
                return 1 if total > 0 else None
            return effective

        unlabeled_batches = batches(total_unlabeled, self.unlabeled_batch_size)
        labeled_batches = batches(total_labeled, self.labeled_batch_size)

        if unlabeled_batches and unlabeled_batches > 0:
            self.num_batches_per_epoch = unlabeled_batches
        elif labeled_batches:
            self.num_batches_per_epoch = labeled_batches
        else:
            self.num_batches_per_epoch = None
    
    def __iter__(self) -> Iterator:
        iterate_once = getattr(self.unlabeled_dataset, "iterate_once", None)
        if callable(iterate_once):
            unlabeled_iter = cast(Iterator[Dict[str, Any]], iterate_once())
        else:
            unlabeled_iter = cast(Iterator[Dict[str, Any]], iter(self.unlabeled_dataset))
        
        labeled_iter = self._iterate_eternally(self.labeled_dataset)
        unlabeled_exhausted = False
        batches_emitted = 0
        
        while not unlabeled_exhausted:
            if self.num_batches_per_epoch is not None and batches_emitted >= self.num_batches_per_epoch:
                break
            batch_samples = []
            unlabeled_count = 0
            
            while unlabeled_count < self.unlabeled_batch_size and not unlabeled_exhausted:
                try:
                    sample = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_exhausted = True
                    break
                if sample is None:
                    continue
                sample['is_labeled'] = False
                batch_samples.append(sample)
                unlabeled_count += 1
            
            if unlabeled_count < self.unlabeled_batch_size:
                break
            
            for _ in range(self.labeled_batch_size):
                try:
                    sample = next(labeled_iter)
                    sample['is_labeled'] = True
                    batch_samples.append(sample)
                except StopIteration:
                    print("Warning:  Labeled data")
                    return
            
            np.random.shuffle(batch_samples)
            for sample in batch_samples:
                yield sample

            batches_emitted += 1
        
        return

    def __len__(self) -> int:
        if self.num_batches_per_epoch is None:
            raise TypeError("StreamingMixedDataset length is undefined (unlabeled dataset length unknown)")
        return self.num_batches_per_epoch * self.total_batch_size
    
    def _iterate_eternally(self, dataset: IterableDataset) -> Iterator:
        while True:
            for sample in dataset:
                yield sample


def create_streaming_mixed_dataloader(
    labeled_dataset: IterableDataset,
    unlabeled_dataset: IterableDataset,
    task,
    batch_size: int = 32,
    labeled_batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    audio_augmentation: Optional[Callable] = None,
) -> DataLoader:
    mixed_dataset = StreamingMixedDataset(
        labeled_dataset=labeled_dataset,
        unlabeled_dataset=unlabeled_dataset,
        labeled_batch_size=labeled_batch_size,
        total_batch_size=batch_size,
    )
    
    def augmented_collate_fn(samples):
        labeled_samples = []
        unlabeled_samples = []
        
        for sample in samples:
            is_labeled = sample.pop('is_labeled')
            if is_labeled:
                labeled_samples.append(sample)
            else:
                unlabeled_samples.append(sample)

        labeled_meta_list = [s.get('meta', {}) for s in labeled_samples] if labeled_samples else []
        unlabeled_meta_list = [s.get('meta', {}) for s in unlabeled_samples] if unlabeled_samples else []
        
        labeled_batch: Optional[Dict[str, Any]]
        unlabeled_batch: Optional[Dict[str, Any]]

        if labeled_samples:
            labeled_batch = task.collate_fn(labeled_samples, stage="train")
        else:
            labeled_batch = None
        
        if unlabeled_samples:
            X_list = [s['X'] for s in unlabeled_samples]
            unlabeled_X = torch.stack(X_list, dim=0)
            unlabeled_batch = {'X': unlabeled_X}
            unlabeled_batch['meta_list'] = unlabeled_meta_list
        else:
            unlabeled_batch = None
        
        if labeled_batch is not None and unlabeled_batch is not None:
            batch_size_total = len(labeled_samples) + len(unlabeled_samples)
            
            labeled_indices = []
            unlabeled_indices = []
            idx = 0
            for sample in samples:
                pass
            
            all_X = torch.cat([labeled_batch['X'], unlabeled_batch['X']], dim=0)
            
            all_y = torch.zeros(
                (batch_size_total, *labeled_batch['y'].shape[1:]),
                dtype=labeled_batch['y'].dtype,
                device=labeled_batch['y'].device
            )
            all_y[:len(labeled_samples)] = labeled_batch['y']
            
            is_labeled = torch.zeros(batch_size_total, dtype=torch.bool)
            is_labeled[:len(labeled_samples)] = True
            
            batch = {
                'X': all_X,
                'y': all_y,
                'is_labeled': is_labeled,
            }
            
            meta_list = labeled_meta_list + unlabeled_meta_list
            if meta_list:
                batch['meta_list'] = meta_list
            if 'meta' in labeled_batch:
                batch['meta'] = labeled_batch.get('meta', {})
        
        elif labeled_batch is not None:
            batch = labeled_batch
            batch['is_labeled'] = torch.ones(len(labeled_samples), dtype=torch.bool)
            meta_list = labeled_meta_list
            if meta_list:
                batch['meta_list'] = meta_list
        
        elif unlabeled_batch is not None:
            batch = {
                'X': unlabeled_batch['X'],
                'is_labeled': torch.zeros(len(unlabeled_samples), dtype=torch.bool),
            }
            meta_list = unlabeled_meta_list
            if meta_list:
                batch['meta_list'] = meta_list
        else:
            raise ValueError("Empty batch")
        
        
        batch['X_original'] = batch['X']
        
        return batch
    
    dataloader = DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=augmented_collate_fn,
    )
    
    return dataloader
