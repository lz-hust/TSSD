from typing import Iterator, Union, Dict
from pyannote.database.protocol.protocol import Protocol, ProtocolFile


class SemiSupervisedProtocol(Protocol):
    
    def unlabeled_train_iter(self) -> Iterator[Union[Dict, ProtocolFile]]:
        raise NotImplementedError(
            "This protocol does not implement unlabeled training data support."
            "Please implement 'unlabeled_train_iter()' method in subclass."
        )
    
    def unlabeled_train(self) -> Iterator[ProtocolFile]:
        try:
            files = self.unlabeled_train_iter()
        except (AttributeError, NotImplementedError) as e:
            import warnings
            warnings.warn(
                f"Unlabeled training data unavailable: {e}. "
                "Training will continue in supervised mode (using only labeled data)."
            )
            return iter([])
        
        for file in files:
            yield self.preprocess(file)
    
    def has_unlabeled_train(self) -> bool:
        try:
            next(iter(self.unlabeled_train()))
            return True
        except (StopIteration, NotImplementedError):
            return False


class MixedDataProtocol(SemiSupervisedProtocol):
    
    def __init__(self, preprocessors=None, labeled_ratio: float = 1.0):
        super().__init__(preprocessors=preprocessors)
        self.labeled_ratio = labeled_ratio
    
    def mixed_train(self) -> Iterator[ProtocolFile]:
        import itertools
        
        labeled_iter = self.train()
        unlabeled_iter = self.unlabeled_train()
        
        def add_label_flag(file_iter, is_labeled):
            for file in file_iter:
                file['is_labeled'] = is_labeled
                yield file
        
        labeled_flagged = add_label_flag(labeled_iter, True)
        unlabeled_flagged = add_label_flag(unlabeled_iter, False)
        
        ratio_int = int(self.labeled_ratio) if self.labeled_ratio >= 1.0 else int(1.0 / self.labeled_ratio)
        
        if self.labeled_ratio >= 1.0:
            
            while True:
                labeled_batch = list(itertools.islice(labeled_flagged, ratio_int))
                if not labeled_batch:
                    yield from unlabeled_flagged
                    break
                
                yield from labeled_batch
                
                unlabeled_file = next(unlabeled_flagged, None)
                if unlabeled_file is None:
                    yield from labeled_flagged
                    break
                
                yield unlabeled_file
        else:
            
            while True:
                unlabeled_batch = list(itertools.islice(unlabeled_flagged, ratio_int))
                if not unlabeled_batch:
                    yield from labeled_flagged
                    break
                
                yield from unlabeled_batch
                
                labeled_file = next(labeled_flagged, None)
                if labeled_file is None:
                    yield from unlabeled_flagged
                    break
                
                yield labeled_file
