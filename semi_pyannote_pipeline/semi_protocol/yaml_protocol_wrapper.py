import sys
from semi_protocol.semi_protocol import SemiSupervisedProtocol
from pyannote.database import registry
from pyannote.audio.core.io import Audio
from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.database.protocol.speaker_diarization import SpeakerDiarizationProtocol
from typing import Iterator, Union, Dict, Optional
import os


class YAMLProtocolWrapper(SemiSupervisedProtocol, SpeakerDiarizationProtocol):
    
    def __init__(
        self,
        yaml_protocol_name: str,
        unlabeled_uri_file: Optional[str] = None,
        audio_template: Optional[str] = None,
        database_root: Optional[str] = None,
        preprocessors=None
    ):
        super().__init__(preprocessors=preprocessors)
        
        self.yaml_protocol = registry.get_protocol(
            yaml_protocol_name, 
            preprocessors=preprocessors
        )
        
        self.name = yaml_protocol_name
        
        self.unlabeled_uri_file = unlabeled_uri_file
        self.audio_template = audio_template
        self.database_root = database_root or ''
    
    def train_iter(self) -> Iterator[Union[Dict, ProtocolFile]]:
        return self.yaml_protocol.train_iter()
    
    def development_iter(self) -> Iterator[Union[Dict, ProtocolFile]]:
        return self.yaml_protocol.development_iter()
    
    def test_iter(self) -> Iterator[Union[Dict, ProtocolFile]]:
        return self.yaml_protocol.test_iter()
    
    def unlabeled_train_iter(self) -> Iterator[ProtocolFile]:
        import os
        import torchaudio
        from pyannote.core import Timeline, Segment
        
        if not self.unlabeled_uri_file:
            return iter([])
        
        if self.database_root:
            uri_file_path = os.path.join(self.database_root, self.unlabeled_uri_file)
        else:
            uri_file_path = self.unlabeled_uri_file
        
        if not os.path.exists(uri_file_path):
            raise FileNotFoundError(f"Unlabeled data file does not exist: {uri_file_path}")
        
        with open(uri_file_path, 'r') as f:
            for line in f:
                uri = line.strip()
                if not uri or uri.startswith('#'):
                    continue
                
                database_name = self.name.split('.')[0] if hasattr(self, 'name') else 'Unknown'
                data = {
                    "uri": uri,
                    "database": database_name
                }
                
                if self.audio_template:
                    subset = 'train'
                    
                    audio_path = self.audio_template.format(
                        uri=uri, 
                        subset=subset
                    )
                    if self.database_root:
                        audio_path = os.path.join(self.database_root, audio_path)
                    
                    data['audio'] = audio_path
                    
                    try:
                        info = torchaudio.info(audio_path)
                        audio_duration = info.num_frames / info.sample_rate
                        data['annotated'] = Timeline([Segment(0, audio_duration)], uri=uri)
                    except Exception as e:
                        print(f"Warning: Cannot get audio duration: {audio_path}, error: {e}")
                        continue
                
                yield data
    
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
