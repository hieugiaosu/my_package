import pandas as pd
import torch
from asteroid.data import LibriMix


class Libri2MixWithAsteroid(LibriMix):
    def __init__(self, csv_dir, task="sep_clean", sample_rate=16000, n_src=2, segment=3, return_embeddings=False, embeddings_dir=None):
        super().__init__(csv_dir, task, sample_rate, n_src, segment, True)
        self.return_embeddings = return_embeddings
        if return_embeddings:
            self.embeddings = pd.read_csv(embeddings_dir)
            self.embeddings = {str(k): torch.tensor(eval(v)) for k, v in zip(self.embeddings['speaker'], self.embeddings['embed'])}
    def __len__(self):
        return 2*super().__len__()
    
    def __getitem__(self, idx):
        src_idx = idx // 2

        # Get the mixture and the sources
        mix, source, id_speaker = super().__getitem__(src_idx)

        source = source[0] if idx % 2 != 0 else source[1]
        speaker_id = str(id_speaker[0]) if idx % 2 != 0 else str(id_speaker[1])
        if self.return_embeddings:
            return mix, source, self.embeddings[speaker_id]
        else:
            return mix, source