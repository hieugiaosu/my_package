from modal import App, Image, Volume

image = (
    Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel", add_python="3.11")
    .pip_install( "numpy", "pandas","einops"
                 "asteroid", "librosa", "soundfile", "pydub", "resemblyzer"
                 )
)

volume = Volume.from_name("libri2mix", create_if_missing=True)

app = App()

libri_dir = "/libri2mix"
@app.function(image=image, gpu=None, volumes={libri_dir: volume}, timeout=3600 * 24)
def create_embedding_df():
    import os

    import numpy as np
    import pandas as pd
    import torch
    from resemblyzer import VoiceEncoder, preprocess_wav

    voice_encoder = VoiceEncoder()

    csv_dir = "/libri2mix/Libri2Mix/wav16k/min/metadata"

    csv_file = os.listdir(csv_dir)
    csv_file = [f for f in csv_file if f.startswith("mixture_") and f.endswith("both.csv")]
    for filename in csv_file:
        print(f"Processing {filename}")
        embedding_data = []
        df = pd.read_csv(os.path.join(csv_dir, filename))
        df = df.to_dict(orient='records')
        for idx,record in enumerate(df):
            print(f"Processing {idx+1}/{len(df)}")
            mixture_id = record['mixture_ID']
            sp1_id, sp2_id = mixture_id.split('_')
            source1_path = record['source_1_path']
            source2_path = record['source_2_path']
            wav1 = preprocess_wav(source1_path)
            wav2 = preprocess_wav(source2_path)
            embed1 = voice_encoder.embed_utterance(wav1)
            embed2 = voice_encoder.embed_utterance(wav2)
            embedding_data.append({
                'speaker': sp1_id,
                'embed': embed1.tolist()
            })
            embedding_data.append({
                'speaker': sp2_id,
                'embed': embed2.tolist()
            })

        df = pd.DataFrame(embedding_data)
        df.to_csv(f"/libri2mix/{filename.replace('mixture','embedding')}", index=False)
        volume.commit()

@app.local_entrypoint()
def main():
    create_embedding_df.remote()