from modal import App, Image, Mount, Volume
import modal
import modal.gpu

image = (
    Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "numpy==1.23.5", "einops", "pandas==1.5.3", "scikit-learn==1.2.2", "resemblyzer"
    )
    # .run_commands(
    #     "pip install -U git+https://github.com/sustcsonglin/flash-linear-attention",
    # )
)
mounts = [
    Mount.from_local_dir("C:/Users/User/Documents/ĐH/Tốt nghiệp/WHYV2/WHYV2",remote_path='/root/WHYV2'),
    # Mount.from_local_file('C:/Users/User/Documents/ĐH/Tốt nghiệp/WHYV2/whyv3.pth', remote_path='/root/init.pth'),
    ]
app = App(mounts=mounts)

volume = Volume.from_name(
    "whyv3", create_if_missing=True
)
data_volume = Volume.from_name(
    "wham", create_if_missing=True
)
libri_volumn = Volume.from_name(
    "librispeech", create_if_missing=True
)

MODEL_DIR = "/model"
DATA_DIR = "/data"
LIBRI_DIR = "/librispeech"
@app.function(image=image, gpu=modal.gpu.A100(count=1, size="80GB"), volumes={MODEL_DIR: volume, DATA_DIR: data_volume, LIBRI_DIR:libri_volumn}, timeout=3600 * 24)
def train_whyv3():
    def commit_func():
        volume.commit()
    from WHYV2.network.models.whyv3 import WHYV3
    from WHYV2.data.module import Cluster, TrainDataLoaderWithCluster, ValDatasetWithCluster, TrainDatasetWithCluster, WhamNoiseAugmentation
    from WHYV2.data.metadata import CLUSTER_META_DATA
    from WHYV2.training import WHYV3Pipeline
    import torch
    import numpy as np
    from resemblyzer import VoiceEncoder

    class ResemblyzerVoiceEncoder:
        def __init__(self, device) -> None:
            self.model = VoiceEncoder(device)
            
        def __call__(self, audio: torch.Tensor):
            if audio.ndimension() == 1:
                return torch.tensor(self.model.embed_utterance(audio.numpy())).float().cpu()
            else:
                e = torch.stack([torch.tensor(self.model.embed_utterance(audio[i,:].numpy())).float().cpu() 
                                for i in range(audio.shape[0])])
                return e
    class MyDS(TrainDatasetWithCluster):
        def __init__(self,cluster,embedding_model,augmentation = None,num_speaker_per_cluster:int=6, sampling_rate = 8000, log = True, emb_mix = True):
            super().__init__(cluster,embedding_model,augmentation,num_speaker_per_cluster,sampling_rate,log,emb_mix)
        def reset(self):
            if len(self) == 0:
                super().reset()
            else:
                print('end epoch')
        def __len__(self):
            return super().__len__()
        def __getitem__(self,idx):
            return super().__getitem__(idx)
    
    root = f"{LIBRI_DIR}/train-clean-100/LibriSpeech/train-clean-100/"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = WHYV3(n_layers=5)
    speaker_embedder = ResemblyzerVoiceEncoder(device)
    cluster_list = [Cluster(i, CLUSTER_META_DATA[i].DISTANCE_META, root, cluster_meta=CLUSTER_META_DATA[i].AUDIO_FILE_META, val_size=0.02, use_ratio=0.4) for i in CLUSTER_META_DATA.keys()]
    train_augmentation = WhamNoiseAugmentation(
        f"{DATA_DIR}/wham_noise/tr/",
        f"{DATA_DIR}/wham_noise/metadata/noise_meta_tr.csv",
        (5, 20),8000,4
    )
    val_augmentation = WhamNoiseAugmentation(
        f"{DATA_DIR}/wham_noise/cv/",
        f"{DATA_DIR}/wham_noise/metadata/noise_meta_cv.csv",
        (5, 20),8000,4
    )
    np.random.seed(14022003)
    train_set = MyDS(cluster_list, speaker_embedder, num_speaker_per_cluster=18,emb_mix=False)
    val_set = ValDatasetWithCluster(cluster_list, speaker_embedder,emb_mix=False)

    print(len(train_set), len(val_set))

    pipe = WHYV3Pipeline(
        model=model,
        train_dataset=train_set,
        val_dataset=val_set,
        optimizer="AdamW",
        optimizer_param={"lr": 1e-4, "weight_decay": 1.0e-3},
        train_batch_size=12,
        val_batch_size=12,
        epochs=8,
        time_limit=3600 * 24,
        device=device,
        using_multi_gpu=False,
        checkpoint_path=MODEL_DIR+'/',
        checkpoint_name="whyv3.pth",
        use_checkpoint=MODEL_DIR+'/whyv3.pth',
        checkpoint_rate=1,
        patient=2,
        checkpoint_from_epoch=1,
        train_dataloader_class=TrainDataLoaderWithCluster,
        warm_up=100,
        checkpoint_call_back=commit_func,
        wham_train=train_augmentation,
        wham_val=val_augmentation
    )

    pipe.train(-12.5346)

@app.local_entrypoint()
def main():
    train_whyv3.remote()