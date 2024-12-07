import modal
from modal import App, Image, Mount, Volume

image = (
    Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel", add_python="3.11")
    .pip_install( "numpy", "pandas","resemblyzer",
                 "asteroid", "librosa", "soundfile", "pydub",
                 "einops","tensorboard"
                 )
)

volume = Volume.from_name("libri2mix", create_if_missing=True)
libri_dir = "/libri2mix"
model_volume = Volume.from_name("reproduce", create_if_missing=True)
model_dir = "/model"

mounts = [
    Mount.from_local_dir('./WHYV2', remote_path='/root/WHYV2'),
]

app = App(mounts=mounts)

@app.function(image=image, gpu="A100", volumes={libri_dir: volume, model_dir: model_volume}, timeout=3600 * 24)
def test_train():
    import gc
    import time

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from asteroid.data import LibriMix
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from WHYV2.criterion.si_sdr import SISDRLossWithPIT
    from WHYV2.data.module import Libri2MixWithAsteroid
    from WHYV2.network.models.rotate_and_scale_model import \
        WHYVwithRotateCondition
    from WHYV2.network.models.whyv.whyv_version1 import TF_Gridnet
    from WHYV2.training import TrainingPipeline
    train_dataset = LibriMix(
        csv_dir='/libri2mix/Libri2Mix/wav8k/min/metadata/train',
        task="sep_noisy",
        sample_rate=8000,
        n_src=2,
        segment=4,
    )
    
    val_dataset = LibriMix(
        csv_dir='/libri2mix/Libri2Mix/wav8k/min/metadata/dev',
        task="sep_noisy",
        sample_rate=8000,
        n_src=2,
        segment=4,
    )


    model = TF_Gridnet(n_layers=4, n_srcs=2)
    def commit_func():
        model_volume.commit()
    class Pipeline(TrainingPipeline):
        def __init__(
                self, 
                model: nn.Module, 
                train_dataset, 
                val_dataset, 
                optimizer="AdamW", 
                optimizer_param={
                    "lr":1e-4,
                    "weight_decay":1.0e-2
                }, 
                train_batch_size=8, 
                val_batch_size=8, 
                epochs=200, 
                time_limit=86400, 
                device=None, 
                using_multi_gpu=True, 
                checkpoint_path="/", 
                checkpoint_name="model.pth", 
                checkpoint_rate=1, 
                patient=3, 
                checkpoint_from_epoch=1, 
                use_checkpoint=None,
                train_dataloader_class = DataLoader,
                val_dataloader_class = DataLoader,
                checkpoint_call_back = None
                ):
            super().__init__(model, train_dataset, val_dataset, optimizer, optimizer_param, train_batch_size, val_batch_size, epochs, time_limit, device, using_multi_gpu, checkpoint_path, checkpoint_name, checkpoint_rate, patient, checkpoint_from_epoch, use_checkpoint, train_dataloader_class, val_dataloader_class,checkpoint_call_back)
            print("This pipeline is train in mixed precision")
            self.si_sdr_fn = SISDRLossWithPIT(reduction="mean")
            self.scaler = GradScaler('cuda')
            self.writer = SummaryWriter(log_dir=f"{checkpoint_path}/tensorboard_logs")
            self.count = 0

        def train_iter(self,epoch,start_time):
            print(f"----------------------------train---------epoch: {epoch}/{self.epochs}--------------------------")
            if self.using_multi_gpu:
                self.model.module.train()
            else:
                self.model.train()
            
            tot_loss, num_batch = 0, 0
            total_batch = len(self.train_loader)

            for data in self.train_loader:
                self.count += 1
                with autocast(device_type='cuda',enabled=False):
                    self.optimizer.zero_grad()
                    mix, src0 = data
                    mix = mix.to(self.device).float()
                    src0 = src0.to(self.device).float()
                    num_batch += 1
                    # Use autocast for mixed precision
                    yHat = self.model(mix)
                    loss = self.si_sdr_fn(yHat,src0)
                        
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.module.parameters() if self.using_multi_gpu else self.model.parameters(), 
                        5.0, norm_type=2.0, error_if_nonfinite=False, foreach=None
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    tot_loss += loss.cpu().detach().item()
                    self.writer.add_scalar("Loss/Train-batch", loss.cpu().detach().item(),self.count)
                    print(f"--------------batch:{num_batch}/{total_batch}---------loss:{loss.cpu().detach().item()}----------")
                    del mix, src0, yHat, loss
                    torch.cuda.empty_cache()
                    gc.collect()
                    if time.time() - start_time > self.time_limit:
                        print('-------------------out of time-----------------------')
                        break
            return tot_loss / num_batch, num_batch
        
        def validate_iter(self):
            print("-------------------------validate---------------------------")
            if self.using_multi_gpu:
                self.model.module.eval()
            else:
                self.model.eval()
            tot_loss, num_batch = 0, 0
            with torch.no_grad():
                for data in self.val_loader:
                    mix, src0 = data
                    mix = mix.to(self.device).float()
                    src0 = src0.to(self.device).float()
                    num_batch += 1
                    with autocast(device_type='cuda',enabled=False):  # Use autocast for mixed precision
                        yHat = self.model(mix)
                        loss = self.si_sdr_fn(yHat, src0)
                    tot_loss += loss.cpu().detach().item()
                    del mix, src0, yHat, loss
                    torch.cuda.empty_cache()
                    gc.collect()
            return tot_loss / num_batch, num_batch
        
        def train(self,initial_loss = 40):
            self.count = 0
            best_loss = initial_loss
            count = 0
            start_time = time.time()
            for epoch in range(1, self.epochs + 1):
                train_start_time = time.time()
                train_loss, train_num_batch = self.train_iter(epoch,start_time)
                train_end_time = time.time()
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                print(f"[TRAIN] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss = {train_loss:.4f} dB | Speed = ({train_end_time - train_start_time:.2f}s/{train_num_batch:d})")
                if epoch % self.checkpoint_rate == 0 and epoch >= self.checkpoint_from_epoch:
                    valid_start_time = time.time()
                    val_loss, valid_num_batch = self.validate_iter()
                    valid_end_time = time.time()
                    
                    self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                    print(f"[VALID] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss (SI-SDR) = {val_loss:.4f} dB | Speed = ({valid_end_time - valid_start_time:.2f}s/{valid_num_batch:d})")
                    if val_loss < best_loss:
                        self.checkpoint()
                        count = 0
                        best_loss = val_loss
                    else:
                        count += 1
                        if count > self.patient:
                            print('early stopping because loss is not decreasing')
                            break
                if time.time() - start_time > self.time_limit:
                    print("-------------out of time------------------")
                    break

    pipeline = Pipeline(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer="AdamW",
        val_batch_size=8,
        train_batch_size=8,
        optimizer_param={
            "lr":8e-4,
            "weight_decay":1.0e-3
        },
        epochs=20,
        time_limit=86400,
        device="cuda",
        using_multi_gpu=False,
        checkpoint_path=model_dir+'/',
        checkpoint_name="tf_gridnet.pth",
        checkpoint_rate=1,
        patient=2,
        checkpoint_from_epoch=1,
        checkpoint_call_back=commit_func,
    )

    pipeline.train(0.5)

# @app.function(image=image,volumes={model_dir: model_volume})
# @modal.web_endpoint()
# async def serve_tensorboard():
#     import subprocess
#     """Serve TensorBoard logs via a Modal web endpoint."""
#     log_dir = "/model/tensorboard_logs"
#     subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006", "--host", "0.0.0.0"])
#     return {"url": "tensorboard is running. Access it at the endpoint."}


@app.local_entrypoint()
def main():
    test_train.remote()