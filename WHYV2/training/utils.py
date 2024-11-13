from abc import ABC
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import gc

class TrainingPipeline(ABC):
    def __init__(
            self,
            model:torch.nn.Module,
            train_dataset,
            val_dataset,
            optimizer = "AdamW",
            optimizer_param = {
                "lr" : 1e-3
            },
            train_batch_size = 8,
            val_batch_size = 8,
            epochs = 200,
            time_limit = 86400,
            device = None,
            using_multi_gpu = True,
            checkpoint_path = "/",
            checkpoint_name = "model.pth",
            checkpoint_rate = 1,
            patient = 3,
            checkpoint_from_epoch = 1,
            use_checkpoint = None,
            train_dataloader_class = DataLoader,
            val_dataloader_class = DataLoader,
            checkpoint_call_back = None
            ):
        super().__init__()
        self.model = model
        if use_checkpoint:
            self.model.load_state_dict(torch.load(use_checkpoint,map_location='cpu'))
            torch.cuda.empty_cache()
            gc.collect()
        self.device = device
        self.using_multi_gpu = using_multi_gpu
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cuda':
            if self.using_multi_gpu:
                if torch.cuda.device_count() == 1:
                    self.using_multi_gpu = False

        self.optimizer = getattr(optim,optimizer)(self.model.parameters(),**optimizer_param)

        if self.using_multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)
        self.checkpoint_file = checkpoint_path + checkpoint_name
        self.time_limit = time_limit
        self.epochs = epochs
        self.checkpoint_rate = checkpoint_rate
        self.patient = patient
        self.checkpoint_from_epoch = checkpoint_from_epoch

        self.train_loader = train_dataloader_class(train_dataset,batch_size=train_batch_size,shuffle=True,drop_last=True)
        self.val_loader = val_dataloader_class(val_dataset,batch_size=val_batch_size,shuffle=False)
        self.checkpoint_call_back = checkpoint_call_back

    def checkpoint(self):
        if self.using_multi_gpu:
            torch.save(self.model.module.state_dict(),self.checkpoint_file)
        else:
            torch.save(self.model.state_dict(),self.checkpoint_file)

        if self.checkpoint_call_back is not None:
            self.checkpoint_call_back()
    
    def set_epochs(self,epochs):
        self.epochs = epochs