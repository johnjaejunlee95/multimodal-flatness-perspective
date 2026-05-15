from torch.optim import SGD, AdamW, lr_scheduler, Adam

from dataloader.KineticSound_loader import KineticSoundLoader
from model.ResNet import Audio_ResNet, Visual_ResNet, LateFusion_ResNet

class KineticSound_Task(object):
    def __init__(self, cfgs):
        super(KineticSound_Task, self).__init__()
        self.cfgs = cfgs
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.load_dataloader()
        self.model = self.load_model()
        self.optimizer, self.scheduler = self.build_optimizer()
        
    def load_dataloader(self):
        loader = KineticSoundLoader(self.cfgs)
        train_dataloader = loader.train_dataloader
        val_dataloader = loader.val_dataloader
        test_dataloader = loader.test_dataloader
        return train_dataloader, val_dataloader, test_dataloader
    
    def load_model(self):
        self.cfgs.num_classes = 31
        if self.cfgs.modality == "Audio":
            net = Audio_ResNet(self.cfgs, num_classes=self.cfgs.num_classes)
        elif self.cfgs.modality == "Visual":
            net = Visual_ResNet(self.cfgs, modality="visual", num_classes=self.cfgs.num_classes)
        elif self.cfgs.modality in ("Multimodal", "Unimodal"):
            net = LateFusion_ResNet(self.cfgs, modality2="visual", num_classes=self.cfgs.num_classes)
        else:
            raise NotImplementedError(f"Unknown modality: {self.cfgs.modality}")
        return net
    
    def build_optimizer(self):
        opt_params = {"lr": self.cfgs.learning_rate, "weight_decay": self.cfgs.weight_decay}
        if self.cfgs.optim == "sgd":
            base_optimizer = SGD
            opt_params.update({"momentum": 0.9})
        elif self.cfgs.optim == "adamw":
            base_optimizer = AdamW
        elif self.cfgs.optim == "adam":
            base_optimizer = Adam
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.cfgs.optim}")

        optimizer = base_optimizer(self.model.parameters(), **opt_params)
        sched_opt = optimizer

        if self.cfgs.lr_scheduler == "lrstep":
            scheduler = lr_scheduler.StepLR(sched_opt, step_size=self.cfgs.lr_decay_step, gamma=self.cfgs.lr_decay_ratio)
        elif self.cfgs.lr_scheduler == "cosinestep":
            scheduler = lr_scheduler.CosineAnnealingLR(sched_opt, eta_min=1e-6, last_epoch=-1)
        elif self.cfgs.lr_scheduler == "cosinestepwarmup":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(sched_opt, eta_min=1e-6, last_epoch=-1)
        else:
            scheduler = None

        return optimizer, scheduler
