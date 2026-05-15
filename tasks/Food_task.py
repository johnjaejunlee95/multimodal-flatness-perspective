from torch.optim import SGD, AdamW, lr_scheduler, Adam
from dataloader.Food_loader import Food101DataLoader
from model.FoodNet import TextEncoder, VisionEncoder, LateFusionClassifier
from transformers import get_cosine_schedule_with_warmup

class Food_Task(object):
    def __init__(self, cfgs):
        super(Food_Task, self).__init__()
        self.cfgs = cfgs
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.load_dataloader()
        self.model = self.load_model()
        self.optimizer, self.scheduler = self.build_optimizer()
        
    def load_dataloader(self):
        loader = Food101DataLoader(self.cfgs)
        train_dataloader = loader.train_dataloader
        val_dataloader = loader.val_dataloader
        test_dataloader = loader.test_dataloader
        return train_dataloader, val_dataloader, test_dataloader
    
    def load_model(self):
        if self.cfgs.modality == "Text":
            net = TextEncoder(num_classes=self.cfgs.num_classes, feature_only=False)
        elif self.cfgs.modality == "Visual":
            net = VisionEncoder(model_arch="resnet18", num_classes=self.cfgs.num_classes, feature_only=False)
        elif self.cfgs.modality in ("Multimodal", "Unimodal"):
            net = LateFusionClassifier(self.cfgs, num_classes=self.cfgs.num_classes)
        else:
            raise NotImplementedError(f"Unknown modality: {self.cfgs.modality}")
        return net
    
    def build_optimizer(self):
        train_steps = len(self.train_dataloader) * self.cfgs.epochs
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

        if self.cfgs.modality == "Multimodal":
            vision_params = []
            text_params = []
            fusion_params = []
            for n, p in self.model.named_parameters():
                if 'vision_encoder' in n:
                    vision_params.append(p)
                elif 'text_encoder' in n:
                    text_params.append(p)
                elif 'classifier' in n:
                    fusion_params.append(p)

            param_list = [
                {"params": vision_params, "lr": self.cfgs.learning_rate, "weight_decay": self.cfgs.weight_decay * 0.1},
                {"params": text_params, "lr": self.cfgs.learning_rate * 0.1, "weight_decay": self.cfgs.weight_decay},
                {"params": fusion_params, "lr": self.cfgs.learning_rate, "weight_decay": self.cfgs.weight_decay},
            ]

            optimizer = base_optimizer(param_list, **opt_params)
        else:
            optimizer = base_optimizer(self.model.parameters(), **opt_params)
        sched_opt = optimizer

        if self.cfgs.lr_scheduler == "lrstep":
            scheduler = lr_scheduler.StepLR(sched_opt, step_size=self.cfgs.lr_decay_step, gamma=self.cfgs.lr_decay_ratio)
        elif self.cfgs.lr_scheduler == "cosinestep":
            scheduler = lr_scheduler.CosineAnnealingLR(sched_opt, eta_min=1e-6, last_epoch=-1)
        elif self.cfgs.lr_scheduler == "cosinestepwarmup":
            scheduler = get_cosine_schedule_with_warmup(
                sched_opt,
                num_warmup_steps=int(train_steps * 0.05),
                num_training_steps=train_steps,
            )
        else:
            scheduler = None

        return optimizer, scheduler
