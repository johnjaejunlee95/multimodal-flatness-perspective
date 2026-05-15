import torch

from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F
from torch import nn
import torchvision
import timm

class TextEncoder(nn.Module):
    def __init__(self, output_dim=512, num_classes=101, feature_only=True):
        super().__init__()
        config = BertConfig()
        self.textEncoder= BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, cache_dir="/nfs2/jjlee/model_cache")  
        
        self.layernorm = nn.LayerNorm(config.hidden_size)  
        self.dropout = nn.Dropout(0.5)
        
        self.linear = nn.Linear(config.hidden_size, output_dim)
        
        self.classifier = nn.Identity()
        if not feature_only:
            self.classifier = nn.Linear(output_dim, num_classes)
        
    def forward(self, tokens):
        hidden_states = self.textEncoder(**tokens)["last_hidden_state"][:,  0]
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        text_embedding = self.linear(hidden_states) 
        output = self.classifier(text_embedding)
        return output

class VisionEncoder(nn.Module):
    def __init__(self, model_arch, num_classes=101, weights="IMAGENET1K_V1", feature_only=True):
        super().__init__()
        self.model_arch=model_arch
        self.num_classes = num_classes

        self.model = torchvision.models.resnet18(weights=weights)
        if feature_only:
            if self.model_arch.startswith('resnet'):
                self.model.fc = nn.Identity()
            elif self.model_arch.startswith('vit'):
                self.model.head = nn.Identity() #nn.Linear(in_features, in_features // 2)
        else:
            if self.model_arch.startswith('resnet'):
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, self.num_classes)
            elif self.model_arch.startswith('vit'):
                in_features = self.model.head.in_features
                self.model.head = nn.Linear(in_features, self.num_classes)
            else:
                raise NotImplementedError(f"Model architecture {self.model_arch} not supported for classifier modification.")

    def forward(self,x):
        y = self.model(x)
        return y
    


class LateFusionClassifier(nn.Module):
    def __init__(self, cfgs, num_classes, is_feature=False):
        super(LateFusionClassifier, self).__init__()
        self.cfgs = cfgs
        self.is_feature = is_feature

        self.text_encoder = TextEncoder(output_dim=512)
        self.vision_encoder = VisionEncoder(model_arch='resnet18')
        
        self.classifier = nn.Linear(512*2, num_classes)
      

    def forward(self, vision, text):
        text_feature = self.text_encoder(text)       # [B, 512]
        vision_feature = self.vision_encoder(vision) # [B, 512]
        
        encoded_feature = torch.cat((text_feature, vision_feature), dim=1)  # [B, 1024]

        out = self.classifier(encoded_feature)

        if self.is_feature:
            return out, encoded_feature
        return out