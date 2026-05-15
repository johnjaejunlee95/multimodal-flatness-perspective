import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_blocks import resnet18

class LateFusion_ResNet(nn.Module):
    def __init__(self, cfgs, modality1='audio', modality2='image', num_classes=10):
        super().__init__()
        self.cfgs = cfgs
        self.mode = "classify"

        self.audio_encoder = resnet18(modality=modality1)
        self.visual_encoder = resnet18(modality=modality2)
        self.head = nn.Linear(512*int(1 + cfgs.used_frames), num_classes)

    def forward(self, audio, image, pad_audio=False, pad_visual=False):
        if pad_audio:
            audio = torch.zeros_like(audio, dtype=audio.dtype, device=audio.device)
        if pad_visual:
            image = torch.zeros_like(image, dtype=image.dtype, device=image.device)

        audio_feature = self.audio_encoder(audio)
        visual_feature = self.visual_encoder(image)
        
        (C1, C, H, W) = visual_feature.size()
        B = audio_feature.size()[0]
        visual_feature = visual_feature.view(B, -1, C, H, W)
        visual_feature = visual_feature.permute(0, 2, 1, 3, 4)

        audio_outputs = F.adaptive_avg_pool2d(audio_feature, 1)
        visual_outputs = F.adaptive_avg_pool2d(visual_feature, 1)

        audio_outputs = torch.flatten(audio_outputs, start_dim=1)#.norm(dim=1)
        visual_outputs = torch.flatten(visual_outputs, start_dim=1)#.norm(dim=1)

        encoded_outputs = torch.cat((audio_outputs, visual_outputs), dim=1)
        
        out = self.head(encoded_outputs)

        if self.mode == "feature":
            return out, encoded_outputs
        return out


class Audio_ResNet(nn.Module):
    def __init__(self, cfgs, num_classes=10):
        super().__init__()
        self.cfgs = cfgs

        self.audio_encoder = resnet18(modality="audio")
        self.head = nn.Linear(512, num_classes)

    def forward(self, audio):
        audio_encoded_outputs = self.audio_encoder(audio)
        audio_outputs = F.adaptive_avg_pool2d(audio_encoded_outputs, 1)

        audio_outputs = torch.flatten(audio_outputs, start_dim=1)
        outputs = self.head(audio_outputs)
        return outputs


class Visual_ResNet(nn.Module):
    def __init__(self, cfgs, modality='image', num_classes=10):
        super().__init__()
        self.cfgs = cfgs

        self.visual_encoder = resnet18(modality=modality)
        self.head = nn.Linear(512 * cfgs.used_frames, num_classes)
        

    def forward(self, visual):
        B = visual.size(0)
        visual_encoded_outputs = self.visual_encoder(visual)
        T, C, H, W = visual_encoded_outputs.size()

        visual_encoded_outputs = visual_encoded_outputs.view(B, -1, C, H, W)
        visual_encoded_outputs = visual_encoded_outputs.permute(0, 2, 1, 3, 4)

        visual_outputs = F.adaptive_avg_pool2d(visual_encoded_outputs, 1)
        visual_outputs = torch.flatten(visual_outputs, 1)

        return self.head(visual_outputs)
