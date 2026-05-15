import sys
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "main.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root containing main.py.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from utils.function_tools import * 
from utils.config import Config
from utils.logger import *
from dataloader.CREMAD_loader import CREMAD_Dataloder
from model.ResNet import *
from utils.checkpoint_tools import load_state_dict_from_manifest
from analysis.scripts.common.analysis_runtime import load_manifest, parse_analysis_args



@torch.no_grad()
def test(model, testloader, device, main_modality, multimodal=False):
    model.eval()
    test_loss = 0
    for batch_idx, (spec, image, labels) in enumerate(testloader): 
        spec = spec.float().to(device)
        image = image.float().to(device)
        labels = labels.long().to(device)

        if multimodal is True:
            logits = model(spec.unsqueeze(1).float(), image.float())
        else: 
            if main_modality.lower() == 'vision':
                logits = model(image.float())
            else:
                logits = model(spec.unsqueeze(1).float())
                
        loss = F.cross_entropy(logits, labels)

        test_loss += loss.item()
        
    return test_loss / len(testloader)

def build_model(cfgs):
    if cfgs.modality == "Audio":
        net = Audio_ResNet(cfgs, num_classes=cfgs.num_classes)
    elif cfgs.modality == "Visual":
        net = Visual_ResNet(cfgs, modality="visual", num_classes=cfgs.num_classes)
    elif cfgs.modality in ("Multimodal", "Unimodal"):
        net = LateFusion_ResNet(cfgs, modality2="visual", num_classes=cfgs.num_classes)
    else:
        raise NotImplementedError(f"Unknown modality: {cfgs.modality}")
    return net

def main():
    analysis_args = parse_analysis_args()
    manifest = load_manifest(analysis_args.manifest)
    
    logger = get_logger(f"CREMAD_LPF", logger_dir="results/LPF")
    
    vision_cfgs = Config()
    vision_cfgs.modality = 'Visual'

    audio_cfgs = Config()
    audio_cfgs.modality = 'Audio'

    multimodal_cfgs = Config()
    multimodal_cfgs.modality = 'Multimodal'

    multimodal_cfgs2 = Config()
    multimodal_cfgs2.modality = 'Multimodal' 
    
    loader = CREMAD_Dataloder(multimodal_cfgs)
    # dataloader = loader.train_dataloader
    dataloader = loader.test_dataloader
    
    
    vision_model =  build_model(vision_cfgs)
    audio_model = build_model(audio_cfgs)
    multimodal_model = build_model(multimodal_cfgs)
    multimodal_model2 = build_model(multimodal_cfgs2)
   
   
    vision_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="CREMAD",
        modality="Visual",
        variant="latefusion",
        seed="13",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    audio_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="CREMAD",
        modality="Audio",
        variant="latefusion",
        seed="13",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    multimodal_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="CREMAD",
        modality="Multimodal",
        variant="latefusion",
        seed="13",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    mutlimodal_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="CREMAD",
        modality="Multimodal",
        variant="latefusion_v2",
        seed="13",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )

    
    vision_model.load_state_dict(vision_checkpoint)
    audio_model.load_state_dict(audio_checkpoint)
    multimodal_model.load_state_dict(multimodal_checkpoint)
    multimodal_model2.load_state_dict(mutlimodal_checkpoint)
    

    vision_model = vision_model.to('cuda')
    audio_model = audio_model.to('cuda')
    multimodal_model = multimodal_model.to('cuda')
    multimodal_model2 = multimodal_model2.to('cuda')

        
    mcmc_iter = 50
    sigma = 0.005
    device = 'cuda'

    audio_model_params = [p.data.clone() for p in audio_model.parameters()]
    vision_model_params = [p.data.clone() for p in vision_model.parameters()]
    multimodal_model_params = [p.data.clone() for p in multimodal_model.parameters()]
    multimodal_model2_params = [p.data.clone() for p in multimodal_model2.parameters()]

    
    vision_loss, audio_loss, multimodal_loss, multimodal_loss_ver2 = 0.0, 0.0, 0.0, 0.0
    
    original_vision_loss = test(vision_model, dataloader, device, 'Vision')
    original_audio_loss = test(audio_model, dataloader, device, 'Audio')
    original_multimodal_loss = test(multimodal_model, dataloader, device, 'Multimodal', multimodal=True)
    original_multimodal_loss_ver2 = test(multimodal_model2, dataloader, device, 'Multimodal', multimodal=True)
    
    for iter in range(mcmc_iter):
        
        for mp1, p1 in zip(audio_model.parameters(), audio_model_params):
            mp1.data.copy_(p1 + torch.randn_like(p1, device=mp1.device).mul_(sigma))
        for mp2, p2 in zip(vision_model.parameters(), vision_model_params):
            mp2.data.copy_(p2 + torch.randn_like(p2, device=mp2.device).mul_(sigma))
        for mp3, p3 in zip(multimodal_model.parameters(), multimodal_model_params):
            mp3.data.copy_(p3 + torch.randn_like(p3, device=mp3.device).mul_(sigma))
        for mp4, p4 in zip(multimodal_model2.parameters(), multimodal_model2_params):
            mp4.data.copy_(p4 + torch.randn_like(p4, device=mp4.device).mul_(sigma))
       
            
        vision_loss += test(vision_model, dataloader, device, 'Vision')
        audio_loss += test(audio_model, dataloader, device, 'Audio')
        multimodal_loss += test(multimodal_model, dataloader, device, 'Vision', multimodal=True)
        multimodal_loss_ver2 += test(multimodal_model2, dataloader, device, 'Vision', multimodal=True)
       
       
    vision_loss /= mcmc_iter
    audio_loss /= mcmc_iter
    multimodal_loss /= mcmc_iter
    multimodal_loss_ver2 /= mcmc_iter


    load_weights(audio_model, audio_model_params)
    load_weights(vision_model, vision_model_params)
    load_weights(multimodal_model, multimodal_model_params)
    load_weights(multimodal_model2, multimodal_model2_params)
    
    
    logger.info(f"Vision LPF Loss: {vision_loss:.4f}, Original Vision Loss: {original_vision_loss:.4f}")
    logger.info(f"Audio LPF Loss: {audio_loss:.4f}, Original Audio Loss: {original_audio_loss:.4f}")
    logger.info(f"Multimodal LPF Loss: {multimodal_loss:.4f}, Original Multimodal Loss: {original_multimodal_loss:.4f}")
    logger.info(f"Multimodal Shuffling LPF Loss: {multimodal_loss_ver2:.4f}, Original Multimodal Loss: {original_multimodal_loss_ver2:.4f}")
    
    logger.info(f"Vision Discrepency: {vision_loss - original_vision_loss:.4f}")
    logger.info(f"Audio Discrepency: {audio_loss - original_audio_loss:.4f}")
    logger.info(f"Multimodal Discrepency: {multimodal_loss - original_multimodal_loss:.4f}")
    logger.info(f"Multimodal Shuffling Discrepency: {multimodal_loss_ver2 - original_multimodal_loss_ver2:.4f}")
    
    
if __name__ == '__main__':
        
    main()
