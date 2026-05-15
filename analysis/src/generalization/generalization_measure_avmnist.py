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
from utils.logger import *
from utils.config import Config
from dataloader.AVMNIST_loader import AVMNISTLoader
from model.ResNet import *
from utils.checkpoint_tools import load_state_dict_from_manifest
from analysis.scripts.common.analysis_runtime import load_manifest, parse_analysis_args
@torch.no_grad()
def test(model, testloader, device, main_modality, multimodal=False):
    model.eval()
    test_loss = 0
    for batch_idx, (audio, vision, labels) in enumerate(testloader): 
        audio = audio.float().to(device)
        vision = vision.float().to(device)
        labels = labels.long().to(device)

        if multimodal is True:
            logits = model(audio, vision)
            
        else:
            if main_modality.lower() == 'vision':
                logits = model(vision)
            else:
                logits = model(audio)

        loss = F.cross_entropy(logits, labels)
        test_loss += loss.item()
        
    return test_loss / len(testloader)


def build_model(cfgs):
    if cfgs.fusion_type == 'late_fusion':
        if cfgs.modality == 'Audio':
            net = Audio_ResNet(cfgs)
        elif cfgs.modality == 'Visual':
            net = Visual_ResNet(cfgs)
        elif cfgs.modality == 'Multimodal':
            if cfgs.methods =='OGM_GE':
                net = GradMod(cfgs)
            else:
                net = LateFusion_ResNet(cfgs)
        else:
            raise NotImplementedError
    elif cfgs.fusion_type == 'early_fusion':
        net = GradMod(cfgs)
    return net


def main():
    analysis_args = parse_analysis_args()
    manifest = load_manifest(analysis_args.manifest)

    
    logger = get_logger(f"AVMNIST_LPF", logger_dir="results/LPF")
    
    vision_cfgs = Config()
    vision_cfgs.modality = 'Visual'

    audio_cfgs = Config()
    audio_cfgs.modality = 'Audio'

    multimodal_cfgs = Config()
    multimodal_cfgs.modality = 'Multimodal'

    multimodal_cfgs2 = Config()
    multimodal_cfgs2.modality = 'Multimodal' 
        
    loader = AVMNISTLoader(multimodal_cfgs)
    dataloader = loader.test_dataloader
    
    audio_model = build_model(audio_cfgs)
    vision_model = build_model(vision_cfgs)
    multimodal_model = build_model(multimodal_cfgs)
    multimodal_model2 = build_model(multimodal_cfgs2)

    
   
    multimodal_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="AVMNIST",
        modality="Multimodal",
        variant="sam",
        seed="456",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    mutlimodal_checkpoint2 = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="AVMNIST",
        modality="Multimodal",
        variant="dml_sam",
        seed="456",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    
    
    multimodal_model.load_state_dict(multimodal_checkpoint)
    multimodal_model2.load_state_dict(mutlimodal_checkpoint2)

    
   
    multimodal_model = multimodal_model.to('cuda')
    multimodal_model2 = multimodal_model2.to('cuda')
    
        
    mcmc_iter = 50
    sigma = 0.004
    device = 'cuda'

    with torch.no_grad():
       
        multimodal_model_params = [p.data.clone() for p in multimodal_model.parameters()]
        multimodal_model_params2 = [p.data.clone() for p in multimodal_model2.parameters()]

    vision_loss, audio_loss, multimodal_loss, multimodal_loss_ver2 = 0.0, 0.0, 0.0, 0.0
    
    for iter in range(mcmc_iter):
        
        
        for mp3, p3 in zip(multimodal_model.parameters(), multimodal_model_params):
            mp3.data.copy_(p3 + torch.randn_like(p3, device=mp3.device).mul_(sigma))
        for mp4, p4 in zip(multimodal_model2.parameters(), multimodal_model_params2):
            mp4.data.copy_(p4 + torch.randn_like(p4, device=mp4.device).mul_(sigma))
      
            
       
        multimodal_loss += test(multimodal_model, dataloader, device, 'Vision', multimodal=True)
        multimodal_loss_ver2 += test(multimodal_model2, dataloader, device, 'Vision', multimodal=True)
        

    multimodal_loss /= mcmc_iter
    multimodal_loss_ver2 /= mcmc_iter
   
    

   
    load_weights(multimodal_model, multimodal_model_params)
    load_weights(multimodal_model2, multimodal_model_params2)
    
    
   
    original_multimodal_loss = test(multimodal_model, dataloader, device, 'Multimodal', multimodal=True)
    original_multimodal_loss2 = test(multimodal_model2, dataloader, device, 'Multimodal', multimodal=True)
    
    
   
    logger.info(f'Multimodal LPF Loss: {multimodal_loss:.4f}, Original Multimodal Loss: {original_multimodal_loss:.4f}')
    logger.info(f'Multimodal Shuffling LPF Loss: {multimodal_loss_ver2:.4f}, Original Multimodal Shuffling Loss: {original_multimodal_loss2:.4f}')
    
   
    logger.info(f"Multimodal Discrepency: {multimodal_loss - original_multimodal_loss:.4f}")
    logger.info(f"Multimodal Shuffling Discrepency: {multimodal_loss_ver2 - original_multimodal_loss2:.4f}")
   

if __name__ == '__main__':
        
    main()
