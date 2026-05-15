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
from dataloader.Food_loader import Food101DataLoader
from model.FoodNet import *
from utils.checkpoint_tools import load_state_dict_from_manifest
from analysis.scripts.common.analysis_runtime import load_manifest, parse_analysis_args
from transformers import BertTokenizer
from torch.utils.data import Subset

@torch.no_grad()
def test(model, testloader, device, main_modality=None, multimodal=False):
    model.eval()
    test_loss = 0
    for image, text, label in testloader:
        image = image.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)
        text_token = tokenize_batch(text).to(device)
        # text_token = {k: v.to(device, non_blocking=True) for k, v in text_token.items()}

        if multimodal is True:
            logits = model(image, text_token)
        else:
            if main_modality.lower() == 'visual':
                logits = model(image)
            else:
                logits = model(text_token)
                
        loss = F.cross_entropy(logits, label)
        test_loss += loss.item()

        
    return test_loss / len(testloader)


tokenizer_global = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    cache_dir="/nfs2/jjlee/model_cache",
    add_prefix_space=False
)
def tokenize_batch(texts):
    tokens = tokenizer_global(
        texts,
        padding='max_length',
        max_length=40,
        truncation=True,
        return_tensors='pt'
    )
    return tokens

def build_model(cfgs):
    if cfgs.modality == 'Text':
        net = TextEncoder(num_classes = cfgs.num_classes, feature_only=False)
    elif cfgs.modality == 'Visual':
        net = VisionEncoder(model_arch='resnet18', num_classes=cfgs.num_classes, feature_only=False)
    elif cfgs.modality == 'Multimodal' or cfgs.modality == 'Unimodal':
        net = LateFusionClassifier(cfgs, num_classes=cfgs.num_classes)
    return net


def main():
    analysis_args = parse_analysis_args()
    manifest = load_manifest(analysis_args.manifest)

    
    logger = get_logger(f"Food101_LPF", logger_dir="results/LPF")
    
    vision_cfgs = Config()
    vision_cfgs.modality = 'Visual'

    text_cfgs = Config()
    text_cfgs.modality = 'Text'

    multimodal_cfgs = Config()
    multimodal_cfgs.modality = 'Multimodal'

    multimodal_cfgs2 = Config()
    multimodal_cfgs2.modality = 'Multimodal' 
    
        
    loader = Food101DataLoader(multimodal_cfgs)
    dataloader = loader.test_dataloader
    # half_len = len(test_dataset) // 2
    # indices = list(range(half_len))
    # dataloader = torch.utils.data.DataLoader(
    #     Subset(test_dataset, indices),
    #     batch_size=loader.test_dataloader.batch_size,
    #     shuffle=False,
    #     num_workers=loader.test_dataloader.num_workers,
    #     pin_memory=True
    # )
    
    text_model = build_model(text_cfgs)
    vision_model = build_model(vision_cfgs)
    multimodal_model = build_model(multimodal_cfgs)
    multimodal_model2 = build_model(multimodal_cfgs2)

    
    text_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Text",
        variant="standard",
        seed="135",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    vision_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Visual",
        variant="standard",
        seed="135",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    multimodal_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Multimodal",
        variant="sml",
        seed="135",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    multimodal_checkpoint2 = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Multimodal",
        variant="dml",
        seed="135",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    
    
    text_model.load_state_dict(text_checkpoint)
    vision_model.load_state_dict(vision_checkpoint)
    multimodal_model.load_state_dict(multimodal_checkpoint)
    multimodal_model2.load_state_dict(multimodal_checkpoint2)

    
    text_model = text_model.to('cuda')
    vision_model = vision_model.to('cuda')
    multimodal_model = multimodal_model.to('cuda')
    multimodal_model2 = multimodal_model2.to('cuda')
    
        
    mcmc_iter = 10
    sigma = 0.007
    device = 'cuda'

    with torch.no_grad():
        text_model_params = [p.data.clone() for p in text_model.parameters()]
        vision_model_params = [p.data.clone() for p in vision_model.parameters()]
        multimodal_model_params = [p.data.clone() for p in multimodal_model.parameters()]
        multimodal_model_params2 = [p.data.clone() for p in multimodal_model2.parameters()]

    vision_loss, text_loss, multimodal_loss, multimodal_loss_ver2 = 0.0, 0.0, 0.0, 0.0
    
    for iter in range(mcmc_iter):
        
        for mp1, p1 in zip(text_model.parameters(), text_model_params):
            mp1.data.copy_(p1 + torch.randn_like(p1, device=mp1.device).mul_(sigma*2))
        for mp2, p2 in zip(vision_model.parameters(), vision_model_params):
            mp2.data.copy_(p2 + torch.randn_like(p2, device=mp2.device).mul_(sigma))
        for mp3, p3 in zip(multimodal_model.parameters(), multimodal_model_params):
            mp3.data.copy_(p3 + torch.randn_like(p3, device=mp3.device).mul_(sigma))
        for mp4, p4 in zip(multimodal_model2.parameters(), multimodal_model_params2):
            mp4.data.copy_(p4 + torch.randn_like(p4, device=mp4.device).mul_(sigma))
      
            
        vision_loss += test(vision_model, dataloader, device, 'Visual')
        text_loss += test(text_model, dataloader, device, 'Text')
        multimodal_loss += test(multimodal_model, dataloader, device,  multimodal=True)
        multimodal_loss_ver2 += test(multimodal_model2, dataloader, device, multimodal=True)
        
    
    vision_loss /= mcmc_iter
    text_loss /= mcmc_iter
    multimodal_loss /= mcmc_iter
    multimodal_loss_ver2 /= mcmc_iter
   
    

    load_weights(text_model, text_model_params)
    load_weights(vision_model, vision_model_params)
    load_weights(multimodal_model, multimodal_model_params)
    load_weights(multimodal_model2, multimodal_model_params2)
    
    
    original_vision_loss = test(vision_model, dataloader, device, 'Visual')
    original_text_loss = test(text_model, dataloader, device, 'Text')
    original_multimodal_loss = test(multimodal_model, dataloader, device,  multimodal=True)
    original_multimodal_loss2 = test(multimodal_model2, dataloader, device, multimodal=True)
    
    
    logger.info(f'Vision LPF Loss: {vision_loss:.4f}, Original Vision Loss: {original_vision_loss:.4f}')
    logger.info(f'Audio LPF Loss: {text_loss:.4f}, Original Audio Loss: {original_text_loss:.4f}')
    logger.info(f'Multimodal LPF Loss: {multimodal_loss:.4f}, Original Multimodal Loss: {original_multimodal_loss:.4f}')
    logger.info(f'Multimodal Shuffling LPF Loss: {multimodal_loss_ver2:.4f}, Original Multimodal Shuffling Loss: {original_multimodal_loss2:.4f}')
    
    logger.info(f"Vision Discrepency: {vision_loss - original_vision_loss:.4f}")
    logger.info(f"Audio Discrepency: {text_loss - original_text_loss:.4f}")
    logger.info(f"Multimodal Discrepency: {multimodal_loss - original_multimodal_loss:.4f}")
    logger.info(f"Multimodal Shuffling Discrepency: {multimodal_loss_ver2 - original_multimodal_loss2:.4f}")
   

if __name__ == '__main__':
        
    main()
