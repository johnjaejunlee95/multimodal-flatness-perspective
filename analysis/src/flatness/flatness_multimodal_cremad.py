import sys
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "main.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root containing main.py.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataloader.CREMAD_loader import CREMAD_Dataloder
from model.ResNet import *
from copy import deepcopy
import numpy as np
import torch
from utils.config import Config
import torch.nn.functional as F
import os
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

@torch.no_grad()
def add_perturbation(model, random_vector):
    random_vector = random_vector.split([p.numel() for p in model.parameters()])
    for p, rv in zip(model.parameters(), random_vector):
        p.add_(rv.view_as(p))

def mc_flatness(model, testloader, device, radius_list, MC_num=5, main_modality='vision', multimodal=False ):
    print(f"Computing {model.cfgs.methods} {main_modality.lower()} flatness with {MC_num} Monte Carlo iterations")
    flatness_results = []
    for radius in radius_list:
        avg_loss = 0.0
        num_param = sum(p.numel() for p in model.parameters())  # Total parameters

        # tqdmbar2 = tqdm(range(MC_num), desc=f'Radius {int(radius)}', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=1, leave=False)
        for _ in range(MC_num):
            perturbed_model = deepcopy(model)

            if int(radius) == 0:
                perturbed_loss = test(perturbed_model, testloader, device, main_modality, multimodal)
                avg_loss += perturbed_loss*MC_num
                break
            # Generate normalized random perturbation
            num_param = sum(p.numel() for n, p in perturbed_model.named_parameters()) # if ("classifier" not in n) or ('classification' not in n)
            random_param = torch.randn(num_param)
            random_unit_param = random_param / torch.norm(random_param)
            random_unit_param = random_unit_param / torch.norm(random_unit_param)
            rand_vector = random_unit_param.to(device)

            rand_vector = rand_vector*radius
            # Apply perturbation
            add_perturbation(perturbed_model, rand_vector)
            perturbed_loss = test(perturbed_model, testloader, device, main_modality, multimodal)
            
            avg_loss += perturbed_loss

        avg_loss /= MC_num 

        flatness_results.append(avg_loss)

    return flatness_results



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

    vision_cfgs = Config()
    vision_cfgs.modality = 'Visual'

    audio_cfgs = Config()
    audio_cfgs.modality = 'Audio'

    multimodal_cfgs = Config()
    multimodal_cfgs.modality = 'Multimodal'

    multimodal_cfgs2 = Config()
    multimodal_cfgs2.modality = 'Multimodal' 
        
    loader = CREMAD_Dataloder(multimodal_cfgs)
    mode= multimodal_cfgs.mode
    if mode == 'train':
        dataloader = loader.train_dataloader
    else:
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
    
        
    radius_list = np.arange(0, 31, 3)
    monte_carlo_iter = 150
    device = 'cuda'
    
    
    main_modality = 'Multimodal'
    audio_flatness = mc_flatness(audio_model, dataloader, device, radius_list, monte_carlo_iter, "Audio")  
    vision_flatness = mc_flatness(vision_model, dataloader, device, radius_list, monte_carlo_iter, "Vision")
    multimodal_flatness = mc_flatness(multimodal_model, dataloader, device, radius_list, monte_carlo_iter, main_modality = main_modality, multimodal=True)
    multimodal_flatness2 = mc_flatness(multimodal_model2, dataloader, device, radius_list, monte_carlo_iter, main_modality = main_modality, multimodal=True)
    
    audio_flatness = np.array(audio_flatness)
    vision_flatness = np.array(vision_flatness)
    multimodal_flatness = np.array(multimodal_flatness)
    multimodal_flatness2 = np.array(multimodal_flatness2)

    if not os.path.exists("results/flatness/CREMAD"):
        os.makedirs("results/flatness/CREMAD", exist_ok=True)
    
    torch.save(audio_flatness, f"results/flatness/CREMAD/#audio_flatness_{mode}.pth")
    torch.save(vision_flatness, f"results/flatness/CREMAD/#vision_flatness_{mode}.pth")
    torch.save(multimodal_flatness, f"results/flatness/CREMAD/#multimodal_flatness_{mode}.pth")
    torch.save(multimodal_flatness2, f"results/flatness/CREMAD/#multimodal_flatness2_{mode}.pth")
    

if __name__ == "__main__":
    main()
