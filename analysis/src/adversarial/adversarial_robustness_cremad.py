import sys
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "main.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root containing main.py.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from dataloader.CREMAD_loader import CREMAD_Dataloder
from model.ResNet import *
from copy import deepcopy
from utils.config import Config
from utils.function_tools import *
from utils.logger import *
from utils.checkpoint_tools import load_state_dict_from_manifest
from analysis.scripts.common.analysis_runtime import load_manifest, parse_analysis_args


class AdversarialAttacker:
    """Class for generating adversarial examples"""
    def __init__(self, model, epsilon=0.1, alpha=0.001, iterations=0, multimodal=False, main_modality='vision'):
        self.model = model
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha  # Step size
        self.iterations = iterations
        self.is_multimodal = multimodal
        self.main_modality = main_modality
        
    def fgsm_attack(self, model, vision, audio, labels):
        """Fast Gradient Sign Method attack"""
        # Clone the images to avoid modifying the original
        perturbed_images = vision.clone().detach().requires_grad_(True)
        perturbed_audio = audio.clone().detach().requires_grad_(True)
        
        
        for _ in range(self.iterations):
            
            # Get model predictions
            if self.main_modality.lower() == 'vision':
                if self.is_multimodal is True:
                    logits = model(audio, perturbed_images)
                else:
                    logits = model(perturbed_images)
            else:
                if self.is_multimodal is True:
                    logits = model(perturbed_audio, vision)
                else:
                    logits = model(perturbed_audio)

            loss = F.cross_entropy(logits, labels)
            # Backward pass
            loss.backward()
                
            # Create perturbation using sign of gradient (single step)
            # with torch.no_grad():
            if self.main_modality.lower() == 'vision':
                perturbation = self.epsilon * perturbed_images.grad.sign()
                perturbed_images = torch.clamp(vision + perturbation, 0, 1)
                perturbed_images = perturbed_images.clone().detach().requires_grad_(True)
            else:
                perturbation = self.epsilon * perturbed_audio.grad.sign()
                perturbed_audio = torch.clamp(audio + perturbation, 0, 1)
                perturbed_audio = perturbed_audio.clone().detach().requires_grad_(True)
        
        if self.main_modality.lower() == 'vision':
            return perturbed_images
        else:
            return perturbed_audio
    
    def pgd_attack(self, model, vision, audio, labels):
        """Projected Gradient Descent attack"""
        perturbed_images = vision.clone().detach() + self.epsilon * torch.randn_like(vision.detach())
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        perturbed_audio = audio.clone().detach() + self.epsilon * torch.randn_like(audio.detach())
        # perturbed_audio = torch.clamp(perturbed_audio, 0, 1)
        
        for _ in range(self.iterations):
            perturbed_images.requires_grad_(True)
            perturbed_audio.requires_grad_(True)
            
            # Get model predictions
            if self.main_modality.lower() == 'vision':
                if self.is_multimodal is True:
                    logits = model(audio, perturbed_images)
                else:
                    logits = model(perturbed_images)
            else:
                if self.is_multimodal is True:
                    logits = model(perturbed_audio, vision)
                else:
                    logits = model(perturbed_audio)

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
            # Update perturbed images with gradient sign
            with torch.no_grad():
                if self.main_modality.lower() == 'vision':
                    perturbed_images.data = perturbed_images.data + self.alpha * perturbed_images.grad.sign()
                    perturbation = torch.clamp(perturbed_images.data - vision.data, -self.epsilon, self.epsilon)
                    perturbed_images.data = torch.clamp(vision.data + perturbation, 0, 1)
                else:
                    perturbed_audio.data = perturbed_audio.data + self.alpha * perturbed_audio.grad.sign()
                    perturbation = torch.clamp(perturbed_audio.data - audio.data, -self.epsilon, self.epsilon)
                    perturbed_audio.data = audio.data + perturbation #torch.clamp(audio.data + perturbation, 0, 1)

            perturbed_images.grad = None
            perturbed_audio.grad = None
            
            
        if self.main_modality.lower() == 'vision':
            return perturbed_images.detach()
        else:
            return perturbed_audio.detach()
                

# @torch.no_grad()
def adversarial_evaluation(model, testloader, device, num_samples=40, main_modality='vision', multimodal=False, attacker=None, attack_type=None):
    correct_clean = 0
    correct_adv = 0
    total_clean = 0
    total_adv = 0
    
    # Evaluate on clean examples first
    model.eval() 
    with torch.no_grad():
        for batch_idx, (spec, image, labels) in enumerate(testloader): 
            spec = spec.unsqueeze(1).float().to(device)
            image = image.float().to(device)
            labels = labels.long().to(device)

            if multimodal is True:
                clean_logits = model(spec, image)
            else:
                if main_modality.lower() == 'vision':
                    clean_logits = model(image)
                else:
                    clean_logits = model(spec)

            pred = torch.argmax(clean_logits, dim=1)
            correct_clean += (pred == labels).sum().item()
            total_clean += labels.size(0)
    
    # Evaluate on adversarial examples
    for _ in range(num_samples):
        for batch_idx, (spec, image, labels) in enumerate(testloader): 
            spec = spec.unsqueeze(1).float().to(device)
            image = image.float().to(device)
            labels = labels.long().to(device)

            # Generate adversarial examples  
            model.eval()  # Need gradients for attacks

            with torch.enable_grad():
                if attack_type == 'fgsm':
                    if main_modality.lower() == 'vision':
                        image = attacker.fgsm_attack(model, image, spec, labels)
                    else:
                        spec = attacker.fgsm_attack(model, image, spec, labels)
                elif attack_type == 'pgd':
                    if main_modality.lower() == 'vision':
                        image = attacker.pgd_attack(model, image, spec, labels)
                    else:
                        spec = attacker.pgd_attack(model, image, spec, labels)
                
            # Evaluate adversarial examples
            model.eval()
            with torch.no_grad():
                if multimodal is True:
                    logits_adv = model(spec, image)
                else:
                    if main_modality.lower() == 'vision':
                        logits_adv = model(image)
                    else:
                        logits_adv = model(spec)
                        
                pred = torch.argmax(logits_adv, dim=1)
                correct_adv += (pred == labels).sum().item()
                total_adv += labels.size(0)
    
    acc_clean = 100 * correct_clean / total_clean
    acc_adv = 100 * correct_adv / total_adv
    
    return acc_clean, acc_adv


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



def evaluate_and_store_results(model_name, model, dataloader, device, attacker, attack_type, results_dict, logger, **kwargs):

    logger.info(f"Evaluating CREMA-D {model_name} on clean and adversarial examples ({attack_type} attack)...")
    acc_clean, acc_adv = adversarial_evaluation(model, dataloader, device, attacker=attacker, attack_type=attack_type, **kwargs)
    logger.info(f"Clean accuracy: {acc_clean:.2f}%")
    logger.info(f"Adversarial accuracy ({attack_type}): {acc_adv:.2f}%")

    results_dict[model_name]['clean'].append(acc_clean)
    results_dict[model_name]['adv'].append(acc_adv)


def main():
    analysis_args = parse_analysis_args()
    manifest = load_manifest(analysis_args.manifest)

    
    logger = get_logger(f"CREMA-D", logger_dir="results/adversarial_robustness")
    
    vision_cfgs = Config()
    vision_cfgs.modality = 'Visual'

    audio_cfgs = Config()
    audio_cfgs.modality = 'Audio'

    multimodal_cfgs = Config()
    multimodal_cfgs.modality = 'Multimodal'

    multimodal_cfgs2 = Config()
    multimodal_cfgs2.modality = 'Multimodal' 


    loader = CREMAD_Dataloder(multimodal_cfgs)
    dataloader = loader.test_dataloader


    vision_model =  build_model(vision_cfgs)
    audio_model = build_model(audio_cfgs)
    multimodal_model = build_model(multimodal_cfgs)
    multimodal_model2 = build_model(multimodal_cfgs2)
    
    
    vision_results = {'clean': [], 'adv': []}
    multimodal_vision_results = {'clean': [], 'adv': []}
    multimodal2_vision_results = {'clean': [], 'adv': []}
    
    audio_results = {'clean': [], 'adv': []}
    multimodal_audio_results = {'clean': [], 'adv': []}
    multimodal2_audio_results = {'clean': [], 'adv': []}
    
    all_results = {
        'Vision Model': vision_results,
        'Multimodal Vision Model 1': multimodal_vision_results,
        'Multimodal Vision Model 2': multimodal2_vision_results,
        'Audio Model': audio_results,
        'Multimodal Audio Model 1': multimodal_audio_results,
        'Multimodal Audio Model 2': multimodal2_audio_results
    }
    
    
     # --- Run 1 ---
    logger.info("--- RUN 0 ---")
    
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
    multimodal_checkpoint2 = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="CREMAD",
        modality="Multimodal",
        variant="latefusion_v2",
        seed="13",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    
    audio_model.load_state_dict(audio_checkpoint)
    vision_model.load_state_dict(vision_checkpoint)
    multimodal_model.load_state_dict(multimodal_checkpoint)
    multimodal_model2.load_state_dict(multimodal_checkpoint2)

    audio_model = audio_model.to('cuda')
    vision_model = vision_model.to('cuda')
    multimodal_model = multimodal_model.to('cuda')
    multimodal_model2 = multimodal_model2.to('cuda')
        
    audio_attacker = AdversarialAttacker(deepcopy(audio_model),  epsilon=0.1, alpha=0.001, iterations=0, main_modality='audio', multimodal=False)
    evaluate_and_store_results('Audio Model', deepcopy(audio_model), dataloader, 'cuda', main_modality='audio', attacker=audio_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    multimodal_attacker = AdversarialAttacker(deepcopy(multimodal_model),  epsilon=0.1, alpha=0.001, iterations=0, main_modality='audio', multimodal=True)
    evaluate_and_store_results('Multimodal Audio Model 1', deepcopy(multimodal_model), dataloader, 'cuda', main_modality='audio', multimodal=True, attacker=multimodal_attacker, attack_type='pgd', results_dict=all_results, logger=logger)

    multimodal_attacker2 = AdversarialAttacker(deepcopy(multimodal_model2),  epsilon=0.1, alpha=0.001, iterations=0, main_modality='audio', multimodal=True)
    evaluate_and_store_results('Multimodal Audio Model 2', deepcopy(multimodal_model2), dataloader, 'cuda', main_modality='audio', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)

    vision_attacker = AdversarialAttacker(deepcopy(vision_model), epsilon=0.1, alpha=0.001, iterations=0, main_modality='vision', multimodal=False)
    evaluate_and_store_results('Vision Model', deepcopy(vision_model), dataloader, 'cuda', main_modality='vision', attacker=vision_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    multimodal_attacker = AdversarialAttacker(deepcopy(multimodal_model), epsilon=0.1, alpha=0.001, iterations=0, main_modality='vision', multimodal=True)
    evaluate_and_store_results('Multimodal Vision Model 1', deepcopy(multimodal_model), dataloader, 'cuda', main_modality='vision', multimodal=True, attacker=multimodal_attacker, attack_type='pgd',results_dict=all_results, logger=logger)

    multimodal_attacker2 = AdversarialAttacker(deepcopy(multimodal_model2), epsilon=0.1, alpha=0.001,  iterations=0, main_modality='vision', multimodal=True)
    evaluate_and_store_results('Multimodal Vision Model 2', deepcopy(multimodal_model2), dataloader, 'cuda', main_modality='vision', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)

    # --- Run 1 ---
    logger.info("--- RUN 1 ---")
    
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
    multimodal_checkpoint2 = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="CREMAD",
        modality="Multimodal",
        variant="latefusion_v2",
        seed="13",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )

    audio_model.load_state_dict(audio_checkpoint)
    vision_model.load_state_dict(vision_checkpoint)
    multimodal_model.load_state_dict(multimodal_checkpoint)
    multimodal_model2.load_state_dict(multimodal_checkpoint2)

    audio_model = audio_model.to('cuda')
    vision_model = vision_model.to('cuda')
    multimodal_model = multimodal_model.to('cuda')
    multimodal_model2 = multimodal_model2.to('cuda')
    
    audio_attacker = AdversarialAttacker(deepcopy(audio_model),  epsilon=0.1, alpha=0.001, iterations=0, main_modality='audio', multimodal=False)
    evaluate_and_store_results('Audio Model', deepcopy(audio_model), dataloader, 'cuda', main_modality='audio', attacker=audio_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    multimodal_attacker = AdversarialAttacker(deepcopy(multimodal_model),  epsilon=0.1, alpha=0.001, iterations=0, main_modality='audio', multimodal=True)
    evaluate_and_store_results('Multimodal Audio Model 1', deepcopy(multimodal_model), dataloader, 'cuda', main_modality='audio', multimodal=True, attacker=multimodal_attacker, attack_type='pgd', results_dict=all_results, logger=logger)

    multimodal_attacker2 = AdversarialAttacker(deepcopy(multimodal_model2),  epsilon=0.1, alpha=0.001, iterations=0, main_modality='audio', multimodal=True)
    evaluate_and_store_results('Multimodal Audio Model 2', deepcopy(multimodal_model2), dataloader, 'cuda', main_modality='audio', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)

    vision_attacker = AdversarialAttacker(deepcopy(vision_model), epsilon=0.1, alpha=0.001, iterations=0, main_modality='vision', multimodal=False)
    evaluate_and_store_results('Vision Model', deepcopy(vision_model), dataloader, 'cuda', main_modality='vision', attacker=vision_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    multimodal_attacker = AdversarialAttacker(deepcopy(multimodal_model), epsilon=0.1, alpha=0.001, iterations=0, main_modality='vision', multimodal=True)
    evaluate_and_store_results('Multimodal Vision Model 1', deepcopy(multimodal_model), dataloader, 'cuda', main_modality='vision', multimodal=True, attacker=multimodal_attacker, attack_type='pgd',results_dict=all_results, logger=logger)

    multimodal_attacker2 = AdversarialAttacker(deepcopy(multimodal_model2), epsilon=0.1, alpha=0.001,  iterations=0, main_modality='vision', multimodal=True)
    evaluate_and_store_results('Multimodal Vision Model 2', deepcopy(multimodal_model2), dataloader, 'cuda', main_modality='vision', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)

    logger.info("--- RUN 2 ---")
    
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
    multimodal_checkpoint2 = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="CREMAD",
        modality="Multimodal",
        variant="latefusion_v2",
        seed="13",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )

    audio_model.load_state_dict(audio_checkpoint)
    vision_model.load_state_dict(vision_checkpoint)
    multimodal_model.load_state_dict(multimodal_checkpoint)
    multimodal_model2.load_state_dict(multimodal_checkpoint2)

    audio_model = audio_model.to('cuda')
    vision_model = vision_model.to('cuda')
    multimodal_model = multimodal_model.to('cuda')
    multimodal_model2 = multimodal_model2.to('cuda')

    
    audio_attacker = AdversarialAttacker(deepcopy(audio_model),  epsilon=0.1, alpha=0.001, iterations=0, main_modality='audio', multimodal=False)
    evaluate_and_store_results('Audio Model', deepcopy(audio_model), dataloader, 'cuda', main_modality='audio', attacker=audio_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    multimodal_attacker = AdversarialAttacker(deepcopy(multimodal_model),  epsilon=0.1, alpha=0.001, iterations=0, main_modality='audio', multimodal=True)
    evaluate_and_store_results('Multimodal Audio Model 1', deepcopy(multimodal_model), dataloader, 'cuda', main_modality='audio', multimodal=True, attacker=multimodal_attacker, attack_type='pgd', results_dict=all_results, logger=logger)

    multimodal_attacker2 = AdversarialAttacker(deepcopy(multimodal_model2),  epsilon=0.1, alpha=0.001, iterations=0, main_modality='audio', multimodal=True)
    evaluate_and_store_results('Multimodal Audio Model 2', deepcopy(multimodal_model2), dataloader, 'cuda', main_modality='audio', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)

    vision_attacker = AdversarialAttacker(deepcopy(vision_model), epsilon=0.1, alpha=0.001, iterations=0, main_modality='vision', multimodal=False)
    evaluate_and_store_results('Vision Model', deepcopy(vision_model), dataloader, 'cuda', main_modality='vision', attacker=vision_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    multimodal_attacker = AdversarialAttacker(deepcopy(multimodal_model), epsilon=0.1, alpha=0.001, iterations=0, main_modality='vision', multimodal=True)
    evaluate_and_store_results('Multimodal Vision Model 1', deepcopy(multimodal_model), dataloader, 'cuda', main_modality='vision', multimodal=True, attacker=multimodal_attacker, attack_type='pgd',results_dict=all_results, logger=logger)

    multimodal_attacker2 = AdversarialAttacker(deepcopy(multimodal_model2), epsilon=0.1, alpha=0.001,  iterations=0, main_modality='vision', multimodal=True)
    evaluate_and_store_results('Multimodal Vision Model 2', deepcopy(multimodal_model2), dataloader, 'cuda', main_modality='vision', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)


    # --- Calculate and print averages and standard deviations ---
    logger.info("="*50)
    logger.info("--- Final Accuracy Statistics ---")
    logger.info("="*50)


    for model_name, results in all_results.items():
        clean_accs = np.array(results['clean'])
        adv_accs = np.array(results['adv'])

        mean_clean = np.mean(clean_accs)
        std_clean = np.std(clean_accs)

        mean_adv = np.mean(adv_accs)
        std_adv = np.std(adv_accs)

        logger.info(f"\n{model_name}:")
        logger.info(f"  Clean Accuracy: {mean_clean:.2f}% (Std: {std_clean:.2f}%)")
        logger.info(f"  Adversarial Accuracy (FGSM): {mean_adv:.2f}% (Std: {std_adv:.2f}%)")


if __name__ == "__main__":
    main()