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

from dataloader.Food_loader import Food101DataLoader
from model.FoodNet import *
from utils.config import Config
from utils.function_tools import *
from utils.logger import *
from utils.checkpoint_tools import load_state_dict_from_manifest
from analysis.scripts.common.analysis_runtime import load_manifest, parse_analysis_args
from copy import deepcopy

class AdversarialAttacker:
    """Class for generating adversarial examples"""
    def __init__(self, model, epsilon=0.1, alpha=0.001, iterations=10, multimodal=False, main_modality='image'):
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
            if self.main_modality.lower() == 'image':
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
            if self.main_modality.lower() == 'image':
                perturbation = self.epsilon * perturbed_images.grad.sign()
                perturbed_images = torch.clamp(vision + perturbation, 0, 1)
                perturbed_images = perturbed_images.clone().detach().requires_grad_(True)
            else:
                perturbation = self.epsilon * perturbed_audio.grad.sign()
                perturbed_audio = torch.clamp(audio + perturbation, 0, 1)
                perturbed_audio = perturbed_audio.clone().detach().requires_grad_(True)
        
        if self.main_modality.lower() == 'image':
            return perturbed_images
        else:
            return perturbed_audio
    
    def pgd_attack(self, model, vision, audio, labels):
        """Projected Gradient Descent attack"""
        # Clone the images to avoid modifying the original
        perturbed_images = vision.clone().detach() + self.epsilon * torch.randn_like(vision.detach())
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        
        
        # perturbed_audio = audio
        # perturbed_audio = audio.clone().detach() + self.epsilon * torch.randn_like(audio.detach())
        # perturbed_audio = torch.clamp(perturbed_audio, 0, 1)
        
        for _ in range(self.iterations):
            perturbed_images.requires_grad_(True)
            # perturbed_audio.requires_grad_(True)
            
            # Get model predictions
            if self.main_modality.lower() == 'image':
                if self.is_multimodal is True:
                    logits = model(perturbed_images, audio)
                else:
                    logits = model(perturbed_images)
            else:
                if self.is_multimodal is True:
                    logits = model(vision, audio)
                else:
                    logits = model(audio)

            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Update perturbed images with gradient sign
            with torch.no_grad():
                perturbed_images.data = perturbed_images.data + self.alpha * perturbed_images.grad.sign()
                perturbation = torch.clamp(perturbed_images.data - vision.data, -self.epsilon, self.epsilon)
                perturbed_images.data = torch.clamp(vision.data + perturbation, 0, 1)

            perturbed_images.grad = None 
            
            
        return perturbed_images.detach()
       
                
    # def fgsm_multimodal_attack(self, model, vision, audio, labels):
    #     """Fast Gradient Sign Method attack for multimodal inputs"""
    #     if not self.is_multimodal:
    #         raise ValueError("FGSM attack for both modalities requires multimodal=True")

    #     # Clone and detach inputs, then enable gradient tracking for both
    #     perturbed_vision = vision.clone().detach().requires_grad_(True)
    #     perturbed_audio = audio.clone().detach().requires_grad_(True)

    #     # Single step FGSM logic:
    #     # Get model predictions
    #     logits = model(perturbed_audio, perturbed_vision)
    #     loss = F.cross_entropy(logits, labels)

    #     # Zero existing gradients before backward pass
    #     if perturbed_vision.grad is not None:
    #         perturbed_vision.grad.zero_()
    #     if perturbed_audio.grad is not None:
    #         perturbed_audio.grad.zero_()

    #     # Backward pass to get gradients for both vision and audio
    #     loss.backward()
        
    #     with torch.no_grad():
    #         perturbation_vision = self.epsilon * perturbed_vision.grad.sign()
    #         perturbation_audio = self.epsilon * perturbed_audio.grad.sign()
                  
    #         perturbed_vision = torch.clamp(vision + perturbation_vision, 0, 1)
    #         perturbed_audio = torch.clamp(audio + perturbation_audio, 0, 1)

    #     return perturbed_vision, perturbed_audio


    # def pgd_multimodal_attack(self, model, vision, audio, labels):
    #     """Projected Gradient Descent attack for multimodal inputs"""
    #     if not self.is_multimodal:
    #         raise ValueError("PGD attack for both modalities requires multimodal=True")

    #     perturbed_vision = vision.clone().detach() + self.epsilon * torch.randn_like(vision.detach())
    #     perturbed_vision = torch.clamp(perturbed_vision, 0, 1)

    #     perturbed_audio = audio.clone().detach() + self.epsilon * torch.randn_like(audio.detach()) 
    #     # perturbed_audio = torch.clamp(perturbed_audio, 0, 1)

    #     for _ in range(self.iterations):
    #         # Enable gradient tracking for both perturbed inputs
    #         perturbed_vision.requires_grad_(True)
    #         perturbed_audio.requires_grad_(True)

    #         # Get model predictions
    #         logits = model(perturbed_audio, perturbed_vision)
    #         loss = F.cross_entropy(logits, labels)

    #         # Zero existing gradients before backward pass
    #         if perturbed_vision.grad is not None:
    #             perturbed_vision.grad.zero_()
    #         if perturbed_audio.grad is not None:
    #             perturbed_audio.grad.zero_()

    #         loss.backward()

    #         with torch.no_grad():
    #             perturbed_vision.data = perturbed_vision.data + self.alpha * perturbed_vision.grad.sign()
    #             perturbation_vision = torch.clamp(perturbed_vision.data - vision.data, -self.epsilon, self.epsilon)
    #             perturbed_vision.data = torch.clamp(vision.data + perturbation_vision, 0, 1)

    #             perturbed_audio.data = perturbed_audio.data + self.alpha * perturbed_audio.grad.sign()
    #             perturbation_audio = torch.clamp(perturbed_audio.data - audio.data, -self.epsilon, self.epsilon)
    #             perturbed_audio.data = audio.data + perturbation_audio, #torch.clamp(audio.data + perturbation_audio, 0, 1)

    #     return perturbed_vision.detach(), perturbed_audio.detach()
    

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

def adversarial_evaluation(model, testloader, device, num_samples=5, main_modality='image', multimodal=False, attacker=None, attack_type=None):
    correct_clean = 0
    correct_adv = 0
    total_clean = 0
    total_adv = 0
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="/nfs2/jjlee/model_cache", add_prefix_space=False)
    # Evaluate on clean examples first
    model.eval()
    with torch.inference_mode():
        for batch_idx, (image, text, labels) in enumerate(testloader): 
            image = image.float().to(device)
            labels = labels.long().to(device)
            
            text_token = tokenizer(text, padding='max_length', max_length=40, truncation=True, return_tensors='pt').to(device)

            if multimodal is True:
                clean_logits = model(image, text_token)
            else:
                if main_modality.lower() == 'text':
                    clean_logits = model(text_token)
                else:
                    clean_logits = model(image)

            pred = torch.argmax(clean_logits, dim=1)
            correct_clean += (pred == labels).sum().item()
            total_clean += labels.size(0)
    
    # Evaluate on adversarial examples
    for _ in range(num_samples):
        for batch_idx, (image, text, labels) in enumerate(testloader): 
            image = image.float().to(device)
            labels = labels.long().to(device)
            text_token = tokenizer(text, padding='max_length', max_length=40, truncation=True, return_tensors='pt').to(device)

            # Generate adversarial examples
            model.eval()  # Need gradients for attacks
            
            with torch.enable_grad():
                if attack_type == 'fgsm':
                    if main_modality.lower() == 'text':
                        text_token = attacker.fgsm_attack(model, image, text_token, labels)
                    else:
                        image = attacker.fgsm_attack(model, image, text_token, labels)
                elif attack_type == 'pgd':
                    if main_modality.lower() == 'text':
                        text_token = attacker.pgd_attack(model, image, text_token, labels)
                    else:
                        image = attacker.pgd_attack(model, image, text_token, labels)
                elif attack_type == 'fgsm_multimodal':
                    text_token, image = attacker.fgsm_multimodal_attack(model, image, text_token, labels)
                elif attack_type == 'pgd_multimodal':
                    text_token, image = attacker.pgd_multimodal_attack(model, image, text_token, labels)
            
            # Evaluate adversarial examples
            model.eval()
            with torch.inference_mode():
                if multimodal is True:
                    logits_adv = model(image, text_token)
                else:
                    if main_modality.lower() == 'image':
                        logits_adv = model(image)
                    else:
                        logits_adv = model(text_token)
                        
                pred = torch.argmax(logits_adv, dim=1)
                correct_adv += (pred == labels).sum().item()
                total_adv += labels.size(0)
    
    acc_clean = 100 * correct_clean / total_clean
    acc_adv = 100 * correct_adv / total_adv
    
    return acc_clean, acc_adv



def build_model(cfgs):
    if cfgs.modality == 'Text':
        net = TextEncoder(num_classes = cfgs.num_classes, feature_only=False)
    elif cfgs.modality == 'Image':
        net = VisionEncoder(model_arch='resnet18', num_classes=cfgs.num_classes, feature_only=False)
    elif cfgs.modality == 'Multimodal' or cfgs.modality == 'Unimodal':
        net = LateFusionClassifier(cfgs, num_classes=cfgs.num_classes)
    return net

def evaluate_and_store_results(model_name, model, dataloader, device, attacker, attack_type, results_dict, logger, **kwargs):
    """
    Evaluates a model, stores the accuracy results, and prints them.
    """
    logger.info(f"Evaluating AVMNIST {model_name} on clean and adversarial examples ({attack_type} attack)...")
    acc_clean, acc_adv = adversarial_evaluation(model, dataloader, device, attacker=attacker, attack_type=attack_type, **kwargs)
    logger.info(f"Clean accuracy: {acc_clean:.2f}%")
    logger.info(f"Adversarial accuracy ({attack_type}): {acc_adv:.2f}%")


    results_dict[model_name]['clean'].append(acc_clean)
    results_dict[model_name]['adv'].append(acc_adv)

def main():
    analysis_args = parse_analysis_args()
    manifest = load_manifest(analysis_args.manifest)

    
    logger = get_logger(f"Food101", logger_dir="results/adversarial_robustness")
    
    vision_cfgs = Config()
    vision_cfgs.modality = 'Image'

    text_cfgs = Config()
    text_cfgs.modality = 'Text'

    multimodal_cfgs = Config()
    multimodal_cfgs.modality = 'Multimodal'
    multimodal_cfgs.seed = 1234

    multimodal_cfgs2 = Config()
    multimodal_cfgs2.modality = 'Multimodal' 

    loader = Food101DataLoader(multimodal_cfgs)
    # dataloader = loader.train_dataloader
    dataloader = loader.test_dataloader
    
  
    text_model = build_model(text_cfgs)
    vision_model = build_model(vision_cfgs)
    multimodal_model = build_model(multimodal_cfgs)
    multimodal_model2 = build_model(multimodal_cfgs2)
    
    vision_results = {'clean': [], 'adv': []}
    multimodal_vision_results = {'clean': [], 'adv': []}
    multimodal2_vision_results = {'clean': [], 'adv': []}
    
    text_results = {'clean': [], 'adv': []}
    multimodal_text_results = {'clean': [], 'adv': []}
    multimodal2_text_results = {'clean': [], 'adv': []}
    
    all_results = {
        'Image Model': vision_results,
        'Multimodal Image Model 1': multimodal_vision_results,
        'Multimodal Image Model 2': multimodal2_vision_results,
        'Text Model': text_results,
        'Multimodal Text Model 1': multimodal_text_results,
        'Multimodal Text Model 2': multimodal2_text_results
    }
    
    
     # --- Run 1 ---
    logger.info("--- RUN 0 ---")

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
        variant="v2_dml",
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
    
    # text_attacker = AdversarialAttacker(text_model, epsilon=0.01, alpha=0.001, main_modality='text', multimodal=False)
    # evaluate_and_store_results('Text Model', text_model, dataloader, 'cuda', main_modality='text', attacker=text_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    # multimodal_attacker = AdversarialAttacker(multimodal_model, epsilon=0.01, alpha=0.001, main_modality='text', multimodal=True)
    # evaluate_and_store_results('Multimodal Text Model 1', multimodal_model, dataloader, 'cuda', main_modality='text', multimodal=True, attacker=multimodal_attacker, attack_type='pgd', results_dict=all_results, logger=logger)

    # multimodal_attacker2 = AdversarialAttacker(multimodal_model2, epsilon=0.01, alpha=0.001, main_modality='text', multimodal=True)
    # evaluate_and_store_results('Multimodal Text Model 2', multimodal_model2, dataloader, 'cuda', main_modality='text', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)

    vision_attacker = AdversarialAttacker(vision_model, epsilon=0.1, alpha=0.0005, iterations=3, main_modality='image', multimodal=False)
    evaluate_and_store_results('Image Model', vision_model, dataloader, 'cuda', main_modality='image', attacker=vision_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    multimodal_attacker = AdversarialAttacker(multimodal_model, epsilon=0.1, alpha=0.0005, iterations=3, main_modality='image', multimodal=True)
    evaluate_and_store_results('Multimodal Image Model 1', multimodal_model, dataloader, 'cuda', main_modality='image', multimodal=True, attacker=multimodal_attacker, attack_type='pgd',results_dict=all_results, logger=logger)

    multimodal_attacker2 = AdversarialAttacker(multimodal_model2, epsilon=0.1, alpha=0.0005, iterations=3, main_modality='image', multimodal=True)
    evaluate_and_store_results('Multimodal Image Model 2', multimodal_model2, dataloader, 'cuda', main_modality='image', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)

    # --- Run 1 ---
    logger.info("--- RUN 1 ---")
    text_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Text",
        variant="standard",
        seed="159",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    vision_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Visual",
        variant="standard",
        seed="159",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    multimodal_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Multimodal",
        variant="sml",
        seed="159",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    multimodal_checkpoint2 = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Multimodal",
        variant="dml",
        seed="159",
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

    
    # text_attacker = AdversarialAttacker(text_model, epsilon=0.01, alpha=0.001, main_modality='text', multimodal=False)
    # evaluate_and_store_results('Text Model', text_model, dataloader, 'cuda', main_modality='text', attacker=text_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    # multimodal_attacker = AdversarialAttacker(multimodal_model, epsilon=0.01, alpha=0.001, main_modality='text', multimodal=True)
    # evaluate_and_store_results('Multimodal Text Model 1', multimodal_model, dataloader, 'cuda', main_modality='text', multimodal=True, attacker=multimodal_attacker, attack_type='pgd', results_dict=all_results, logger=logger)

    # multimodal_attacker2 = AdversarialAttacker(multimodal_model2, epsilon=0.01, alpha=0.001, main_modality='text', multimodal=True)
    # evaluate_and_store_results('Multimodal Text Model 2', multimodal_model2, dataloader, 'cuda', main_modality='text', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)

    vision_attacker = AdversarialAttacker(vision_model, epsilon=0.1, alpha=0.0005, iterations=3, main_modality='image', multimodal=False)
    evaluate_and_store_results('Image Model', vision_model, dataloader, 'cuda', main_modality='image', attacker=vision_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    multimodal_attacker = AdversarialAttacker(multimodal_model, epsilon=0.1, alpha=0.0005, iterations=3, main_modality='image', multimodal=True)
    evaluate_and_store_results('Multimodal Image Model 1', multimodal_model, dataloader, 'cuda', main_modality='image', multimodal=True, attacker=multimodal_attacker, attack_type='pgd',results_dict=all_results, logger=logger)

    multimodal_attacker2 = AdversarialAttacker(multimodal_model2, epsilon=0.1, alpha=0.0005, iterations=3, main_modality='image', multimodal=True)
    evaluate_and_store_results('Multimodal Image Model 2', multimodal_model2, dataloader, 'cuda', main_modality='image', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)


    logger.info("--- RUN 2 ---")
    text_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Text",
        variant="standard",
        seed="246",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    vision_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Visual",
        variant="standard",
        seed="246",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    multimodal_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Multimodal",
        variant="sml",
        seed="246",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    multimodal_checkpoint2 = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="Food101",
        modality="Multimodal",
        variant="dml",
        seed="246",
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

    
    # text_attacker = AdversarialAttacker(text_model, epsilon=0.01, alpha=0.001, main_modality='text', multimodal=False)
    # evaluate_and_store_results('Text Model', text_model, dataloader, 'cuda', main_modality='text', attacker=text_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    # multimodal_attacker = AdversarialAttacker(multimodal_model, epsilon=0.01, alpha=0.001, main_modality='text', multimodal=True)
    # evaluate_and_store_results('Multimodal Text Model 1', multimodal_model, dataloader, 'cuda', main_modality='text', multimodal=True, attacker=multimodal_attacker, attack_type='pgd', results_dict=all_results, logger=logger)

    # multimodal_attacker2 = AdversarialAttacker(multimodal_model2, epsilon=0.01, alpha=0.001, main_modality='text', multimodal=True)
    # evaluate_and_store_results('Multimodal Text Model 2', multimodal_model2, dataloader, 'cuda', main_modality='text', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)

    vision_attacker = AdversarialAttacker(vision_model, epsilon=0.1, alpha=0.0005, iterations=3, main_modality='image', multimodal=False)
    evaluate_and_store_results('Image Model', vision_model, dataloader, 'cuda', main_modality='image', attacker=vision_attacker, attack_type='pgd', results_dict=all_results, logger=logger)
    
    multimodal_attacker = AdversarialAttacker(multimodal_model, epsilon=0.1, alpha=0.0005, iterations=3, main_modality='image', multimodal=True)
    evaluate_and_store_results('Multimodal Image Model 1', multimodal_model, dataloader, 'cuda', main_modality='image', multimodal=True, attacker=multimodal_attacker, attack_type='pgd',results_dict=all_results, logger=logger)

    multimodal_attacker2 = AdversarialAttacker(multimodal_model2, epsilon=0.1, alpha=0.0005, iterations=3, main_modality='image', multimodal=True)
    evaluate_and_store_results('Multimodal Image Model 2', multimodal_model2, dataloader, 'cuda', main_modality='image', multimodal=True, attacker=multimodal_attacker2, attack_type='pgd', results_dict=all_results, logger=logger)


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
        logger.info(f"  Adversarial Accuracy (PGD): {mean_adv:.2f}% (Std: {std_adv:.2f}%)")


if __name__ == "__main__":
    main()