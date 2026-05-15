import sys
import os
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "main.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root containing main.py.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from dataloader.Food_loader import Food101DataLoader
from model.FoodNet import *
from utils.config import Config
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
import torch.nn.functional as F
from utils.checkpoint_tools import load_state_dict_from_manifest
from analysis.scripts.common.analysis_runtime import load_manifest, parse_analysis_args

# ===================== DDP INIT & CLEANUP =====================
def setup_ddp():
    """Initialize torch.distributed and return local rank."""
    dist.init_process_group(backend='nccl')  # GPU backend
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

# ===================== TOKENIZER =====================
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

# ===================== TEST LOOP (global + per-GPU losses) =====================
@torch.no_grad()
def test(model, dataloader, main_modality, multimodal=False, device=None, return_all=False):
    """
    return_all=False → return global average loss
    return_all=True  → return per-GPU losses list (rank 0 only)
    """
    model.eval()
    local_total_loss = 0.0
    local_count = 0

    for image, text, label in dataloader:
        image = image.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)
        text_token = tokenize_batch(text)
        text_token = {k: v.to(device, non_blocking=True) for k, v in text_token.items()}

        if multimodal:
            logits = model(image, text_token)
        else:
            if main_modality.lower() == 'vision':
                logits = model(image)
            else:
                logits = model(text_token)

        loss = F.cross_entropy(logits, label, reduction='sum')
        local_total_loss += loss.item()
        local_count += label.size(0)

    # Compute per-GPU average loss
    local_avg_loss = local_total_loss / max(local_count, 1)
    device = torch.device(device)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Gather per-GPU average losses
    local_loss_tensor = torch.tensor([local_avg_loss], device=device)
    all_losses_tensor = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_losses_tensor, local_loss_tensor)
    per_gpu_losses = [x.item() for x in all_losses_tensor]

    if return_all:
        if rank == 0:
            return per_gpu_losses  # only rank 0 returns
        else:
            return None
    else:
        # Compute global average loss using total sums
        total_loss_tensor = torch.tensor([local_total_loss], device=device)
        count_tensor = torch.tensor([local_count], device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / max(count_tensor.item(), 1)
        return avg_loss

# ===================== PARAM UTILS =====================
@torch.no_grad()
def flatten_params(model):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        params = model.module.parameters()
    else:
        params = model.parameters()
    return torch.cat([p.data.view(-1) for p in params], dim=0)

@torch.no_grad()
def assign_flat_params(model, flat_tensor):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        params = model.module.parameters()
    else:
        params = model.parameters()
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat_tensor[offset:offset+numel].view_as(p))
        offset += numel

# ===================== MC FLATNESS INPLACE =====================
def mc_flatness_inplace(model, dataloader, radius_list, MC_num=5,
                        main_modality='vision', multimodal=False, device=None, return_all=False):
    """
    If return_all=True, returns per-GPU losses for each radius (rank 0 only).
    """
    print(f"[Rank {dist.get_rank()}] Computing {main_modality.lower()} flatness with {MC_num} MC iterations")
    flatness_results = []

    original_params = flatten_params(model).clone()
    num_param = original_params.numel()

    for radius in radius_list:
        if int(radius) == 0:
            loss = test(model, dataloader, main_modality, multimodal, device, return_all=return_all)
            flatness_results.append(loss)
            continue

        avg_loss = 0.0
        for _ in range(MC_num):
            random_param = torch.randn(num_param, device=device)
            random_unit_param = random_param / torch.norm(random_param)
            rand_vector = random_unit_param * radius

            perturbed_params = original_params + rand_vector
            assign_flat_params(model, perturbed_params)

            perturbed_loss = test(model, dataloader, main_modality, multimodal, device, return_all=False)
            avg_loss += perturbed_loss

        avg_loss /= MC_num
        flatness_results.append(avg_loss)
        assign_flat_params(model, original_params)

    return flatness_results

# ===================== MAIN =====================
def main():
    analysis_args = parse_analysis_args()
    manifest = load_manifest(analysis_args.manifest)

    # ---------- DDP setup ----------
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # ---------- Configs ----------
    vision_cfgs = Config(); vision_cfgs.modality = 'Visual'
    text_cfgs = Config(); text_cfgs.modality = 'Text'
    multimodal_cfgs = Config(); multimodal_cfgs.modality = 'Multimodal'
    multimodal_cfgs2 = Config(); multimodal_cfgs2.modality = 'Multimodal'

    # ---------- Dataset + DistributedSampler ----------
    loader = Food101DataLoader(multimodal_cfgs)
    dataset = loader.test_dataset
    mode = 'test'
    # dataset = loader.train_dataset
    # mode = 'train'
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=multimodal_cfgs.batch_size,
        num_workers=multimodal_cfgs.num_workers,
        pin_memory=True,
        sampler=sampler
    )

    # ---------- Build Models ----------
    def build_model(cfgs):
        cfgs.num_classes = 101
        if cfgs.modality == 'Text':
            net = TextEncoder(num_classes=cfgs.num_classes, feature_only=False)
        elif cfgs.modality == 'Visual':
            net = VisionEncoder(model_arch='resnet18', num_classes=cfgs.num_classes, feature_only=False)
        else:
            net = LateFusionClassifier(cfgs, num_classes=cfgs.num_classes)
        return net

    text_model = build_model(text_cfgs)
    vision_model = build_model(vision_cfgs)
    multimodal_model = build_model(multimodal_cfgs)
    multimodal_model2 = build_model(multimodal_cfgs2)

    # Load checkpoints here (map_location='cpu') if needed
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
    mutlimodal_checkpoint2 = load_state_dict_from_manifest(
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
    multimodal_model2.load_state_dict(mutlimodal_checkpoint2)

    # ---------- Move to device & wrap in DDP ----------
    text_model.to(device)
    vision_model.to(device)
    multimodal_model.to(device)
    multimodal_model2.to(device)

    text_model = DDP(text_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    vision_model = DDP(vision_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    multimodal_model = DDP(multimodal_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    multimodal_model2 = DDP(multimodal_model2, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # ---------- Run MC Flatness ----------
    radius_list = np.arange(0, 21, 2)
    monte_carlo_iter = 30
    
    if rank == 0:
        print(f"Radius list: {radius_list}")
        print(f"Monte Carlo iterations: {monte_carlo_iter}")
        print(f"Dataset mode: {mode}, Dataloader size: {len(dataloader)}, Batch size per GPU: {multimodal_cfgs.batch_size}, World size: {world_size}")

    # Example: per-GPU losses at radius=0 (rank 0 collects)
    # per_gpu_losses = test(text_model, dataloader, "Text", device=device, return_all=True)
    # if rank == 0 and per_gpu_losses is not None:
    #     print("Per-GPU losses at radius=0:", per_gpu_losses)
    #     torch.save(per_gpu_losses, f"results/flatness/Food101/per_gpu_losses_text_r0.pth")

    # Global average flatness curves
    vision_flatness = mc_flatness_inplace(vision_model, dataloader, radius_list, monte_carlo_iter, "Vision", device=device)
    text_flatness = mc_flatness_inplace(text_model, dataloader, radius_list, monte_carlo_iter, "Text", device=device)
    
    multimodal_flatness = mc_flatness_inplace(multimodal_model, dataloader, radius_list, monte_carlo_iter,
                                              main_modality='Multimodal', multimodal=True, device=device)
    multimodal_flatness2 = mc_flatness_inplace(multimodal_model2, dataloader, radius_list, monte_carlo_iter,
                                               main_modality='Multimodal', multimodal=True, device=device)

    # ---------- Save only on rank 0 ----------
    if rank == 0:
        os.makedirs("results/flatness/Food101", exist_ok=True)
        torch.save(text_flatness, f"results/flatness/Food101/text_flatness_{mode}v2.pth")
        torch.save(vision_flatness, f"results/flatness/Food101/vision_flatness_{mode}v2.pth")
        torch.save(multimodal_flatness, f"results/flatness/Food101/SML_flatness_{mode}v2.pth")
        torch.save(multimodal_flatness2, f"results/flatness/Food101/DML_flatness2_{mode}v2.pth")

    cleanup_ddp()

if __name__ == "__main__":
    main()
