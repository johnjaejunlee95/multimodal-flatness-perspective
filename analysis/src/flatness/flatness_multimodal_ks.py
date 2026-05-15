import os
import sys
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "main.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root containing main.py.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataloader.KineticSound_loader import KineticSoundLoader
from model.ResNet import *
from utils.config import Config
from utils.checkpoint_tools import load_state_dict_from_manifest
from analysis.scripts.common.analysis_runtime import load_manifest, parse_analysis_args


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


@torch.no_grad()
def test(model, dataloader, device, main_modality, multimodal=False):
    model.eval()
    local_total_loss = 0.0
    local_count = 0

    for audio, vision, labels in dataloader:
        audio = audio.float().to(device, non_blocking=True)
        vision = vision.float().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        if multimodal:
            logits = model(audio, vision)
        elif main_modality.lower() == "vision":
            logits = model(vision)
        else:
            logits = model(audio)

        loss = F.cross_entropy(logits, labels, reduction="sum")
        local_total_loss += loss.item()
        local_count += labels.size(0)

    total_loss_tensor = torch.tensor([local_total_loss], device=device)
    count_tensor = torch.tensor([local_count], device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    return total_loss_tensor.item() / max(count_tensor.item(), 1)


@torch.no_grad()
def flatten_params(model):
    params = model.module.parameters() if isinstance(model, nn.parallel.DistributedDataParallel) else model.parameters()
    return torch.cat([p.data.view(-1) for p in params], dim=0)


@torch.no_grad()
def assign_flat_params(model, flat_tensor):
    params = model.module.parameters() if isinstance(model, nn.parallel.DistributedDataParallel) else model.parameters()
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat_tensor[offset : offset + numel].view_as(p))
        offset += numel


def mc_flatness_inplace(model, dataloader, device, radius_list, mc_num=5, main_modality="vision", multimodal=False):
    print(f"[Rank {dist.get_rank()}] Computing {main_modality.lower()} flatness with {mc_num} MC iterations")
    flatness_results = []
    original_params = flatten_params(model).clone()
    num_param = original_params.numel()

    for radius in radius_list:
        if int(radius) == 0:
            flatness_results.append(test(model, dataloader, device, main_modality, multimodal))
            continue

        avg_loss = 0.0
        for _ in range(mc_num):
            random_param = torch.randn(num_param, device=device)
            random_unit_param = random_param / torch.norm(random_param)
            perturbed_params = original_params + random_unit_param * radius
            assign_flat_params(model, perturbed_params)
            avg_loss += test(model, dataloader, device, main_modality, multimodal)

        avg_loss /= mc_num
        flatness_results.append(avg_loss)
        assign_flat_params(model, original_params)

    return flatness_results


def build_model(cfgs):
    if cfgs.modality == "Audio":
        return Audio_ResNet(cfgs, num_classes=cfgs.num_classes)
    if cfgs.modality == "Visual":
        return Visual_ResNet(cfgs, modality="visual", num_classes=cfgs.num_classes)
    if cfgs.modality == "Multimodal":
        return LateFusion_ResNet(cfgs, modality2="visual", num_classes=cfgs.num_classes)
    raise NotImplementedError


def main():
    analysis_args = parse_analysis_args()
    manifest = load_manifest(analysis_args.manifest)

    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    vision_cfgs = Config(); vision_cfgs.modality = "Visual"
    audio_cfgs = Config(); audio_cfgs.modality = "Audio"
    multimodal_cfgs = Config(); multimodal_cfgs.modality = "Multimodal"
    multimodal_cfgs2 = Config(); multimodal_cfgs2.modality = "Multimodal"
    multimodal_cfgs.seed = 123

    loader = KineticSoundLoader(multimodal_cfgs, split=True)
    dataset = loader.train_dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=multimodal_cfgs.batch_size,
        num_workers=multimodal_cfgs.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    audio_model = build_model(audio_cfgs)
    vision_model = build_model(vision_cfgs)
    multimodal_model = build_model(multimodal_cfgs)
    multimodal_model2 = build_model(multimodal_cfgs2)

    audio_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="KineticSound",
        modality="Audio",
        variant="standard",
        seed="135",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    vision_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="KineticSound",
        modality="Visual",
        variant="standard",
        seed="135",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    multimodal_checkpoint = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="KineticSound",
        modality="Multimodal",
        variant="sml",
        seed="135",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )
    multimodal_checkpoint2 = load_state_dict_from_manifest(
        manifest=manifest,
        dataset="KineticSound",
        modality="Multimodal",
        variant="dml",
        seed="135",
        map_location="cpu",
        profile=analysis_args.profile,
        ckpt_root_override=analysis_args.ckpt_root_override,
    )

    audio_model.load_state_dict(audio_checkpoint)
    vision_model.load_state_dict(vision_checkpoint)
    multimodal_model.load_state_dict(multimodal_checkpoint)
    multimodal_model2.load_state_dict(multimodal_checkpoint2)

    audio_model.to(device)
    vision_model.to(device)
    multimodal_model.to(device)
    multimodal_model2.to(device)

    audio_model = DDP(audio_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    vision_model = DDP(vision_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    multimodal_model = DDP(multimodal_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    multimodal_model2 = DDP(multimodal_model2, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    radius_list = np.arange(0, 31, 3)
    monte_carlo_iter = 20
    mode = "train"

    audio_flatness = mc_flatness_inplace(audio_model, dataloader, device, radius_list, monte_carlo_iter, "Audio")
    vision_flatness = mc_flatness_inplace(vision_model, dataloader, device, radius_list, monte_carlo_iter, "Vision")
    multimodal_flatness = mc_flatness_inplace(
        multimodal_model,
        dataloader,
        device,
        radius_list,
        monte_carlo_iter,
        main_modality="Multimodal",
        multimodal=True,
    )
    multimodal_flatness2 = mc_flatness_inplace(
        multimodal_model2,
        dataloader,
        device,
        radius_list,
        monte_carlo_iter,
        main_modality="Multimodal",
        multimodal=True,
    )

    if rank == 0:
        os.makedirs("results/flatness/KineticsSounds", exist_ok=True)
        torch.save(audio_flatness, f"results/flatness/KineticsSounds/audio_flatness_{mode}.pth")
        torch.save(vision_flatness, f"results/flatness/KineticsSounds/vision_flatness_{mode}.pth")
        torch.save(multimodal_flatness, f"results/flatness/KineticsSounds/multimodal_flatness_{mode}.pth")
        torch.save(multimodal_flatness2, f"results/flatness/KineticsSounds/multimodal_flatness2_{mode}.pth")

    cleanup_ddp()


if __name__ == "__main__":
    main()
