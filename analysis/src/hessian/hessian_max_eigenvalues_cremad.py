import sys
from pathlib import Path
import os

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "main.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root containing main.py.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch

from utils.config import Config
from utils.logger import *
from dataloader.CREMAD_loader import CREMAD_Dataloder
from model.ResNet import *
from utils import hessian
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split, DataLoader
from utils.checkpoint_tools import load_state_dict_from_manifest
from analysis.scripts.common.analysis_runtime import load_manifest, parse_analysis_args


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


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _reduce_max(local_values, device, distributed):
    local_max = max((_to_float(v) for v in local_values), default=float("-inf"))
    max_tensor = torch.tensor(local_max, dtype=torch.float32, device=device)
    if distributed:
        torch.distributed.all_reduce(max_tensor, op=torch.distributed.ReduceOp.MAX)
    return float(max_tensor.item())


def main():
    analysis_args = parse_analysis_args()
    manifest = load_manifest(analysis_args.manifest)

    vision_cfgs = Config()
    vision_cfgs.modality = "Visual"

    audio_cfgs = Config()
    audio_cfgs.modality = "Audio"

    multimodal_cfgs = Config()
    multimodal_cfgs.modality = "Multimodal"

    multimodal_cfgs2 = Config()
    multimodal_cfgs2.modality = "Multimodal"

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1
    world_size = 1
    rank = 0
    local_rank = 0
    if distributed:
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(local_rank)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print("Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d." % (rank, world_size))
    else:
        print("Training with a single process on 1 GPUs.")
    assert rank >= 0

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    loader = CREMAD_Dataloder(multimodal_cfgs)
    dataloader = loader.train_dataloader

    percentage = 1.0
    subset_size = int(percentage * len(dataloader.dataset))
    datasets, _ = random_split(dataloader.dataset, [subset_size, len(dataloader.dataset) - subset_size])

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(datasets, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        datasets,
        batch_size=16,
        shuffle=(train_sampler is None),
        num_workers=8,
        sampler=train_sampler,
        drop_last=False,
    )

    vision_model = build_model(vision_cfgs)
    audio_model = build_model(audio_cfgs)
    multimodal_model = build_model(multimodal_cfgs)
    multimodal_model2 = build_model(multimodal_cfgs2)

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

    vision_model.load_state_dict(vision_checkpoint)
    audio_model.load_state_dict(audio_checkpoint)
    multimodal_model.load_state_dict(multimodal_checkpoint)
    multimodal_model2.load_state_dict(multimodal_checkpoint2)

    vision_model = vision_model.to(device)
    audio_model = audio_model.to(device)
    multimodal_model = multimodal_model.to(device)
    multimodal_model2 = multimodal_model2.to(device)

    if distributed:
        audio_model = DDP(audio_model, device_ids=[local_rank], broadcast_buffers=False)
        vision_model = DDP(vision_model, device_ids=[local_rank], broadcast_buffers=False)
        multimodal_model = DDP(multimodal_model, device_ids=[local_rank], broadcast_buffers=False)
        multimodal_model2 = DDP(multimodal_model2, device_ids=[local_rank], broadcast_buffers=False)

        audio_model = audio_model.module
        vision_model = vision_model.module
        multimodal_model = multimodal_model.module
        multimodal_model2 = multimodal_model2.module

    loss_fn = torch.nn.CrossEntropyLoss()

    output_dir = "results/hessian_eigenvalues/CREMAD/"
    os.makedirs(output_dir, exist_ok=True)

    print("Build CREMAD Hessian dataloaders")
    audio_hessian_dataloader = []
    vision_hessian_dataloader = []
    multimodal_hessian_dataloader = []
    multimodal_hessian_dataloader2 = []
    for spec, image, targets in dataloader:
        spec = spec.unsqueeze(1).float()
        image = image.float()
        audio_hessian_dataloader.append((spec, targets))
        vision_hessian_dataloader.append((image, targets))
        multimodal_hessian_dataloader.append((spec, image, targets))
        multimodal_hessian_dataloader2.append((spec, image, targets))

    print("Run CREMAD Hessian eigencalc (full local shard)")
    audio_hessian_comp = hessian(audio_model, loss_fn, dataloader=audio_hessian_dataloader, cuda=use_cuda)
    vision_hessian_comp = hessian(vision_model, loss_fn, dataloader=vision_hessian_dataloader, cuda=use_cuda)
    multimodal_hessian_comp = hessian(multimodal_model, loss_fn, dataloader=multimodal_hessian_dataloader, cuda=use_cuda)
    multimodal_hessian_comp2 = hessian(multimodal_model2, loss_fn, dataloader=multimodal_hessian_dataloader2, cuda=use_cuda)

    audio_top_eigenvalues, _ = audio_hessian_comp.eigenvalues(top_n=1)
    vision_top_eigenvalues, _ = vision_hessian_comp.eigenvalues(top_n=1)
    multimodal_top_eigenvalues, _ = multimodal_hessian_comp.eigenvalues(top_n=1)
    multimodal_top_eigenvalues2, _ = multimodal_hessian_comp2.eigenvalues(top_n=1)

    audio_max_eigens = [audio_top_eigenvalues[0]]
    vision_max_eigens = [vision_top_eigenvalues[0]]
    multimodal_max_eigens = [multimodal_top_eigenvalues[0]]
    multimodal_max_eigens2 = [multimodal_top_eigenvalues2[0]]

    torch.save(audio_max_eigens, f"{output_dir}#audio_max_eigens_{local_rank}.pth")
    torch.save(vision_max_eigens, f"{output_dir}#vision_max_eigens_{local_rank}.pth")
    torch.save(multimodal_max_eigens, f"{output_dir}#multimodal_max_eigens_{local_rank}.pth")
    torch.save(multimodal_max_eigens2, f"{output_dir}#multimodal_max_eigens2_{local_rank}.pth")

    merged_max_eigens = {
        "audio": _reduce_max(audio_max_eigens, device, distributed),
        "vision": _reduce_max(vision_max_eigens, device, distributed),
        "multimodal": _reduce_max(multimodal_max_eigens, device, distributed),
        "multimodal2": _reduce_max(multimodal_max_eigens2, device, distributed),
    }
    if rank == 0:
        torch.save(merged_max_eigens, f"{output_dir}merged_max_eigens.pth")

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
