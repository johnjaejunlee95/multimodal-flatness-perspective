import sys
from pathlib import Path
import os

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "main.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root containing main.py.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch

from utils.logger import *
from utils.config import Config
from dataloader.Food_loader import Food101DataLoader
from model.FoodNet import *
from utils import hessian
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split, DataLoader
from utils.checkpoint_tools import load_state_dict_from_manifest
from analysis.scripts.common.analysis_runtime import load_manifest, parse_analysis_args


def build_model(cfgs):
    if cfgs.modality == "Text":
        net = TextEncoder(num_classes=cfgs.num_classes, feature_only=False)
    elif cfgs.modality == "Visual":
        net = VisionEncoder(model_arch="resnet18", num_classes=cfgs.num_classes, feature_only=False)
    elif cfgs.modality == "Multimodal" or cfgs.modality == "Unimodal":
        net = LateFusionClassifier(cfgs, num_classes=cfgs.num_classes)
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

    visual_cfgs = Config()
    visual_cfgs.modality = "Visual"

    text_cfgs = Config()
    text_cfgs.modality = "Text"

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

    text_model = build_model(text_cfgs)
    visual_model = build_model(visual_cfgs)
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
    visual_model.load_state_dict(vision_checkpoint)
    multimodal_model.load_state_dict(multimodal_checkpoint)
    multimodal_model2.load_state_dict(multimodal_checkpoint2)

    text_model = text_model.to(device)
    visual_model = visual_model.to(device)
    multimodal_model = multimodal_model.to(device)
    multimodal_model2 = multimodal_model2.to(device)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    if distributed:
        text_model = DDP(text_model, device_ids=[local_rank], broadcast_buffers=False)
        visual_model = DDP(visual_model, device_ids=[local_rank], broadcast_buffers=False)
        multimodal_model = DDP(multimodal_model, device_ids=[local_rank], broadcast_buffers=False)
        multimodal_model2 = DDP(multimodal_model2, device_ids=[local_rank], broadcast_buffers=False)

        text_model = text_model.module
        visual_model = visual_model.module
        multimodal_model = multimodal_model.module
        multimodal_model2 = multimodal_model2.module

    multimodal_cfgs.seed = 123

    loader = Food101DataLoader(multimodal_cfgs)
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

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        cache_dir="/nfs2/jjlee/model_cache",
        add_prefix_space=False,
    )

    output_dir = "results/hessian_eigenvalues/Food101/"
    os.makedirs(output_dir, exist_ok=True)

    print("Build Food101 Hessian dataloaders")
    text_hessian_dataloader = []
    visual_hessian_dataloader = []
    multimodal_hessian_dataloader = []
    multimodal_hessian_dataloader2 = []
    for visual, text, targets in dataloader:
        text_token = tokenizer(text, padding="max_length", max_length=40, truncation=True, return_tensors="pt")
        text_hessian_dataloader.append((text_token, targets))
        visual_hessian_dataloader.append((visual, targets))
        multimodal_hessian_dataloader.append((visual, text_token, targets))
        multimodal_hessian_dataloader2.append((visual, text_token, targets))

    print("Run Food101 Hessian eigencalc (full local shard)")
    text_hessian_comp = hessian(text_model, loss_fn, dataloader=text_hessian_dataloader, cuda=use_cuda)
    visual_hessian_comp = hessian(visual_model, loss_fn, dataloader=visual_hessian_dataloader, cuda=use_cuda)
    multimodal_hessian_comp = hessian(multimodal_model, loss_fn, dataloader=multimodal_hessian_dataloader, cuda=use_cuda)
    multimodal_hessian_comp2 = hessian(multimodal_model2, loss_fn, dataloader=multimodal_hessian_dataloader2, cuda=use_cuda)

    text_top_eigenvalues, _ = text_hessian_comp.eigenvalues(top_n=1)
    visual_top_eigenvalues, _ = visual_hessian_comp.eigenvalues(top_n=1)
    multimodal_top_eigenvalues, _ = multimodal_hessian_comp.eigenvalues(top_n=1)
    multimodal_top_eigenvalues2, _ = multimodal_hessian_comp2.eigenvalues(top_n=1)

    text_max_eigens = [text_top_eigenvalues[0]]
    visual_max_eigens = [visual_top_eigenvalues[0]]
    multimodal_max_eigens = [multimodal_top_eigenvalues[0]]
    multimodal_max_eigens2 = [multimodal_top_eigenvalues2[0]]

    torch.save(text_max_eigens, f"{output_dir}#text_max_eigens_{local_rank}.pth")
    torch.save(visual_max_eigens, f"{output_dir}#visual_max_eigens_{local_rank}.pth")
    torch.save(multimodal_max_eigens, f"{output_dir}#multimodal_max_eigens_{local_rank}.pth")
    torch.save(multimodal_max_eigens2, f"{output_dir}#multimodal_max_eigens2_{local_rank}.pth")

    merged_max_eigens = {
        "text": _reduce_max(text_max_eigens, device, distributed),
        "vision": _reduce_max(visual_max_eigens, device, distributed),
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
