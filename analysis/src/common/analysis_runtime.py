import argparse
import copy
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import yaml

from utils.config import Config


DEFAULT_MANIFEST = "analysis/configs/checkpoints.yaml"


@dataclass
class AnalysisArgs:
    manifest: str
    profile: Optional[str]
    ckpt_root_override: Optional[str]


@dataclass
class ManifestBundle:
    manifest: Dict[str, Any]
    dataset: str
    args: AnalysisArgs


def parse_analysis_args(default_manifest: str = DEFAULT_MANIFEST) -> AnalysisArgs:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=str, default=default_manifest)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--ckpt-root-override", type=str, default=None)

    args, unknown = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *unknown]
    return AnalysisArgs(
        manifest=args.manifest,
        profile=args.profile,
        ckpt_root_override=args.ckpt_root_override,
    )


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a dictionary: {manifest_path}")
    if "datasets" not in manifest:
        raise KeyError(f"Manifest missing 'datasets': {manifest_path}")
    return manifest


def _normalize_dataset_name(manifest: Mapping[str, Any], dataset: str) -> str:
    datasets = manifest.get("datasets", {})
    if dataset in datasets:
        return dataset

    dataset_lower = dataset.lower()
    for name, block in datasets.items():
        aliases = block.get("aliases", [])
        if any(str(alias).lower() == dataset_lower for alias in aliases):
            return name

    raise KeyError(f"Dataset '{dataset}' not found in manifest datasets/aliases")


def _resolve_roots(manifest: Mapping[str, Any], override_root: Optional[str]) -> Dict[str, str]:
    roots = copy.deepcopy(manifest.get("roots", {}))
    if "project" not in roots:
        raise KeyError("Manifest roots must include 'project'")

    # Resolve references like {project}, {baseline}, etc.
    for _ in range(5):
        changed = False
        for key, value in list(roots.items()):
            if isinstance(value, str):
                new_value = value.format(**roots)
                if new_value != value:
                    roots[key] = new_value
                    changed = True
        if not changed:
            break

    if override_root:
        override_root = os.path.abspath(override_root)
        roots["project"] = override_root
        for key, value in list(roots.items()):
            if isinstance(value, str):
                roots[key] = value.format(**roots)

    return {k: os.path.abspath(v) if isinstance(v, str) else v for k, v in roots.items()}


def resolve_ckpt(
    manifest: Mapping[str, Any],
    dataset: str,
    modality: str,
    variant: str,
    seed: Optional[Any] = None,
    profile: Optional[str] = None,
    ckpt_root_override: Optional[str] = None,
) -> str:
    normalized_dataset = _normalize_dataset_name(manifest, dataset)
    roots = _resolve_roots(manifest, ckpt_root_override)

    selected_variant = variant
    selected_seed = seed
    if profile:
        profile_data = get_profile(manifest, profile)
        profile_dataset = profile_data.get("dataset")
        if profile_dataset and _normalize_dataset_name(manifest, profile_dataset) != normalized_dataset:
            raise ValueError(
                f"Profile '{profile}' is for dataset '{profile_dataset}', not '{normalized_dataset}'"
            )
        profile_ckpts = profile_data.get("checkpoints", {})
        ckpt_spec = profile_ckpts.get(modality)
        if isinstance(ckpt_spec, dict):
            if selected_variant is None:
                selected_variant = ckpt_spec.get("variant")
            if selected_seed is None:
                selected_seed = ckpt_spec.get("seed")

    dataset_block = manifest["datasets"][normalized_dataset]
    checkpoints = dataset_block.get("checkpoints", {})

    if modality not in checkpoints:
        raise KeyError(f"Modality '{modality}' not found for dataset '{normalized_dataset}'")

    variants = checkpoints[modality]
    if selected_variant not in variants:
        raise KeyError(
            f"Variant '{selected_variant}' not found for dataset '{normalized_dataset}', modality '{modality}'"
        )

    variant_block = variants[selected_variant]
    root_key = variant_block.get("root")
    rel_path = variant_block.get("path")
    default_seed = variant_block.get("default_seed")
    filename = variant_block.get("filename", "ckpt_full_epoch_best.pth.tar")

    if root_key not in roots:
        raise KeyError(f"Root key '{root_key}' not found in manifest roots")

    final_seed = default_seed if selected_seed is None else selected_seed
    if final_seed is None:
        raise ValueError(
            f"Seed is required for {normalized_dataset}/{modality}/{variant} and no default_seed is defined"
        )

    rel_path = rel_path.format(seed=str(final_seed), filename=filename)
    return os.path.join(roots[root_key], rel_path)


def load_checkpoint_state_dict(
    manifest: Mapping[str, Any],
    dataset: str,
    modality: str,
    variant: str,
    seed: Optional[Any] = None,
    map_location: str = "cpu",
    profile: Optional[str] = None,
    ckpt_root_override: Optional[str] = None,
):
    path = resolve_ckpt(
        manifest=manifest,
        dataset=dataset,
        modality=modality,
        variant=variant,
        seed=seed,
        profile=profile,
        ckpt_root_override=ckpt_root_override,
    )
    checkpoint = torch.load(path, map_location=map_location)
    if "state_dict" not in checkpoint:
        raise KeyError(f"'state_dict' missing in checkpoint: {path}")
    return checkpoint["state_dict"]


def get_profile(manifest: Mapping[str, Any], profile: Optional[str], fallback: Optional[str] = None) -> Dict[str, Any]:
    profiles = manifest.get("profiles", {})
    selected = profile or fallback
    if not selected:
        return {}
    if selected not in profiles:
        raise KeyError(f"Profile '{selected}' not found in manifest")
    value = profiles[selected]
    if not isinstance(value, dict):
        raise ValueError(f"Profile '{selected}' must be a mapping")
    return value


def build_cfg_bundle(modality_map: Mapping[str, str]) -> Dict[str, Config]:
    cfg_bundle: Dict[str, Config] = {}
    for key, modality in modality_map.items():
        cfg = Config()
        cfg.modality = modality
        cfg_bundle[key] = cfg
    return cfg_bundle


def build_manifest_bundle(args: AnalysisArgs, dataset: str) -> ManifestBundle:
    manifest = load_manifest(args.manifest)
    normalized_dataset = _normalize_dataset_name(manifest, dataset)
    return ManifestBundle(manifest=manifest, dataset=normalized_dataset, args=args)


def resolve_run_seeds(profile_data: Mapping[str, Any], fallback_seeds: Iterable[Any]) -> list:
    seeds = profile_data.get("seeds") if profile_data else None
    if seeds is None:
        return list(fallback_seeds)
    return list(seeds)
