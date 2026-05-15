import os
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.function_tools import build_dataloader_generator, derive_seed_from_cfg, seed_worker


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
    return classes, class_to_idx, idx_to_class


class Food101DataLoader(object):
    def __init__(self, cfgs):
        super(Food101DataLoader, self).__init__()
        self.cfgs = cfgs
        self.train_dataset = FOOD101Dataset(cfgs, mode="train")
        self.valid_dataset = FOOD101Dataset(cfgs, mode="valid")
        self.test_dataset = FOOD101Dataset(cfgs, mode="test")

    def _build_dataloader(self, dataset: Dataset, shuffle: bool, split: str) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.cfgs.batch_size,
            shuffle=shuffle,
            num_workers=self.cfgs.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=build_dataloader_generator(self.cfgs, split=split),
        )

    @property
    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, shuffle=True, split="train")

    @property
    def val_dataloader(self):
        return self._build_dataloader(self.valid_dataset, shuffle=False, split="valid")

    @property
    def test_dataloader(self):
        return self._build_dataloader(self.test_dataset, shuffle=False, split="test")


class FOOD101Dataset(Dataset):
    def __init__(self, cfgs, mode: str = "train"):
        super().__init__()
        if mode not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.cfgs = cfgs
        self.mode = mode
        self.transform_cfg = getattr(cfgs, "transform", None)
        self.split_seed = derive_seed_from_cfg(cfgs, stream="food_split")
        self.enable_same_label_shuffle = bool(getattr(cfgs, "shuffling", False)) and mode == "train"

        phase = "train" if mode == "train" else "test"
        self.dataset_root = cfgs.data_root
        self.csv_file_path = os.path.join(self.dataset_root, "texts", f"{phase}_titles.csv")
        self.img_dir = os.path.join(self.dataset_root, "images", phase)

        full_data = pd.read_csv(self.csv_file_path)
        self.data = self._select_split(full_data).reset_index(drop=True)

        self.classes, self.class_to_idx, self.idx_to_class = find_classes(self.img_dir)
        self.image_names = self.data.iloc[:, 0].astype(str).tolist()
        self.texts = self.data.iloc[:, 1].astype(str).tolist()
        self.label_names = self.data.iloc[:, -1].astype(str).tolist()
        self.labels = [self.class_to_idx[label_name] for label_name in self.label_names]

        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.visual_transform = self._build_visual_transform()

    def _select_split(self, full_data: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "train":
            return full_data

        train_idx, val_idx = train_test_split(
            np.arange(len(full_data)),
            test_size=0.1,
            random_state=self.split_seed,
            stratify=full_data.iloc[:, -1],
        )
        if self.mode == "test":
            return full_data.iloc[train_idx]
        return full_data.iloc[val_idx]

    def _parse_transform_flags(self) -> Set[str]:
        if not self.transform_cfg:
            return set()

        if isinstance(self.transform_cfg, str):
            tokens = self.transform_cfg.replace(",", " ").split()
            return {token.strip() for token in tokens if token.strip()}

        if isinstance(self.transform_cfg, Sequence):
            return {str(token).strip() for token in self.transform_cfg if str(token).strip()}

        return set()

    def _build_visual_transform(self):
        if self.mode == "train" and self.transform_cfg:
            flags = self._parse_transform_flags()
            transform_list = []

            if "color_jitter" in flags:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    )
                )
            if "cropping" in flags:
                transform_list.append(transforms.RandomResizedCrop(224))
            else:
                transform_list.append(transforms.Resize(size=(224, 224)))
            if "horizontal_flip" in flags:
                transform_list.append(transforms.RandomHorizontalFlip())

            transform_list.append(transforms.ToTensor())
            transform_list.append(
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            )
            return transforms.Compose(transform_list)

        return transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _clean_text(self, raw_text: str) -> str:
        text = re.sub(r"^RT[\s]+", "", raw_text)
        text = re.sub(r"https?:\/\/.*[\r\n]*", "", text)
        text = re.sub(r"#", "", text)
        return text

    def _sample_text_index(self, index: int, label: int) -> int:
        if not self.enable_same_label_shuffle:
            return index

        same_label_indices = self.label_to_indices[label]
        if len(same_label_indices) <= 1:
            return index

        candidate_indices = [idx for idx in same_label_indices if idx != index]
        if not candidate_indices:
            return index

        return int(np.random.choice(candidate_indices))

    def _load_image(self, index: int):
        img_path = os.path.join(self.img_dir, self.label_names[index], self.image_names[index])
        with Image.open(img_path) as img:
            return img.convert("RGB")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.labels[index]
        text_index = self._sample_text_index(index, label)
        text = self._clean_text(self.texts[text_index])

        image = self._load_image(index)
        image = self.visual_transform(image)

        return image, text, label
