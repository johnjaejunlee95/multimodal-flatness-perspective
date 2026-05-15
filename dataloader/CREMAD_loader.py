import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.function_tools import build_dataloader_generator, build_numpy_rng, seed_worker

CREMAD_SAMPLE_RATE = 22050
CREMAD_DURATION_SECONDS = 3
CREMAD_KALDI_NUM_MEL_BINS = 128
CREMAD_KALDI_FRAME_SHIFT_MS = 10.0
CREMAD_KALDI_TARGET_LENGTH = int(CREMAD_DURATION_SECONDS * 1000 / CREMAD_KALDI_FRAME_SHIFT_MS)
CREMAD_KALDI_STD_SCALE = 2.0


class CREMAD_Dataloder(object):
    """Factory wrapper that provides train/valid/test dataloaders for CREMA-D."""

    def __init__(self, cfgs):
        self.cfgs = cfgs
        train_data = CREMADDatasets.load_split_data(
            data_root=cfgs.data_root,
            csv_path=os.path.join(cfgs.data_root, "train.csv"),
        )
        total_size = len(train_data[0])
        if total_size < 2:
            raise RuntimeError("CREMAD train split must contain at least 2 samples for 9:1 train/valid split.")

        indices = build_numpy_rng(cfgs, stream="cremad_split").permutation(total_size).tolist()
        valid_size = max(1, int(total_size * 0.1))
        train_size = total_size - valid_size
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]

        self.train_dataset = CREMADDatasets(
            cfgs,
            mode="train",
            train_data=train_data,
            split_indices=train_indices,
        )
        self.valid_dataset = CREMADDatasets(
            cfgs,
            mode="valid",
            train_data=train_data,
            split_indices=valid_indices,
        )
        self.test_dataset = CREMADDatasets(cfgs, mode="test")

        if self.cfgs.modality in ["Audio", "Multimodal"]:
            print("Computing global audio normalization statistics from training set...")
            norm_mean, norm_std = self._compute_train_audio_stats(self.train_dataset)
            print("Computed global audio mean: {:.4f}, std: {:.4f}".format(norm_mean, norm_std))
            self.train_dataset.set_global_audio_stats(norm_mean, norm_std)
            self.valid_dataset.set_global_audio_stats(norm_mean, norm_std)
            self.test_dataset.set_global_audio_stats(norm_mean, norm_std)
        else:
            print("Audio modality not enabled, skipping global audio normalization statistics computation.")

    def _compute_train_audio_stats(self, dataset: "CREMADDatasets") -> Tuple[float, float]:
        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        with torch.no_grad():
            for audio_path in dataset.audio:
                feature = dataset._extract_log_mel_feature(audio_path)
                total_sum += float(feature.sum().item())
                total_sq_sum += float((feature * feature).sum().item())
                total_count += int(feature.numel())

        if total_count == 0:
            raise RuntimeError("No training audio samples found for CREMAD normalization statistics.")

        mean = total_sum / total_count
        variance = max((total_sq_sum / total_count) - (mean * mean), 0.0)
        std = variance ** 0.5
        if std < 1e-12:
            std = 1.0
        return mean, std

    def _build_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        split = "train" if shuffle else ("valid" if dataset is self.valid_dataset else "test")
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
        return self._build_dataloader(self.train_dataset, shuffle=True)

    @property
    def valid_dataloader(self):
        return self._build_dataloader(self.valid_dataset, shuffle=False)

    @property
    def test_dataloader(self):
        return self._build_dataloader(self.test_dataset, shuffle=False)


class CREMADDatasets(Dataset):
    """CREMA-D multimodal dataset (audio log-Mel spectrogram + one sampled visual frame)."""

    CLASS_TO_INDEX: Dict[str, int] = {
        "NEU": 0,
        "HAP": 1,
        "SAD": 2,
        "FEA": 3,
        "DIS": 4,
        "ANG": 5,
    }

    def __init__(
        self,
        cfgs,
        mode: str = "train",
        train_data: Tuple[List[str], List[str], List[int]] = None,
        split_indices: List[int] = None,
    ):
        super().__init__()

        if mode not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.cfgs = cfgs
        self.mode = mode
        self.transform = getattr(cfgs, "transform", None)
        self.data_root = cfgs.data_root

        self.visual_feature_path = self.data_root
        self.audio_feature_path = os.path.join(self.data_root, "Audio")
        self.train_csv = os.path.join(self.data_root, "train.csv")
        self.test_csv = os.path.join(self.data_root, "test.csv")

        self.target_sample_rate = CREMAD_SAMPLE_RATE

        self.num_mel_bins = CREMAD_KALDI_NUM_MEL_BINS
        self.frame_shift = CREMAD_KALDI_FRAME_SHIFT_MS
        self.target_length = CREMAD_KALDI_TARGET_LENGTH
        self.norm_std_scale = CREMAD_KALDI_STD_SCALE
        self.log_offset = float(getattr(cfgs, "audio_log_offset", 1e-6))
        self.global_audio_mean = None
        self.global_audio_std = None

        self.win_length = int(round(0.025 * self.target_sample_rate))
        self.hop_length = int(round((self.frame_shift / 1000.0) * self.target_sample_rate))
        self.n_fft = 1024
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=0.0,
            f_max=self.target_sample_rate / 2,
            n_mels=self.num_mel_bins,
            power=2.0,
            center=True,
            pad_mode="reflect",
            mel_scale="htk",
        )

        if train_data is None:
            self.train_image, self.train_audio, self.train_label = self.load_split_data(
                self.data_root,
                self.train_csv,
            )
        else:
            self.train_image, self.train_audio, self.train_label = train_data

        self.test_image, self.test_audio, self.test_label = self.load_split_data(
            self.data_root,
            self.test_csv,
        )

        if self.mode in {"train", "valid"}:
            if split_indices is None:
                raise ValueError(f"split_indices must be provided for mode={self.mode}")
            self.image = [self.train_image[i] for i in split_indices]
            self.audio = [self.train_audio[i] for i in split_indices]
            self.label = [self.train_label[i] for i in split_indices]
        else:
            self.image = self.test_image
            self.audio = self.test_audio
            self.label = self.test_label

        self.enable_visual_shuffle = bool(getattr(cfgs, "shuffling", False)) and self.mode == "train"
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.label):
            self.label_to_indices[label].append(idx)

        self.visual_transform = self._build_visual_transform()

    def _parse_transform_flags(self) -> Set[str]:
        if not self.transform:
            return set()

        if isinstance(self.transform, str):
            tokens = self.transform.replace(",", " ").split()
            return {token.strip() for token in tokens if token.strip()}

        if isinstance(self.transform, Sequence):
            return {str(token).strip() for token in self.transform if str(token).strip()}

        return set()

    def _build_visual_transform(self):
        if self.mode == "train" and self.transform:
            transform_flags = self._parse_transform_flags()
            transform_list = []

            if "color_jitter" in transform_flags:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    )
                )

            if "cropping" in transform_flags:
                transform_list.append(transforms.RandomResizedCrop(224))
            else:
                transform_list.append(transforms.Resize(size=(224, 224)))

            if "horizontal_flip" in transform_flags:
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

    @classmethod
    def load_split_data(cls, data_root: str, csv_path: str) -> Tuple[List[str], List[str], List[int]]:
        image_dirs: List[str] = []
        audio_paths: List[str] = []
        labels: List[int] = []
        audio_feature_path = os.path.join(data_root, "Audio")
        visual_feature_path = data_root

        with open(csv_path, encoding="UTF-8-sig") as csv_file:
            for row in csv.reader(csv_file):
                if len(row) < 2:
                    continue
                item_id = row[0]
                class_name = row[1]
                if class_name not in cls.CLASS_TO_INDEX:
                    continue

                audio_path = os.path.join(audio_feature_path, f"{item_id}.wav")
                visual_path = os.path.join(visual_feature_path, "Image-01-FPS", item_id)

                if not (os.path.exists(audio_path) and os.path.exists(visual_path)):
                    continue

                image_dirs.append(visual_path)
                audio_paths.append(audio_path)
                labels.append(cls.CLASS_TO_INDEX[class_name])

        return image_dirs, audio_paths, labels

    def set_global_audio_stats(self, mean: float, std: float):
        self.global_audio_mean = float(mean)
        self.global_audio_std = float(std)

    def _extract_log_mel_feature(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sample_rate)

        waveform = waveform.float()
        waveform = waveform - waveform.mean()
        mel_spec = self.mel_spectrogram(waveform)
        log_mel = torch.log(mel_spec.clamp(min=self.log_offset))
        feature = log_mel.squeeze(0).transpose(0, 1).contiguous()  # [time, mel]

        pad = self.target_length - feature.size(0)
        if pad > 0:
            feature = torch.nn.functional.pad(feature, (0, 0, 0, pad))
        elif pad < 0:
            feature = feature[: self.target_length]
        return feature

    def _normalize_audio_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if self.global_audio_mean is None or self.global_audio_std is None:
            raise RuntimeError("Global audio statistics are not initialized for CREMAD dataset.")
        return (feature - self.global_audio_mean) / (self.global_audio_std * self.norm_std_scale + 1e-10)

    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        feature = self._extract_log_mel_feature(audio_path)
        if self.cfgs.modality in ["Audio", "Multimodal"]:
            return self._normalize_audio_feature(feature)
        else:
            return feature

    def _load_one_image(self, image_dir: str) -> torch.Tensor:
        image_samples = sorted(os.listdir(image_dir))
        if not image_samples:
            raise RuntimeError(f"No image frames found in: {image_dir}")
        
        # candidate_frames = image_samples[1:] if (len(image_samples) > 1) else image_samples
        selected_frame = str(np.random.choice(image_samples))
        image_path = os.path.join(image_dir, selected_frame)

        with Image.open(image_path) as image:
            image_tensor = self.visual_transform(image.convert("RGB"))

        return image_tensor.unsqueeze(0)

    def _sample_visual_index(self, index: int, label: int) -> int:
        if not self.enable_visual_shuffle:
            return index

        same_label_indices = self.label_to_indices[label]
        if len(same_label_indices) <= 1:
            return index

        candidate_indices = [idx for idx in same_label_indices if idx != index]
        if not candidate_indices:
            return index

        return int(np.random.choice(candidate_indices))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        spectrogram = self._preprocess_audio(self.audio[index])
        label = self.label[index]
        visual_index = self._sample_visual_index(index, label)
        images = self._load_one_image(self.image[visual_index])

        return spectrogram, images, label
