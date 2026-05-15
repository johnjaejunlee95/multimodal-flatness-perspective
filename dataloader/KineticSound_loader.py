import os
import random
import copy
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.function_tools import build_dataloader_generator, build_numpy_rng, seed_worker

KINETIC_SAMPLE_RATE = 22050
KINETIC_DURATION_SECONDS = 10
KINETIC_NUM_MEL_BINS = 128
KINETIC_FRAME_SHIFT_MS = 10.0
KINETIC_TARGET_LENGTH = int(KINETIC_DURATION_SECONDS * 1000 / KINETIC_FRAME_SHIFT_MS)
KINETIC_STD_SCALE = 2.0


class KineticSoundLoader(object):
    def __init__(self, cfgs, split=True):
        super(KineticSoundLoader, self).__init__()
        self.cfgs = cfgs
        
        # Load data once and split properly
        full_dataset = KineticsSoundsDataset(cfgs, mode='full', indices=None)
        
        # Create proper train/validation split
        total_size = len(full_dataset.data)
        indices = build_numpy_rng(cfgs, stream="kinetic_split").permutation(total_size).tolist()
        
        train_size = int(total_size * 0.9)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        if not split:     
            train_indices = indices
            val_indices = indices
        
        
        # Create datasets with proper indices
        self.train_dataset = KineticsSoundsDataset(cfgs, mode='train', indices=train_indices)
        self.valid_dataset = KineticsSoundsDataset(cfgs, mode='valid', indices=val_indices)
        self.test_dataset = KineticsSoundsDataset(cfgs, mode='test', indices=None)

        if self.cfgs.modality in ["Audio", "Multimodal"]:
            print("Computing global audio normalization statistics from training set...")
            norm_mean, norm_std = self._compute_train_audio_stats(self.train_dataset)
            print("Computed global audio mean: {:.4f}, std: {:.4f}".format(norm_mean, norm_std))
            self.train_dataset.set_global_audio_stats(norm_mean, norm_std)
            self.valid_dataset.set_global_audio_stats(norm_mean, norm_std)
            self.test_dataset.set_global_audio_stats(norm_mean, norm_std)
        else:
            print("Audio modality not enabled, skipping global audio normalization statistics computation.")

    def _compute_train_audio_stats(self, dataset: "KineticsSoundsDataset"):
        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        with torch.no_grad():
            for class_name, av_file in dataset.data:
                audio_file = os.path.join(dataset.audio_path, class_name, av_file + '.wav')
                feature = dataset._extract_log_mel_feature(audio_file)
                total_sum += float(feature.sum().item())
                total_sq_sum += float((feature * feature).sum().item())
                total_count += int(feature.numel())

        if total_count == 0:
            raise RuntimeError("No training audio samples found for KineticSound normalization statistics.")

        mean = total_sum / total_count
        variance = max((total_sq_sum / total_count) - (mean * mean), 0.0)
        std = variance ** 0.5
        if std < 1e-12:
            std = 1.0
        return mean, std

    @property
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=True,
                          num_workers=self.cfgs.num_workers,
                          pin_memory=True,
                          worker_init_fn=seed_worker,
                          generator=build_dataloader_generator(self.cfgs, split="train"))
    
    @property
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=False,
                          num_workers=self.cfgs.num_workers,
                          pin_memory=True,
                          worker_init_fn=seed_worker,
                          generator=build_dataloader_generator(self.cfgs, split="valid"))
    
    @property
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=False,
                          num_workers=self.cfgs.num_workers,
                          pin_memory=True,
                          worker_init_fn=seed_worker,
                          generator=build_dataloader_generator(self.cfgs, split="test"))


class KineticsSoundsDataset(Dataset):
    def __init__(self, args, mode, indices):
        self.data = []
        self.label = []
        self.mode = mode
        self.indices = indices
        self.cfgs = args
        self.transform = args.transform
        self.used_frames = args.used_frames
        
        data_path = args.data_root
        
        # Handle different modes
        if mode == 'full':
            # Load train data for splitting
            self.csv_path = os.path.join(data_path, 'ks_train_real.txt')
            self.audio_path = os.path.join(data_path, 'train', 'audio')
            self.visual_path = os.path.join(data_path, 'train', 'video')
        elif mode in ['train', 'valid']:
            # For train/valid, we'll use train data but subset it
            self.csv_path = os.path.join(data_path, 'ks_train_real.txt')
            self.audio_path = os.path.join(data_path, 'train', 'audio')
            self.visual_path = os.path.join(data_path, 'train', 'video')
        else:  # test mode
            self.csv_path = os.path.join(data_path, f'ks_test_real.txt')
            self.audio_path = os.path.join(data_path, 'test', 'audio')
            self.visual_path = os.path.join(data_path, 'test', 'video')
        
        
        class_lists = os.listdir(self.audio_path)
        class_lists.sort()  # Ensure consistent order
        
        # Load all data first
        all_data = []
        all_labels = []
        
        with open(self.csv_path) as f:
            for line in f:
                item = line.split("\n")[0].split(",")
                class_name = item[0]
                file_name = item[-1][1:-4]                
                all_data.append([class_name, file_name])
                
                label_idx = class_lists.index(class_name)
                all_labels.append(label_idx)
        
        # Apply indices if provided (for train/valid split)
        if self.indices is not None:
            self.data = [all_data[i] for i in self.indices]
            self.label = [all_labels[i] for i in self.indices]
        else:
            self.data = all_data
            self.label = all_labels
            
        args.num_classes = len(set(self.label))

        self.label_to_indices = {}
        for idx, label in enumerate(self.label):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.target_sample_rate = KINETIC_SAMPLE_RATE
        self.num_mel_bins = KINETIC_NUM_MEL_BINS
        self.frame_shift = KINETIC_FRAME_SHIFT_MS
        self.target_length = KINETIC_TARGET_LENGTH
        self.norm_std_scale = KINETIC_STD_SCALE
        self.audio_log_offset = float(getattr(args, "audio_log_offset", 1e-6))
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

    def set_global_audio_stats(self, mean: float, std: float):
        self.global_audio_mean = float(mean)
        self.global_audio_std = float(std)

    def _extract_log_mel_feature(self, audio_file: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_file)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sample_rate)

        waveform = waveform.float()
        waveform = waveform - waveform.mean()

        mel_spec = self.mel_spectrogram(waveform)
        log_mel = torch.log(mel_spec.clamp(min=self.audio_log_offset))
        feature = log_mel.squeeze(0).transpose(0, 1).contiguous()  # [time, mel]

        pad = self.target_length - feature.size(0)
        if pad > 0:
            feature = torch.nn.functional.pad(feature, (0, 0, 0, pad))
        elif pad < 0:
            feature = feature[: self.target_length]
        return feature

    def _normalize_audio_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if self.global_audio_mean is None or self.global_audio_std is None:
            raise RuntimeError("Global audio statistics are not initialized for KineticSound dataset.")
        return (feature - self.global_audio_mean) / (self.global_audio_std * self.norm_std_scale + 1e-10)

    def _preprocess_audio(self, audio_file: str) -> torch.Tensor:
        feature = self._extract_log_mel_feature(audio_file)
        if self.cfgs.modality in ["Audio", "Multimodal"]:
            feature = self._normalize_audio_feature(feature)
        else:
            feature = feature
        return feature.transpose(0, 1).contiguous().unsqueeze(0).float()  # [1, mel, time]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        class_name, av_file = self.data[idx]
        label = self.label[idx]
        
        audio_file = os.path.join(self.audio_path, class_name, av_file + '.wav')
        spectrogram = self._preprocess_audio(audio_file)
   
   
        if self.mode == 'train' and self.transform:
            transform_list = []
            # Check for specific augmentations in self.transform
            if 'color_jitter' in self.transform:
                transform_list.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
            
            if 'cropping' in self.transform:
                transform_list.append(transforms.RandomResizedCrop(224))
            else:
                transform_list.append(transforms.Resize(size=(224, 224)))
            
            if 'horizontal_flip' in self.transform:
                transform_list.append(transforms.RandomHorizontalFlip())
                
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            transform = transforms.Compose(transform_list)

        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        
        if self.cfgs.shuffling and self.mode == 'train':
            # Get current sample's label            
            same_label_indices = self.label_to_indices[label]
            selected_idx = np.random.choice(same_label_indices, 1)[0]
            
            class_name, av_file = self.data[selected_idx]
        
        path = self.visual_path + f'/{class_name}/' + av_file
        files_list = [lists for lists in os.listdir(path)]
        file_num = len([fn for fn in files_list if fn.endswith("jpg")])
            
        # images = torch.permute(images, (1,0,2,3))
        seg = int(file_num / self.used_frames)
        path1 = []
        image = []
        image_arr = []
        t = [0] * self.used_frames
        for i in range(self.used_frames):
            if self.mode == 'train':
                t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
                if t[i] >= 10:
                    t[i] = 9
            else:
                t[i] = i*seg + max(int(seg/2), 1) if file_num > 6 else 1

            path1 = 'frame_' + str(t[i]) + '.jpg'
            image = Image.open(path + "/" + path1).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0)
            image_arr.append(image)            
            
            if i == 0:
                images = copy.copy(image_arr[i])
            else:
                images = torch.cat((images, image), 0)
                

        return spectrogram, images, torch.tensor(label).long()
