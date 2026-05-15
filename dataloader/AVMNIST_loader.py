import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.function_tools import build_dataloader_generator, build_numpy_rng, seed_worker

class AVMNISTLoader(object):
    def __init__(self, cfgs):
        super(AVMNISTLoader, self).__init__()
        self.cfgs = cfgs
        transform = cfgs.transform
        # Do the splitting once here
        total_size = len(np.load(os.path.join(cfgs.data_root, 'train_labels.npy')))
        indices = build_numpy_rng(cfgs, stream="avmnist_split").permutation(total_size)
        train_size = int(total_size * 0.9)
        
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:]
        
        self.train_dataset = AVMNIST(cfgs, mode='train', indices=self.train_indices)
        self.valid_dataset = AVMNIST(cfgs, mode='valid', indices=self.val_indices)
        self.test_dataset = AVMNIST(cfgs, mode='test')
        

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
                          

class AVMNIST(Dataset):
    def __init__(self, cfgs, mode = 'train', indices=None) -> None:
        super(AVMNIST,self).__init__()
        self.cfgs = cfgs
        self.mode = mode
        self.data_root = cfgs.data_root
        self.transform = cfgs.transform
        
        image_data_path = os.path.join(self.data_root,'image')
        audio_data_path = os.path.join(self.data_root,'audio')
        
        self.train_image = np.load(os.path.join(image_data_path,'train_data.npy'))
        self.train_audio = np.load(os.path.join(audio_data_path,'train_data.npy'))
        self.train_label = np.load(os.path.join(self.data_root,'train_labels.npy'))
     
        
        if mode == 'train' or mode == 'valid':
            self.image = self.train_image[indices]
            self.audio = self.train_audio[indices]
            self.label = self.train_label[indices]
                        
        elif mode == 'test':
            self.image = np.load(os.path.join(image_data_path,'test_data.npy'))
            self.audio = np.load(os.path.join(audio_data_path,'test_data.npy'))
            self.label = np.load(os.path.join(self.data_root,'test_labels.npy'))
        
        self.label_to_indices = {}
        for idx, label in enumerate(self.label):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
    def __getitem__(self, index):
        image = self.image[index]
        audio = self.audio[index]
        label = self.label[index]
        
        if self.cfgs.shuffling and self.mode == 'train':
            same_label_indices = self.label_to_indices[label]
            selected_idx = np.random.choice(same_label_indices, 1)
            audio = self.audio[selected_idx[0]]
            
        audio = audio / 255.0
        audio = np.expand_dims(audio,0)
        audio = torch.from_numpy(audio)
        if self.transform and self.mode == 'train':
            transform_list = []
            if 'color_jitter' in self.transform:
                transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1))
                
            if 'cropping' in self.transform:
                transform_list.append(transforms.RandomResizedCrop(28))
            else:
                transform_list.append(transforms.Resize(28))
                
            if 'horizontal_flip' in self.transform:
                transform_list.append(transforms.RandomHorizontalFlip())
                
            transform_list.append(transforms.ToTensor())
            transform = transforms.Compose(transform_list)
        else:
            transform = transforms.ToTensor()
            
        image = image.reshape(28,28)
        image = Image.fromarray(image.astype(np.uint8).squeeze(), mode='L')
        image = transform(image)
            
        return audio, image, label
    
    def __len__(self):
        self.length = len(self.image)
        return self.length
