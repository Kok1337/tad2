import os
import cv2
import albumentations as A

import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class FoodDataset(Dataset):
    def __init__(self, data_type=None, transforms=None):
        if data_type is None:
            raise Exception("You are useless")
        self.path = 'data/Food-5K/' + data_type + '/'
        self.images_name = os.listdir(self.path)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images_name)
    
    def __getitem__(self, idx):
        data = self.images_name[idx]
        
        label = data.split('_')[0]
        label = int(label)
        label = torch.tensor(label)
        
        image = cv2.imread(self.path + data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            aug = self.transforms(image=image)
            image = aug['image']
        
        return (image, label)
        
        
image_transformation = A.Compose([
                            A.RandomResizedCrop(256, 256),
                            A.HorizontalFlip(),
                            A.Normalize(),
                            ToTensorV2()
                        ])
train_dataset = FoodDataset('training', image_transformation)
val_dataset = FoodDataset('validation', image_transformation)
test_dataset = FoodDataset('evaluation', image_transformation)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))