import os
import json

import torch
import monai
from monai.data import (
    DataLoader,
    CacheDataset,
    list_data_collate,
)

from monai import transforms as T


def load_datalist_cross_validation(data_path=None, dataframe=None, fold=None):
    train_subjects = []
    val_subjects = []
    for image, label_segmentation, label_classification, fold_x in zip(dataframe['image'], dataframe['label_segmentation'], dataframe['label_classification'], dataframe['fold']):
        subject = {
        'image': [os.path.join(data_path, image_x) for image_x in image],
        'label_segmentation': os.path.join(data_path, label_segmentation),
        'label_classification': label_classification,
        }
        if fold_x == fold:
            val_subjects.append(subject)
        else:
            train_subjects.append(subject)

    return train_subjects, val_subjects

def dataloader_cross_validation(train_subjects, train_transform, val_subjects, val_transform, batch_size=1, num_workers=4):
    train_dataset = CacheDataset(data=train_subjects, transform=train_transform, cache_num=24, cache_rate=1, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=list_data_collate)
    
    val_dataset = CacheDataset(data=val_subjects, transform=val_transform, cache_num=24, cache_rate=1, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=list_data_collate)
    
    return train_loader, val_loader

def rsna_miccai_radiogenomics_cross_validation(dataframe=None, fold=None):
    data_path = '/path/to/brats21/'
    train_subjects, val_subjects = load_datalist_cross_validation(data_path=data_path, dataframe=dataframe, fold=fold)

    train_transform = T.Compose([
            T.LoadImaged(keys=["image", "label_segmentation"]),
            T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label_segmentation"),
            T.CropForegroundd(keys=["image", "label_segmentation"], source_key="image", k_divisible=[128, 128, 128]),
            T.RandSpatialCropd(keys=["image", "label_segmentation"], roi_size=[128, 128, 128], random_size=False),
            T.RandFlipd(keys=["image", "label_segmentation"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "label_segmentation"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "label_segmentation"], prob=0.5, spatial_axis=2),
            T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            T.ToTensord(keys=["image", "label_segmentation", "label_classification"]),
        ])
    val_transform = T.Compose([
            T.LoadImaged(keys=["image", "label_segmentation"]),
            T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label_segmentation"),
            T.CropForegroundd(keys=["image", "label_segmentation"], source_key="image", k_divisible=[128, 128, 128]),
            T.RandSpatialCropd(keys=["image", "label_segmentation"], roi_size=[128, 128, 128], random_size=False),
            T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            T.ToTensord(keys=["image", "label_segmentation", "label_classification"]),
        ])

    return train_subjects, train_transform, val_subjects, val_transform

if __name__ == "__main__":
    flag = 0
