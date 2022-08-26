import os
import json
import math
import numpy as np
import nibabel as nib
from functools import partial

import torch
import monai
from monai import transforms, data


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0,high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist,
                  basedir,
                  fold=0,
                  key='training'):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr=[]
    val=[]
    for d in json_data:
        if 'fold' in d and d['fold'] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

def get_loader(data_dir, datalist_json, fold, test_mode):
    distributed = False
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=[128, 128, 128]),
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if test_mode:
        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if distributed else None
        test_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, sampler=val_sampler, pin_memory=True,)
        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)
        train_sampler = Sampler(train_ds) if distributed else None
        train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=(train_sampler is None), num_workers=2, sampler=train_sampler, pin_memory=True,)
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, sampler=val_sampler, pin_memory=True,)
        loader = [train_loader, val_loader]

    return loader

def get_loader_name(data_dir, datalist_json, fold, test_mode, name):
    distributed = False
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=[128, 128, 128]),
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if test_mode:
        for i in validation files:
            if i == name:
                validation_files = validations_files[]
                val_ds = data.Dataset(data=validation_files, transform=test_transform)
                val_sampler = Sampler(val_ds, shuffle=False) if distributed else None
                test_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, sampler=val_sampler, pin_memory=True,)
                loader = test_loader
            break
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)
        train_sampler = Sampler(train_ds) if distributed else None
        train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=(train_sampler is None), num_workers=2, sampler=train_sampler, pin_memory=True,)
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, sampler=val_sampler, pin_memory=True,)
        loader = [train_loader, val_loader]

    return loader

def generate_segmentations(pretrained_path, fold):
    test_mode = True
    output_directory = '/path/to/project_neuroradiology/outputs'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    test_loader = get_loader(data_dir='/path/to/datasets/brats21/', datalist_json='/path/to/brats21_folds.json', fold=fold, test_mode=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = pretrained_path
    model = monai.networks.nets.SwinUNETR(img_size=128,
                      in_channels=4,
                      out_channels=3,
                      feature_size=48,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      dropout_path_rate=0.0,
                      use_checkpoint=True,
                      )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    model_inferer_test = partial(
        monai.inferers.sliding_window_inference,
        roi_size=[128, 128, 128],
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].cuda()
            affine = batch['image_meta_dict']['original_affine'][0].numpy()
            num = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('_')[1]
            img_name = 'BraTS2021_' + num + '.nii.gz'
            print("Inference on case {}".format(img_name))
            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))
        print("Finished inference!")
        
 def generate_segmentations_name(pretrained_path, fold, name):
    test_mode = True
    output_directory = '/path/to/project_neuroradiology/outputs'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    test_loader = get_loader_name(data_dir='/path/to/datasets/brats21/', datalist_json='/path/to/brats21_folds.json', fold=fold, test_mode=True, name=name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = pretrained_path
    model = monai.networks.nets.SwinUNETR(img_size=128,
                      in_channels=4,
                      out_channels=3,
                      feature_size=48,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      dropout_path_rate=0.0,
                      use_checkpoint=True,
                      )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    model_inferer_test = partial(
        monai.inferers.sliding_window_inference,
        roi_size=[128, 128, 128],
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].cuda()
            affine = batch['image_meta_dict']['original_affine'][0].numpy()
            num = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('_')[1]
            img_name = 'BraTS2021_' + num + '.nii.gz'
            print("Inference on case {}".format(img_name))
            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))
        print("Finished inference!")

if __name__ == "__main__":
    generate_segmentations(pretrained_path='/path/to/weights/fold0_f48_ep300_4gpu_dice0_8854/model.pt', fold=0)
    generate_segmentations(pretrained_path='/path/to/weights/fold1_f48_ep300_4gpu_dice0_9059/model.pt', fold=1)
    generate_segmentations(pretrained_path='/path/to/weights/fold2_f48_ep300_4gpu_dice0_8981/model.pt', fold=2)
    generate_segmentations(pretrained_path='/path/to/weights/fold3_f48_ep300_4gpu_dice0_8924/model.pt', fold=3)
    generate_segmentations(pretrained_path='/path/to/weights/fold4_f48_ep300_4gpu_dice0_9035/model.pt', fold=4)
    
    generate_segmentations_name(pretrained_path='/path/to/weights/fold4_f48_ep300_4gpu_dice0_9035/model.pt', fold=4, name='00010')
    
    
