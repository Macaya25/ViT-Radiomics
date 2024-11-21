# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:38:14 2023

@author: Mico
"""


import os
import numpy as np

from skimage.transform import resize
from scipy.ndimage import rotate
from skimage.color import gray2rgb
import torch
import tensorflow_datasets as tfds
from MedSAM.segment_anything import sam_model_registry
from tqdm import tqdm
import pandas as pd
import h5py
import argparse
from transformers import AutoModel
from visualization_utils import (crop_image,
                                 extract_coords,
                                 extract_roi,
                                 visualize_features,
                                 hu_to_rgb_vectorized)

from all_medsams import medsam_normvit_neck, LoRA_sam

def prepare_image(img, model_name='medsam'):
    """ Resize image and convert to torch cuda tensor

    Args:
        img (np.array): image of a slice with shape (h, w, ch) in a range (0-1).

    Returns:
        img_tensor (torch.tensor): image as cuda tensor with shape (batch, h, w, ch).

    """
    if model_name in ['medsam','medsam_normvit','medsam_lora']:
        img = gray2rgb(img)
        img_tensor = resize(img, (1024, 1024))
        
    elif model_name == 'smdino':
        mu = 0.5
        std = 0.5
        img_tensor = resize(img, (224*3, 224*3))
        img_tensor = np.expand_dims(img_tensor, axis=-1)
        img_tensor = (img_tensor - mu) / std
    elif model_name in ['dinov2','rad_dino']:
        mu = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_tensor = resize(img, (224*3, 224*3))
        img_tensor = gray2rgb(img_tensor)
        img_tensor = (img_tensor - mu) / std
    img_tensor = img_tensor.transpose((2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = torch.as_tensor(img_tensor, dtype=torch.float32).cuda()
    return img_tensor


def load_model(model_name, model_path=None):
    """ Load a ViT model as image encoder

    Args:
        model_name (str): 'medsam', 'dinov2' or 'smdino'.
        model_path (str, optional): path to the .pth file. Defaults to None.

    Returns:
        model (torch.nn.Module): loaded torch model.

    """
    print(f"Using {model_name} as backbone\n")
    if model_name == 'smdino':
        from smdino.medical_sdino import load_sdino
        model = load_sdino(model_path)
        device = torch.cuda.current_device()
        model.to(device)
    if model_name == 'dinov2':
        model = load_dinov2()
    elif model_name == 'rad_dino':
        model = load_rad_dino()
    elif model_name == 'medsam':
        model = load_medsam(model_path)
    elif model_name == 'medsam_normvit':
        model = load_medsam_normvit(model_path)
    elif model_name == 'medsam_lora':
        model = load_medsam_lora(model_path)
    """
    elif model_name == 'medsam_mae':
        model = load_medsam(model_path)
    """
    model.model_name = model_name
    return model


def load_dinov2(backbone_size='base'):
    """ Load dinov2 ViT model from torch.hub

    Args:
        backbone_size (str, optional): size of ViT backbone. Defaults to 'base'.

    Returns:
        model (torch.nn.Module): loaded torch model.

    """
    backbone_archs = {"small": "vits14",
                      "base": "vitb14",
                      "large": "vitl14",
                      "giant": "vitg14"}

    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}_reg"
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    model.eval()
    return model


def load_rad_dino():
    """ Load dinov2 ViT model from torch.hub

    Args:
        backbone_size (str, optional): size of ViT backbone. Defaults to 'base'.

    Returns:
        model (torch.nn.Module): loaded torch model.

    """
    repo = "microsoft/rad-dino"
    model = AutoModel.from_pretrained(repo)
    model.eval()
    return model


def load_medsam(model_path):
    """ Load medsam ViT model from a .pth file

    Args:
        model_path (str): path to the .pth file.

    Returns:
        model (torch.nn.Module): loaded torch model.

    """
    device = torch.cuda.current_device()
    model = sam_model_registry['vit_b'](model_path)
    model = model.image_encoder.to(device)
    return model


def load_medsam_normvit(model_path):
    """ Load medsam norm ViT model from a .pth file

    Args:
        model_path (str): path to the .pth file.

    Returns:
        model (torch.nn.Module): loaded torch model.

    """
    sam_model = sam_model_registry["vit_b"](model_path)
    model = medsam_normvit_neck(
        image_encoder=sam_model.image_encoder,
    )
    return model

def load_medsam_lora(model_path):
    """ Load medsam LoRA model from a .pth file

    Args:
        model_path (str): path to the .pth file.

    Returns:
        model (torch.nn.Module): loaded torch model.

    """
    sam_model = sam_model_registry["vit_b"](model_path)
    lora_sam = LoRA_sam(sam_model, rank = 128)
    model = lora_sam.sam.image_encoder
    return model

def get_dense_descriptor(model, img):
    """ Use medsam or DinoV2 to extract patch embeddings

    Args:
        model (torch.nn.Module): ViT image encoder
        img (np.array): image of an slice with shape (N, M, CH).

    Returns:
        features (np.array): slice feature maps with shape (N//patch_size, M//patch_size, feature_dim).

    """
    img_tensor = prepare_image(img, model_name=model.model_name)
    
    if model.model_name=='medsam':
        with torch.no_grad():
            features_tensor = model(img_tensor)
            features = features_tensor.cpu().detach().numpy()
            features = np.squeeze(features)
            features = np.transpose(features, (1, 2, 0))

    elif model.model_name in ['medsam_normvit','medsam_lora']:
        features_tensor = model(img_tensor)
        features = features_tensor.cpu().detach().numpy()
        features = np.squeeze(features)
        features = np.transpose(features, (1, 2, 0))
        
    elif model.model_name == 'dinov2': # Dino
        with torch.no_grad():
            features_dict = model.forward_features(img_tensor)
            features_tensor = features_dict['x_norm_patchtokens']
            features = features_tensor.cpu().detach().numpy()
            del features_dict
            features = np.squeeze(features)
            featmap_size = int(np.sqrt(features.shape[0]))
            features = features.reshape(featmap_size, featmap_size, features.shape[1])
            
    elif model.model_name == 'rad_dino':
        with torch.no_grad():
            features_dict = model(img_tensor,return_dict=True)
            features_tensor = features_dict["last_hidden_state"]
            features = features_tensor.cpu().detach().numpy()
            del features_dict
            features = np.squeeze(features)[1:]
            featmap_size = int(np.sqrt(features.shape[0]))
            features = features.reshape(featmap_size, featmap_size, features.shape[1])

    del img_tensor
    del features_tensor
    torch.cuda.empty_cache()

    return features


def save_features(filename, all_features, all_masks, patient_id):
    """ Save features and mask in a .hdf5 dataset file

    Args:
        filename (str): dataset.hdf5 path.
        all_features (list(np.array)): list of feature maps of each slice.
        all_masks (list(np.array)): list of nodule maks of each slice.
        patient_id (str): id of the patient.

    """
    with h5py.File(filename, 'a') as h5f:
        if patient_id in h5f:
            print(f'features for {patient_id} already exists')
            del h5f[patient_id]
        patient_group = h5f.create_group(patient_id)
        for i, (feature, mask) in enumerate(zip(all_features, all_masks)):
            patient_group.create_dataset(f'features/{i}',
                                         compression="lzf",
                                         data=feature,
                                         chunks=feature.shape)
            patient_group.create_dataset(f'masks/{i}',
                                         compression="lzf",
                                         data=mask,
                                         chunks=mask.shape)


def tfds2voxels(ds, patient_id, pet=False):
    """ Get and stack patient slices into volumetrics arrays

    Args:
        ds (tf.data.Dataset): tfds dataset of a specific modality.
        patient_id (str): id of the patient.
        pet (bool, optional): pet images are divided by the mean of the liver pet. Defaults to False.

    Returns:
        img (np.array): CT or PET 3D data with shape (H, W, slices).
        mask (np.array): 3D nodule mask with shape (H, W, slices).
        label (np.array): nodule EGFR class 0: wildtype, 1: mutant, 2: unknown 3: not collected.
        spatial_res (np.array(tuple)): (x, y, z) spatial resolution of the exam.

    """
    img = []
    mask = []
    label = []
    for sample in ds[patient_id]:
        pet_liver_mean = 1
        if pet:
            pet_liver = sample['pet_liver'].numpy()
            pet_liver_mean = pet_liver[pet_liver != 0].mean() + 1e-10
        img += [sample['img_exam'].numpy() / pet_liver_mean]
        mask += [sample['mask_exam'].numpy()]
        label += [sample['egfr_label'].numpy()]
    img = np.dstack(img)
    mask = np.dstack(mask)
    spatial_res = sample['exam_metadata']['space_directions'].numpy()
    spatial_res = np.abs(spatial_res)
    if spatial_res.min() <= 0:
        spatial_res = np.repeat(spatial_res.max(), spatial_res.shape)
        print(f'\nWarning: {patient_id} has null voxel resolution')
    return img, mask, label, spatial_res


def windowing_ct(width, level):
    """Generate CT bounds

    Args:
        width (int): window width in the HU scale.
        level (int): center of the windows in the HU scale.

    Returns:
        lower_bound (int): lower CT value.
        upper_bound (int): upper CT value.

    reference values:
    chest
    - lungs W:1500 L:-600
    - mediastinum W:350 L:50

    abdomen
    - soft tissues W:400 L:50
    - liver W:150 L:30

    spine
    - soft tissues W:250 L:50
    - bone W:1800 L:400

    head and neck
    - brain W:80 L:40
    - subdural W:130-300 L:50-100
    - stroke W:8 L:32 or W:40 L:40
    - temporal bones W:2800 L:600 or W:4000 L:700
    - soft tissues: W:350–400 L:20–60

    source: https://radiopaedia.org/articles/windowing-ct
    """
    lower_bound = level - width/2
    upper_bound = level + width/2
    return lower_bound, upper_bound


def generate_features(model, img_3d, mask_3d, display=False):
    """ Extract feature map of each slice and crop them to focus on nodule region.

    Args:
        model (torch.nn.Module): ViT image encoder
        img_3d (np.array): CT or PET 3D data with shape (H, W, slices, Ch).
        mask_3d (np.array): 3D nodule boolean mask with shape (H, W, slices).
        display (bool, optional): To visualize images and extracted features. Defaults to False.

    Returns:
        features_list (List(np.array)): featuremap of each slice cropped to the nodule region.
        mask_list (List(np.array)):  binary mask of each slice cropped to the nodule region.

    """
    bigger_mask = np.sum(mask_3d, axis=-1) > 0

    h, w = bigger_mask.shape
    xmin, ymin, xmax, ymax = extract_coords(bigger_mask, margin=2)
    crop_size = max(xmax-xmin, ymax-ymin)*2
    xmid, ymid = int(xmin + (xmax-xmin)/2), int(ymin + (ymax-ymin)/2)
    xmin, ymin, xmax, ymax = xmid-crop_size, ymid-crop_size, xmid+crop_size, ymid+crop_size

    img_3d = crop_image(img_3d, xmin, ymin, xmax, ymax)
    mask_3d = crop_image(mask_3d, xmin, ymin, xmax, ymax)
    bigger_mask = crop_image(bigger_mask, xmin, ymin, xmax, ymax)

    features_list = []
    mask_list = []
    for slice_i in range(0, img_3d.shape[2]):
        mask = mask_3d[:, :, slice_i] > 0
        if mask.sum() < 1:
            continue
        img = img_3d[:, :, slice_i]
        features = get_dense_descriptor(model, img)
        crop_features = extract_roi(features, bigger_mask)
        crop_mask = extract_roi(mask, bigger_mask)
        features_list.append(crop_features)
        mask_list.append(crop_mask)
        if display:
            visualize_features(img, features, mask)
    return features_list, mask_list


def apply_window_ct(ct, width, level):
    """ Normalize CT image using a window in the HU scale

    Args:
        ct (np.array): ct image.
        width (int): window width in the HU scale.
        level (int): center of the windows in the HU scale.

    Returns:
        ct (np.array): Normalized image in a range 0-1.

    """
    ct_min_val, ct_max_val = windowing_ct(width, level)
    ct_range = ct_max_val - ct_min_val
    ct = (ct - ct_min_val) / ct_range
    ct = np.clip(ct, 0, 1)
    return ct


def flip_image(image, mask, flip_type):
    """ Flip a 3D image and mask horizontal or vertically

    Args:
        image (np.array): CT or PET 3D data with shape (H, W, slices, Ch).
        mask (np.array): 3D nodule boolean mask with shape (H, W, slices).
        flip_type (str): None, 'horizontal' or 'vertical'.

    Returns:
        image_flip (np.array): flipped 3D image.
        mask_flip (np.array): flipped 3D mask.

    """
    image_flip = image.copy()
    mask_flip = mask.copy()
    if flip_type == 'horizontal':
        return image_flip[:, ::-1, ...], mask_flip[:, ::-1, ...]
    elif flip_type == 'vertical':
        return image_flip[::-1, ...], mask_flip[::-1, ...]
    return image_flip, mask_flip


def rotate_image(image, mask, angle, axes=(0, 1)):
    """ Rotates a 3D image and mask in the plane XY

    Args:
        image (np.array): CT or PET 3D data with shape (H, W, slices, Ch).
        mask (np.array): 3D nodule boolean mask with shape (H, W, slices).
        angle (int): rotation angle in degrees.
        axes (tuple, optional): rotation axis. Defaults to (0, 1).

    Returns:
        image_rot (TYPE): rotated 3D image.
        mask_rot (TYPE): rotated 3D mask.

    """
    image_rot = image.copy()
    mask_rot = mask.copy()
    if angle == 0:
        return image_rot, mask_rot
    image_rot = rotate(image_rot, angle, axes=axes, reshape=False, mode='nearest')
    image_rot = np.clip(image_rot, 0, 1)
    mask_rot = rotate(mask_rot, angle, axes=axes, reshape=False, mode='nearest')
    mask_rot = mask_rot > 0
    return image_rot, mask_rot


def get_voxels(hdf5_path, patient_id, modality):
    with h5py.File(hdf5_path, 'r') as h5f:
        pet_liver_mean = 1
        idm = f'{patient_id}_{modality}'
        is_pet = modality == 'pet'

        if is_pet:
            pet_liver = h5f[f'{idm}/pet_liver'][()]
            pet_liver_mean = pet_liver[pet_liver != 0].mean() + 1e-10
        slices = [int(k) for k in h5f[f'{idm}/img_exam'].keys()]
        slices.sort()
        img = np.dstack([h5f[f'{idm}/img_exam/{k}'][()] for k in slices])
        mask = np.dstack([h5f[f'{idm}/mask_exam/{k}'][()] for k in slices])
        spatial_res = np.abs(h5f[f'{idm}/spatial_res'][()])
        label = np.abs(h5f[f'{idm}/egfr_label'][()])

    if is_pet:
        img = img / pet_liver_mean

    return img, mask, label, spatial_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obtener ViT patch embeddings de los dataset lung_radiomics")

    parser.add_argument("-mn", "--model_name", type=str, default="smdino",
                        help="backbone ViT encoder medsam o smdino")
    parser.add_argument("-mp", "--model_path", type=str,
                        default=os.path.join('models', 'backbones', 'smdino', 'model.pth'),
                        help="path del archivo model.pth")
    parser.add_argument("-d", "--dataset_path", type=str, default=os.path.join('data', 'lung_radiomics'),
                        help="path de los datasets tfds santa_maria y stanford")
    parser.add_argument("-f", "--feature_folder", type=str, default=os.path.join('data', 'features'),
                        help="carpeta de salida donde se guardaran los features")
    parser.add_argument("-h5", "--hdf5_path", type=str, default=os.path.join('data', 'lung_radiomics', 'lung_radiomics_datasets.hdf5'),
                        help="path al dataset en formato HDF5 con imagenes isotropicas")
    parser.add_argument("-df", "--df_path", type=str, default=os.path.join('data', 'lung_radiomics', 'lung_radiomics_datasets.csv'),
                        help="path a los metadatos del dataset")
    parser.add_argument("-mod", "--modality", type=str, default='ct',
                        help="path a los metadatos del dataset")
    parser.add_argument("-gpu", "--gpu_device", type=str, default='0', help="cuda device number")

    args = parser.parse_args()
    model_name = args.model_name
    model_path = args.model_path
    dataset_path = args.dataset_path
    feature_folder = args.feature_folder
    ds_path = args.hdf5_path
    df_metdata_path = args.df_path
    second_modality = args.modality
    use_tfds = ds_path is None
    gpu_device = args.gpu_device
    torch.cuda.set_device(int(gpu_device))

    model = load_model(model_name, model_path)
    datasets = ['santa_maria_dataset', 'stanford_dataset']
    modalities = ['pet', second_modality] # [pet, ct] or [pet, chest]
    dataframes = []
    if not use_tfds:
        df_metadata = pd.read_csv(df_metdata_path)
        df_metadata['label'] = (df_metadata['egfr'] == 'Mutant').astype(int)
        patient2label = dict(zip(df_metadata['patient_id'], df_metadata['label']))
        if second_modality == 'pet':
            df_metadata = df_metadata[np.logical_or(df_metadata['has_petct'], df_metadata['has_petchest'])]
        else:
            df_metadata = df_metadata[df_metadata[f'has_{"".join(modalities)}']]
        df_metadata.reset_index(inplace=True, drop=True)

    for dataset_name in datasets:
        features_dir = os.path.join(feature_folder, dataset_name)
        os.makedirs(features_dir, exist_ok=True)
        if use_tfds:
            if dataset_name == 'stanford_dataset':
                ds_pet, info_pet = tfds.load(f'{dataset_name}/pet', data_dir=dataset_path, with_info=True)
                ds_ct, info_ct = tfds.load(f'{dataset_name}/ct', data_dir=dataset_path, with_info=True)
            else:
                ds_pet, info_pet = tfds.load(f'{dataset_name}/pet', data_dir=dataset_path, with_info=True)
                ds_ct, info_ct = tfds.load(f'{dataset_name}/torax3d', data_dir=dataset_path, with_info=True)

            patient_pet = set(list(ds_pet.keys()))
            patient_ct = set(list(ds_ct.keys()))

            patient_ids = list(patient_ct.intersection(patient_pet))
        else:
            dataset_name_sort = dataset_name.replace('_dataset', '')
            patient_ids = list(df_metadata[df_metadata['dataset'] == dataset_name_sort]['patient_id'].unique())

        for patient_id in tqdm(patient_ids, desc=dataset_name):
            for modality in [second_modality]:
                df_path = os.path.join(features_dir, f'{patient_id}_{modality}.parquet')
                features_file = os.path.join(feature_folder, f'features_masks_{modality}.hdf5')
                if not os.path.exists(df_path):
                    if use_tfds:
                        if modality == 'pet':
                            img_raw, mask_raw, label, spatial_res = tfds2voxels(ds_pet, patient_id, pet=True)
                        else:
                            img_raw, mask_raw, label, spatial_res = tfds2voxels(ds_ct, patient_id)

                        label = label[0]
                        if label not in [0, 1]:  # ignore unknown (2) and not collected (3) labels
                            print(f'\nWarning: skip {patient_id} with label {label}')
                        else:
                            nodule_pixels = mask_raw.sum(axis=(0, 1)).round(2)
                            if not nodule_pixels.max():
                                print(f'\nWarning: {patient_id} has empty mask')

                            # normalize pixel values
                            if modality == 'ct':
                                img_raw = apply_window_ct(img_raw, width=800, level=40)
                            else:
                                img_raw = img_raw / img_raw.max()
                    else:
                        label = patient2label[patient_id]
                        img_raw, mask_raw, _, spatial_res = get_voxels(ds_path, patient_id, modality)
                        # normalize pixel values
                        if modality == 'pet':
                            img_raw = img_raw / img_raw.max()
                        else:
                            img_raw = apply_window_ct(img_raw, width=1800, level=40)
                        #img_raw, mask_raw, spatial_res = get_voxels(ds_path, patient_id, modality)

                        # extract patch features of each slice
                        df = {'slice': [],
                              'angle': [],
                              'flip': []}

                        all_features = []
                        all_masks = []
                        angles = []
                        flips = []
                        slices = []
                        # apply flip and rotation to use them as offline data augmentation
                        for flip_type in [None, 'horizontal', 'vertical']:
                            image_flip, mask_flip = flip_image(img_raw, mask_raw, flip_type)
                            for angle in [0, 90]:#range(0, 180, 45):
                                image, mask = rotate_image(image_flip, mask_flip, angle)
                                features, features_mask = generate_features(model=model,
                                                                            img_3d=image,
                                                                            mask_3d=mask,
                                                                            tqdm_text=f'{modality} {patient_id}',
                                                                            display=False)

                                all_masks += features_mask
                                all_features += features

                                df['angle'] += [angle] * len(features)
                                df['flip'] += [flip_type] * len(features)
                                df['slice'] += list(range(0, len(features)))

                        # store metadata of each featuremap in a dataframe
                        df = pd.DataFrame(df)
                        df.reset_index(drop=False, inplace=True)
                        df = df.rename(columns={'index': 'feature_id'})
                        df['patient_id'] = patient_id
                        df['label'] = label
                        df['dataset'] = dataset_name.replace('_dataset', '')
                        df['modality'] = modality
                        df['augmentation'] = np.logical_not(np.logical_and(df['flip'] is None,  df['angle'] == 0))
                        df['spatial_res'] = [spatial_res] * df.shape[0]
                        df.to_parquet(df_path)
                        save_features(features_file, all_features, all_masks, patient_id)
