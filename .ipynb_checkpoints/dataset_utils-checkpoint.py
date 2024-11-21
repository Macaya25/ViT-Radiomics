# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:18:28 2024

@author: Mico
"""
import h5py
import numpy as np
import pandas as pd


def get_voxels(hdf5_path, patient_id, modality):
    pet_liver_mean = 1
    is_pet = modality == 'pet'
    eps = 1e-10
    with h5py.File(hdf5_path, 'r') as h5f:
        idm = f'{patient_id}_{modality}'
        if is_pet:
            pet_liver = h5f[f'{idm}/pet_liver'][()]
            pet_liver_mean = pet_liver[pet_liver != 0].mean() + eps
        slices = [int(k) for k in h5f[f'{idm}/img_exam'].keys()]
        slices.sort()
        img = np.dstack([h5f[f'{idm}/img_exam/{k}'][()] for k in slices])
        mask = np.dstack([h5f[f'{idm}/mask_exam/{k}'][()] for k in slices])
        spatial_res = np.abs(h5f[f'{idm}/spatial_res'][()])
        label = np.abs(h5f[f'{idm}/egfr_label'][()])

    if is_pet:
        img = img / pet_liver_mean

    return img, mask, label, spatial_res

def get_image_mask(hdf5_path, patient_id, modality, slice_id):
    is_pet = modality == 'pet'
    eps = 1e-10
    with h5py.File(hdf5_path, 'r') as h5f:
        idm = f'{patient_id}_{modality}'
        img = h5f[f'{idm}/img_exam/{slice_id}'][()]
        mask = h5f[f'{idm}/mask_exam/{slice_id}'][()]
        if is_pet:
            pet_liver = h5f[f'{idm}/pet_liver'][()]
            pet_liver_mean = pet_liver[pet_liver != 0].mean() + eps
            img = img / pet_liver_mean
    return img, mask

def dataframe_from_hdf5(hdf5_path):
    df = {'patient_id': [], 'dataset': [], 'modality': [], 'slices': [], 'spatial_res': [], 'label': []}
    with h5py.File(hdf5_path, 'r') as h5f:
        patients = list(h5f.keys())
        for idm in patients:
            num_slices = max([int(k) for k in h5f[f'{idm}/img_exam'].keys()])
            spatial_res = np.abs(h5f[f'{idm}/spatial_res'][()])
            label = np.abs(h5f[f'{idm}/egfr_label'][()])

            modality = idm.split('_')[-1]
            patient_id = idm[:-len(modality)-1]
            dataset = 'santa_maria' if 'sm_' in patient_id else 'stanford'
            df['patient_id'].append(patient_id)
            df['dataset'].append(dataset)
            df['modality'].append(modality)
            df['slices'].append(num_slices)
            df['spatial_res'].append(spatial_res)
            df['label'].append(label)
    df = pd.DataFrame(df)
    return df

def percentile_window(ct, qmin, qmax):
    ct_min_val = np.percentile(ct, q=qmin)
    ct_max_val = np.percentile(ct, q=qmax) 
    ct = contrast_stretching(ct, ct_min_val, ct_max_val)
    return ct

def contrast_stretching(ct, ct_min_val, ct_max_val):
    ct_range = ct_max_val - ct_min_val
    ct = (ct - ct_min_val) / ct_range
    ct = np.clip(ct, 0, 1)
    return ct 
    
def apply_window_ct(ct, width, level):
    ct_min_val, ct_max_val = windowing_ct(width, level)
    ct = contrast_stretching(ct, ct_min_val, ct_max_val)
    return ct

def windowing_ct(width, level):
    lower_bound = level - width/2
    upper_bound = level + width/2
    return lower_bound, upper_bound

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    # https://drive.google.com/file/d/1VHpnW2fXLwfGC3lvSiee3w2mPV6AwY84/view?usp=drive_link
    dataset_dir = r'C:\\Users\\Mall chino\\Desktop\\Trabajo\\Datasets\\PET-CT'
    hdf5_path = os.path.join(dataset_dir, 'lung_radiomics_datasets.hdf5')
    df = dataframe_from_hdf5(hdf5_path)

    sample = df.sample(n=1).iloc[0]
    patient_id = sample['patient_id']
    modality = sample['modality']
    dataset = sample['dataset']
    slice_id = np.random.randint(0, sample['slices'])
    img_raw, mask = get_image_mask(hdf5_path, patient_id, modality, slice_id)

    if modality == 'pet':
        img = percentile_window(img_raw, qmin=0, qmax=100)
    else:
        img = apply_window_ct(img_raw, width=800, level=40)
        
    img_mask = mark_boundaries(img, mask)
    plt.imshow(img_mask)
    plt.title(f'{modality}\n{patient_id}\n{dataset}')
    plt.show()
