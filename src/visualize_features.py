# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:18:22 2024

@author: Mico
"""

import os
import pandas as pd
import numpy as np
from skimage.transform import resize
import visualization_utils as viz
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import plotly.express as px
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage


def compute_iou(y_pred, y_true, eps=1e-10):
    intersection = np.sum(np.logical_and(y_pred, y_true))
    iou = intersection / (np.sum(y_true) + np.sum(y_pred) - intersection + eps)
    return iou

def get_volume_embeds(hdf5_path, patient_id, feature_ids):
    features = []
    masks = []

    with h5py.File(hdf5_path, 'r') as h5f:
        for feature_id in feature_ids:
            slice_features = h5f[f'{patient_id}/features/{feature_id}'][()]
            slice_mask = h5f[f'{patient_id}/masks/{feature_id}'][()]
            slice_mask = resize(slice_mask, slice_features.shape[0:2], order=0)
            slice_mask = np.expand_dims(slice_mask, axis=-1)
            features.append(slice_features)
            masks.append(slice_mask)
    features = np.transpose(np.stack(features, axis=0), axes=(3, 0, 1, 2))  # (slice, h, w, 1) -> (h, w, slice, 1)
    features = np.transpose(features, axes=(2, 3, 1, 0))  # (h, w, slice, feat_dim)
    masks = np.transpose(np.stack(masks, axis=0), axes=(1, 2, 0, 3))
    return features, masks

if __name__ == "__main__":
    features_folder = 'features'#'features_medsam'
    
    modality_a = 'pet'
    modality_b = 'ct'
    restore_mask = True
    #modality_b = 'chest'
    
    hdf5_pet_path = os.path.join('..', 'data', features_folder, f'features_masks_{modality_a}.hdf5')
    hdf5_ct_path = os.path.join('..', 'data', features_folder, f'features_masks_{modality_b}.hdf5')
    df_path = os.path.join('..', 'data', features_folder, 'petct.parquet')
    df = pd.read_parquet(df_path)
    df['flip'] = df['flip'].astype(str)
    df.reset_index(drop=True, inplace=True)
    
    #df[df['augmentation'] == False]

    df_pet = df[df['modality'] == modality_a]
    df_ct = df[df['modality'] == modality_b]
    
    sample = df_pet.sample(n=1)
    index_columns = ['patient_id', 'angle', 'flip']
    patient_id, angle, flip = sample[index_columns].values[0]
    selected_index = (patient_id, angle, flip)

    ct_ids = df_ct.set_index(index_columns).loc[selected_index]['feature_id'].to_list()
    ct_spatial_res = df_ct.set_index(index_columns).loc[selected_index]['spatial_res'].to_list()[0]
    pet_ids = df_pet.set_index(index_columns).loc[selected_index]['feature_id'].to_list()
    pet_spatial_res = df_pet.set_index(index_columns).loc[selected_index]['spatial_res'].to_list()[0]
    
    ct_features, ct_masks = get_volume_embeds(hdf5_ct_path, patient_id, ct_ids)
    pet_features, pet_masks = get_volume_embeds(hdf5_pet_path, patient_id, pet_ids)

    io.imshow(pet_masks.mean(axis=2))
    plt.title(f'mask {modality_a}: {patient_id}')
    io.show()

    io.imshow(ct_masks.mean(axis=2))
    plt.title(f'mask {modality_b}: {patient_id}')
    io.show()

    if restore_mask:
        pet_masks[:,:,:,0] = ndimage.binary_dilation(pet_masks[:,:,:,0])
        pet_masks[:,:,:,0] = ndimage.binary_closing(pet_masks[:,:,:,0])
        pet_masks[:,:,:,0] = ndimage.binary_opening(pet_masks[:,:,:,0])
        io.imshow(pet_masks.mean(axis=2))
        plt.title(f'res mask {modality_a}: {patient_id}')
        io.show()

        ct_masks[:,:,:,0] = ndimage.binary_dilation(ct_masks[:,:,:,0])
        ct_masks[:,:,:,0] = ndimage.binary_closing(ct_masks[:,:,:,0])
        ct_masks[:,:,:,0] = ndimage.binary_opening(ct_masks[:,:,:,0])
        io.imshow(ct_masks.mean(axis=2))
        plt.title(f'res mask {modality_b}: {patient_id}')
        io.show()
    volume_render = True
    use_mask = True
    pca_pet = PCA(n_components=3)
    pca_ct = PCA(n_components=3)
    feat_dim = ct_features.shape[-1]
    pca_ct.fit(ct_features.reshape(-1, feat_dim))
    pca_pet.fit(pet_features.reshape(-1, feat_dim))
    #pca_ct.fit(ct_features[np.squeeze(ct_masks)])
    #pca_pet.fit(pet_features[np.squeeze(pet_masks)])
    
    ct_viz = pca_ct.transform(ct_features.reshape(-1, feat_dim))
    pet_viz = pca_pet.transform(pet_features.reshape(-1, feat_dim))
    ct_viz = viz.min_max_scale(ct_viz)
    pet_viz = viz.min_max_scale(pet_viz)
    
    # ct
    df_ct_viz = pd.DataFrame()
    h, w, s = ct_features.shape[0:3]
    X, Y, Z = np.mgrid[0:h, 0:w, 0:s]
    df_ct_viz['x'] = X.flatten() * ct_spatial_res[0] #* 16
    df_ct_viz['y'] = Y.flatten() * ct_spatial_res[1] #* 16
    df_ct_viz['z'] = Z.flatten() * ct_spatial_res[2]
    df_ct_viz['r'] = (ct_viz[:,0]*255).astype(int)
    df_ct_viz['g'] = (ct_viz[:,1]*255).astype(int)
    df_ct_viz['b'] = (ct_viz[:,2]*255).astype(int)
    
    df_ct_viz[ct_masks.flatten()].to_csv('ct_viz.txt', sep=' ', index=False)

    df_pet_viz = pd.DataFrame()
    h, w, s = pet_features.shape[0:3]
    X, Y, Z = np.mgrid[0:h, 0:w, 0:s]
    df_pet_viz['x'] = X.flatten() * pet_spatial_res[0] * 16
    df_pet_viz['y'] = Y.flatten() * pet_spatial_res[1] * 16
    df_pet_viz['z'] = Z.flatten() * pet_spatial_res[2]
    df_pet_viz['r'] = (pet_viz[:,0]*255).astype(int)
    df_pet_viz['g'] = (pet_viz[:,1]*255).astype(int)
    df_pet_viz['b'] = (pet_viz[:,2]*255).astype(int)
    
    df_pet_viz[pet_masks.flatten()].to_csv('pet_viz.txt', sep=' ', index=False)
    

    io.imshow(pet_viz.reshape(pet_features.shape[0:3]+(3,))[:,:,pet_features.shape[2]//2,:])
    plt.title(f'{modality_a}: {patient_id}')
    io.show()
    io.imshow(ct_viz.reshape(ct_features.shape[0:3]+(3,))[:,:,ct_features.shape[2]//2,:])
    plt.title(f'{modality_b}: {patient_id}')
    io.show()
    


    if volume_render:
        pca_pet = PCA(n_components=1)
        pca_ct = PCA(n_components=1)
        
        pca_ct.fit(ct_features[np.squeeze(ct_masks)])
        pca_pet.fit(pet_features[np.squeeze(pet_masks)])
        
        feat_dim = ct_features.shape[-1]
        ct_viz = pca_ct.transform(ct_features.reshape(-1, feat_dim))
        ct_viz = ct_viz.reshape(ct_features.shape[:-1])
    
        pet_viz = pca_pet.transform(pet_features.reshape(-1, feat_dim))
        pet_viz = pet_viz.reshape(pet_features.shape[:-1])
        
        ct_viz = viz.min_max_scale(ct_viz)
        pet_viz = viz.min_max_scale(pet_viz)
        
        pet_pca_mask = (pet_viz > np.percentile(pet_viz, 60))
        ct_pca_mask = (ct_viz > np.percentile(pet_viz, 60))
        pet_iou = compute_iou(pet_pca_mask.flatten(), pet_masks.flatten())
        ct_iou = compute_iou(ct_pca_mask.flatten(), ct_masks.flatten())
        
        io.imshow(pet_pca_mask.mean(axis=2))
        plt.title(f'PCA mask {modality_a}: {patient_id}\nIoU {pet_iou}')
        io.show()

        io.imshow(ct_pca_mask.mean(axis=2))
        plt.title(f'PCA mask {modality_b}: {patient_id}\nIoU {ct_iou}')
        io.show()

        print(f'PET mask IoU {pet_iou}')
        print(f'CT mask IoU {ct_iou}')

        if use_mask:
            ct_viz = ct_viz * np.squeeze(ct_masks)
            pet_viz = pet_viz * np.squeeze(pet_masks)
        
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['CT', 'PET'],
                            specs=[[{'type': 'volume'}, {'type': 'volume'}]])
        h, w, s = ct_viz.shape
        X, Y, Z = np.mgrid[0:h, 0:w, 0:s]
        X=X.flatten() * ct_spatial_res[0] * 16
        Y=Y.flatten() * ct_spatial_res[1] * 16
        Z=Z.flatten() * ct_spatial_res[2]
        fig.add_trace(go.Volume(x=X, y=Y, z=Z,
                        value=ct_viz.flatten(),
                        isomin=0.1,
                        isomax=1,
                        opacity=0.1,
                        colorscale='jet',
                        surface_count=16), row=1, col=1)
        h, w, s = pet_viz.shape
        X, Y, Z = np.mgrid[0:h, 0:w, 0:s]
        X=X.flatten() * pet_spatial_res[0] * 16
        Y=Y.flatten() * pet_spatial_res[1] * 16
        Z=Z.flatten() * pet_spatial_res[2]
        fig.add_trace(go.Volume(x=X, y=Y, z=Z,
                        value=pet_viz.flatten(),
                        isomin=0.1,
                        isomax=1,
                        opacity=0.1,
                        colorscale='jet',
                        surface_count=16), row=1, col=2)
        patient_info = list(sample[['dataset', 'patient_id', 'label']].values[0])
        label = ['EGFR-', 'EGFR+'][patient_info[2]]
        title_text = '<br>'.join([f'dataset: {patient_info[0]}',
                                   f'patient_id: {patient_info[1]}',
                                   f'label: {label}',
                                   f'{features_folder}'])
        #title_text = f'{backbone} features'
        fig.update_layout(title_text=title_text)
        fig.write_html('volume_render.html')

    