import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import random
from tfds_dense_descriptor import get_voxels, flip_image, rotate_image,apply_window_ct
import numpy as np
import torch.nn.functional as F
from skimage.transform import resize
from visualization_utils import (crop_image,
                                 extract_coords,
                                 extract_roi,
                                 visualize_features,
                                 hu_to_rgb_vectorized)

class PETCTDataset3D(Dataset):
    def __init__(self, dataframe, label_encoder, hdf5_ct_path, hdf5_pet_path, modality_a='pet', modality_b='ct', use_augmentation=False, feature_dim=256, arch='conv'):
        self.slice_per_modality = dataframe.groupby(['patient_id', 'modality'])['slice'].max()
        self.df_ct = dataframe[dataframe['modality'] == modality_b].reset_index(drop=True)
        self.df_pet = dataframe[dataframe['modality'] == modality_a].reset_index(drop=True)
        self.modality_a = modality_a
        self.modality_b = modality_b
        if use_augmentation:
            n_samples = len(self.df_ct['patient_id_new'].unique())
            self.dataframe = self.df_ct.copy()
            self.dataframe['patient_id_new_int'] = self.dataframe['patient_id_new'].str.split(':').str[-1]
            self.dataframe['patient_id_new_int'] = self.dataframe['patient_id_new_int'].astype(int)
            self.dataframe.sort_values(by='patient_id_new_int', inplace=True, ascending=False)
            self.dataframe = self.dataframe.groupby(['patient_id'])[['modality', 'dataset', 'label', 'patient_id_new', 'patient_id_new_int']].first()
            self.dataframe.reset_index(inplace=True, drop=False)
            repeat_times = np.clip(np.ceil(n_samples / self.dataframe.shape[0]), 2, 8)
            self.dataframe = pd.DataFrame(np.repeat(self.dataframe.values, repeat_times, axis=0), columns=self.dataframe.columns)
        else:
            self.dataframe = self.df_ct.groupby(['patient_id_new'])[['modality', 'dataset', 'label', 'patient_id']].first()
            self.dataframe.reset_index(inplace=True, drop=False)

        self.use_augmentation = use_augmentation

        self.flip_angles = dataframe.groupby(['flip', 'angle'], as_index=False).size()[['flip', 'angle']]

        self.df_ct = self.df_ct.set_index(['patient_id_new', 'angle', 'flip'])
        self.df_pet = self.df_pet.set_index(['patient_id', 'angle', 'flip'])
        self.df_ct = self.df_ct.sort_index()
        self.df_pet = self.df_pet.sort_index()

        self.hdf5_ct_path = hdf5_ct_path
        self.hdf5_pet_path = hdf5_pet_path
        self.label_encoder = label_encoder
        self.feature_dim = feature_dim
        self.arch = arch

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        enable_random_crop = True
        noise_val = 10
        sample = self.dataframe.iloc[idx]
        patient_id_rew = sample.patient_id_new
        patient_id = sample.patient_id
        label = sample.label
        noise = np.random.random(3) * noise_val - noise_val/2
        scale_noise = np.random.uniform(0.85, 1.15)
        if self.use_augmentation:
            [[flip, angle]] = self.flip_angles.sample(n=1).values
            patient_int = sample.patient_id_new_int
            if patient_int > 0:
                patient_int = np.random.randint(0, patient_int)
            patient_id_rew = f'{patient_id}:{patient_int}'
        else:
            flip = 'None'
            angle = 0
            noise = noise * 0
            scale_noise = 1.0

        ct_slices = self.df_ct.loc[(patient_id_rew, angle, flip)]['slice'].values
        start_slice_index, end_slice_index = ct_slices.argmin(), ct_slices.argmax()
        if enable_random_crop:  # TODO: move to cfg file
            if self.use_augmentation: # random slice crop
                if len(ct_slices) > 7:
                    window_size = int(np.random.randint(7, len(ct_slices), 1))
                    start_slice_index = int(np.random.randint(0, len(ct_slices)-window_size))
                    end_slice_index = int(start_slice_index + window_size)

        feature_ids = self.df_ct.loc[(patient_id_rew, angle, flip)]['feature_id'].values[start_slice_index:end_slice_index]
        spatial_res = self.df_ct.loc[(patient_id_rew, angle, flip)]['spatial_res'].values[0]
        spatial_res = np.abs(spatial_res) * scale_noise
        features_ct = self._get_features(self.hdf5_ct_path, patient_id, feature_ids, angle, flip, noise, spatial_res)
        features_ct = torch.as_tensor(features_ct, dtype=torch.float32)

        ct_slices = ct_slices[start_slice_index:end_slice_index] / self.slice_per_modality.loc[(patient_id, self.modality_b)]
        start_slice, end_slice = ct_slices.min(), ct_slices.max()

        max_slice = self.slice_per_modality[patient_id, self.modality_a]
        start_slice = max(0, int(start_slice*max_slice))
        end_slice = min(max_slice, int(end_slice*max_slice))

        df_pet = self.df_pet.loc[(patient_id, angle, flip)]
        spatial_res = df_pet['spatial_res'].values[0]
        spatial_res = np.abs(spatial_res) * scale_noise
        feature_ids = df_pet[np.logical_and(df_pet['slice'] >= start_slice, df_pet['slice'] <= end_slice)]['feature_id'].values
        features_pet = self._get_features(self.hdf5_pet_path, patient_id, feature_ids, angle, flip, noise, spatial_res)
        features_pet = torch.as_tensor(features_pet, dtype=torch.float32)

        labels = np.array(label)
        labels = np.expand_dims(labels, axis=-1)
        labels = self.label_encoder.transform(labels.reshape(-1, 1)).toarray()
        labels = torch.as_tensor(labels, dtype=torch.float32)

        return features_ct, features_pet, labels, patient_id

    def _get_features(self, hdf5_path, patient_id, feature_ids, angle, flip, noise, spatial_res):
        features = []
        masks = []
        use_mask = False

        with h5py.File(hdf5_path, 'r') as h5f:
            for feature_id in feature_ids:
                slice_features = h5f[f'{patient_id}/features/{feature_id}'][()]
                slice_mask_orig = h5f[f'{patient_id}/masks/{feature_id}'][()]
                slice_mask = resize(slice_mask_orig, slice_features.shape[0:2], order=0)
                slice_mask = np.expand_dims(slice_mask, axis=-1)
                if self.arch == 'conv':
                    features.append(slice_features * slice_mask)  # elementwise prod feature-mask
                else:
                    features.append(slice_features) 
                masks.append(slice_mask)

        features = np.transpose(np.stack(features, axis=0), axes=(3, 0, 1, 2))  # (slice, h, w, feat_dim) -> (feat_dim, slice, h, w)
        if self.arch == 'transformer':
            masks = np.transpose(np.stack(masks, axis=0), axes=(1, 2, 0, 3))  # (slice, h, w, 1) -> (h, w, slice, 1)
            h_orig, w_orig = slice_mask_orig.shape[0:2]
            features = np.transpose(features, axes=(2, 3, 1, 0))  # (h, w, slice, feat_dim)
            h_new, w_new = features.shape[0], features.shape[1]

            x, y, z = np.meshgrid(np.arange(0, features.shape[0]),
                                  np.arange(0, features.shape[1]),
                                  np.arange(0, features.shape[2]))
            x = (x.flatten() / w_new).flatten() * w_orig * spatial_res[0]
            y = (y.flatten() / h_new).flatten() * h_orig * spatial_res[1]
            z = (z.flatten()).flatten() * spatial_res[2]
            
            x = (x - x.mean() + noise[0])
            y = (y - y.mean() + noise[1])
            z = (z - z.mean() + noise[2])
    
            if use_mask:
                masks = masks.flatten()
                x = x[masks]
                y = y[masks]
                z = z[masks]

            pe = positional_encoding_3d(x, y, z, D=self.feature_dim, scale=10000)
            if use_mask:
                features = features.reshape(-1, self.feature_dim)[masks, :] + pe / 4  # (seq_len, feat_dim)
            else:
                features = features.reshape(-1, self.feature_dim) + pe / 4 
        return features


class PETCTDataset3D_onlineV1(Dataset):
    def __init__(self,cfg,df_metdata_path, label_encoder,dataset_name, hdf5_path=None, modality_a='pet',modality_b='ct',use_augmentation=False,same_size=True,mode="train",kfold=0):
        self.use_tfds = hdf5_path is None
        self.hdf5_path=hdf5_path
        self.modalities = [modality_a, modality_b]
        self.modality_a = modality_a
        self.modality_b = modality_b
        self.label_encoder=label_encoder
        self.all_flip=[None]
        self.all_angle=[0]
        kfold_patients = cfg['kfold_patients'][modality_b][dataset_name][kfold][mode]
        if use_augmentation:
            self.all_flip=[None, 'horizontal', 'vertical']
            self.all_angle=[0, 90]#range(0, 180, 45):
        self.use_augmentation = use_augmentation
        self.enable_random_crop=True
        self.same_size=same_size
        self.ct_slices_len=14      
        """
        self.feature_dim = feature_dim
        self.arch = arch
        """
        if not self.use_tfds:
            df_metadata = pd.read_csv(df_metdata_path)
            df_metadata['label'] = (df_metadata['egfr'] == 'Mutant').astype(int)
            self.patient2label = dict(zip(df_metadata['patient_id'], df_metadata['label']))
            if modality_b == 'pet':
                df_metadata = df_metadata[np.logical_or(df_metadata['has_petct'], df_metadata['has_petchest'])]
            else:
                df_metadata = df_metadata[df_metadata[f'has_{"".join(self.modalities)}']]
            df_metadata.reset_index(inplace=True, drop=True)
        
        if self.use_tfds:
            if dataset_name == 'stanford_dataset':
                self.ds_pet, info_pet = tfds.load(f'{dataset_name}/pet', data_dir=dataset_path, with_info=True)
                self.ds_ct, info_ct = tfds.load(f'{dataset_name}/ct', data_dir=dataset_path, with_info=True)
            else:
                self.ds_pet, info_pet = tfds.load(f'{dataset_name}/pet', data_dir=dataset_path, with_info=True)
                self.ds_ct, info_ct = tfds.load(f'{dataset_name}/torax3d', data_dir=dataset_path, with_info=True)
        
            patient_pet = set(list(ds_pet.keys()))
            patient_ct = set(list(ds_ct.keys()))
        
            self.patient_ids = list(patient_ct.intersection(patient_pet))
        else:
            dataset_name_sort = dataset_name.replace('_dataset', '')
            self.patient_ids = list(df_metadata[df_metadata['dataset'] == dataset_name_sort]['patient_id'].unique())

        self.patient_ids=np.array(self.patient_ids)[np.where(np.isin(self.patient_ids,kfold_patients))]
        print("Dataset Size: ",len(self.patient_ids))


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id=self.patient_ids[idx]
        flip=random.choice(self.all_flip)
        angle=random.choice(self.all_angle)
        all_images = []
        all_masks = []
        all_spatial_res = []
        for modality in self.modalities:
            if self.use_tfds:
                if modality == 'pet':
                    img_raw, mask_raw, label, spatial_res = tfds2voxels(self.ds_pet, patient_id, pet=True)
                else:
                    img_raw, mask_raw, label, spatial_res = tfds2voxels(self.ds_ct, patient_id)
        
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
                label = self.patient2label[patient_id]
                img_raw, mask_raw, _, spatial_res = get_voxels(self.hdf5_path, patient_id, modality)
                # normalize pixel values
                if modality == 'pet':
                    img_raw = img_raw / img_raw.max()
                else:
                    img_raw = apply_window_ct(img_raw, width=1800, level=40)
        
                # apply flip and rotation to use them as offline data augmentation
                image_flip, mask_flip = flip_image(img_raw, mask_raw, flip)
                image, mask = rotate_image(image_flip, mask_flip, angle)
                #features, features_mask = generate_features(model=model,
                #                                            img_3d=image,
                #                                            mask_3d=mask,
                #                                            tqdm_text=f'{modality} {patient_id}',
                #          
                all_masks.append(mask)
                all_images.append(image)
                all_spatial_res.append(spatial_res)
        
        if self.same_size:
            largo=[]
            for i in range(2):
                largo.append(np.shape(all_images[i])[2])
            max_z_size=int(np.max(largo))
            max_z_pos=np.argmin(largo)
            min_z_pos=np.argmin(largo)
            h,w,_=np.shape(all_images[min_z_pos])
            
            dato_t = torch.from_numpy(all_images[min_z_pos])
            mask_t = torch.from_numpy(all_masks[min_z_pos].astype(float))
            dato_t = dato_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            mask_t = mask_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            dato_t = F.interpolate(
                        dato_t,
                        size=(max_z_size,h, w),
                        mode="nearest-exact",
                    )
            mask_t = F.interpolate(
                        mask_t,
                        size=(max_z_size,h, w),
                        mode="nearest-exact",
                    )
            all_images[min_z_pos] = np.array(dato_t[0][0].permute(1, 2, 0))
            all_masks[min_z_pos] = np.array(mask_t[0][0].permute(1, 2, 0)).astype(bool)

        ct_slices=max_z_size
        if self.enable_random_crop:  # TODO: move to cfg file
            if self.use_augmentation: # random slice crop
                if ct_slices > 7:
                    max_amount_slices=min(self.ct_slices_len,ct_slices)
                    window_size = int(np.random.randint(7, max_amount_slices, 1))
                    start_slice_index = int(np.random.randint(0, ct_slices-window_size))
                    end_slice_index = int(start_slice_index + window_size)
                else:
                    start_slice_index=0
                    end_slice_index = max_z_size
                all_images[0]=all_images[0][:,:,start_slice_index:end_slice_index]
                all_masks[0]=all_masks[0][:,:,start_slice_index:end_slice_index]
                all_images[1]=all_images[1][:,:,start_slice_index:end_slice_index]
                all_masks[1]=all_masks[1][:,:,start_slice_index:end_slice_index]
                
        labels = np.array(label)
        labels = np.expand_dims(labels, axis=-1)
        labels = self.label_encoder.transform(labels.reshape(-1, 1)).toarray()
        labels = torch.as_tensor(labels, dtype=torch.float32)
        return all_images, all_masks, labels ,patient_id, all_spatial_res


class PETCTDataset3D_onlineV2(Dataset):
    def __init__(self,cfg,df_metdata_path, label_encoder,dataset_name, hdf5_path=None, modality_a='pet',modality_b='ct',use_augmentation=False,same_size=True,mode="train",kfold=0):
        self.use_tfds = hdf5_path is None
        self.hdf5_path=hdf5_path
        self.modalities = [modality_a, modality_b]
        self.modality_a = modality_a
        self.modality_b = modality_b
        self.label_encoder=label_encoder
        self.all_flip=[None]
        self.all_angle=[0]
        self.use_augmentation = use_augmentation
        self.enable_random_crop=True
        self.same_size=same_size
        self.ct_slices_len=13   
        
        df_metadata = pd.read_csv(df_metdata_path)
        dataset_name_sort = dataset_name.replace('_dataset', '')
        df_metadata=df_metadata[df_metadata['dataset'] == dataset_name_sort]
        if mode=="train" or mode=="test":
            kfold_patients = cfg['kfold_patients'][modality_b][dataset_name][kfold][mode]
            df_metadata=df_metadata[df_metadata['patient_id'].isin(kfold_patients)]
        self.df_ct = df_metadata[df_metadata['modality'] == modality_b].reset_index(drop=True)
        self.df_pet = df_metadata[df_metadata['modality'] == modality_a].reset_index(drop=True)
        self.patient2label = dict(zip(df_metadata['patient_id'], df_metadata['label']))

            
        if use_augmentation:
            self.all_flip=[None, 'horizontal', 'vertical']
            self.all_angle=[0, 90]#range(0, 180, 45):
            self.df_metadata = self._process_dataframe(df_metadata)
        else:
            self.df_metadata = self.df_ct.groupby(['patient_id_new'])[['modality', 'dataset', 'label','patient_id','begin','divisor','length','b_mask','f_mask']].first()
            self.df_metadata.reset_index(inplace=True, drop=False)
            

        print("Dataset Size: ",len(self.df_metadata))

    def _process_dataframe(self,df_metdata):
        dataframe=self.df_ct.copy()
        n_samples = len(dataframe['patient_id'])
        dataframe = dataframe.groupby(['patient_id'])[['modality', 'dataset', 'label','patient_id','begin','divisor','length','b_mask','f_mask']].first()
        repeat_times = np.clip(np.ceil(n_samples / dataframe.shape[0]), 2, 8)
        dataframe = pd.DataFrame(np.repeat(dataframe.values, repeat_times, axis=0), columns=dataframe.columns)
        return dataframe

    def __len__(self):
        return len(self.df_metadata)

    def _process_data(self,img_3d, mask_3d):
        bigger_mask = np.sum(mask_3d, axis=-1) > 0

        h, w = bigger_mask.shape
        xmin, ymin, xmax, ymax = extract_coords(bigger_mask, margin=2)
        crop_size = max(xmax-xmin, ymax-ymin)*2
        xmid, ymid = int(xmin + (xmax-xmin)/2), int(ymin + (ymax-ymin)/2)
        xmin, ymin, xmax, ymax = xmid-crop_size, ymid-crop_size, xmid+crop_size, ymid+crop_size
    
        img_3d = crop_image(img_3d, xmin, ymin, xmax, ymax)
        mask_3d = crop_image(mask_3d, xmin, ymin, xmax, ymax)
        bigger_mask = crop_image(bigger_mask, xmin, ymin, xmax, ymax)
        return img_3d,mask_3d,bigger_mask

    def __getitem__(self, idx):
        patient=self.df_metadata.iloc[idx]
        patient_id=patient["patient_id"]
        flip=random.choice(self.all_flip)
        angle=random.choice(self.all_angle)
        all_images = []
        all_masks = []
        all_big_masks = []
        all_spatial_res = []
        for modality in self.modalities:
            if self.use_tfds:
                if modality == 'pet':
                    img_raw, mask_raw, label, spatial_res = tfds2voxels(self.ds_pet, patient_id, pet=True)
                else:
                    img_raw, mask_raw, label, spatial_res = tfds2voxels(self.ds_ct, patient_id)
        
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
                label = self.patient2label[patient_id]
                img_raw, mask_raw, _, spatial_res = get_voxels(self.hdf5_path, patient_id, modality)
                # normalize pixel values
                if modality == 'pet':
                    img_raw = img_raw / img_raw.max()
                else:
                    img_raw = apply_window_ct(img_raw, width=1800, level=40)
        
                # apply flip and rotation to use them as offline data augmentation
            image_flip, mask_flip = flip_image(img_raw, mask_raw, flip)
            img_3d, mask_3d = rotate_image(image_flip, mask_flip, angle)
            img_3d,mask_3d,bigger_mask = self._process_data(img_3d, mask_3d)
            
            all_images.append(img_3d)
            all_masks.append(mask_3d)
            all_big_masks.append(bigger_mask)
            all_spatial_res.append(spatial_res)

        
        #Falta agregar los rangos con mascaras
        if self.same_size:
            largo=[]
            for i in range(2):
                largo.append(np.shape(all_images[i])[2])
            max_z_size=int(np.max(largo))
            min_z_pos=np.argmin(largo)
            h,w,_=np.shape(all_images[min_z_pos])
            
            dato_t = torch.from_numpy(all_images[min_z_pos])
            mask_t = torch.from_numpy(all_masks[min_z_pos].astype(float))
            dato_t = dato_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            mask_t = mask_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            dato_t = F.interpolate(
                        dato_t,
                        size=(max_z_size,h, w),
                        mode="nearest-exact",
                    )
            mask_t = F.interpolate(
                        mask_t,
                        size=(max_z_size,h, w),
                        mode="nearest-exact",
                    )
            all_images[min_z_pos] = np.array(dato_t[0][0].permute(1, 2, 0))
            all_masks[min_z_pos] = np.array(mask_t[0][0].permute(1, 2, 0)).astype(bool)
            
        
        if self.use_augmentation: # random slice crop
            if self.enable_random_crop:  # TODO: move to cfg file
                b_mask=patient["b_mask"]
                f_mask=patient["f_mask"]+1
                amount_w_mask=f_mask-b_mask
                if amount_w_mask > 7:
                    max_amount_slices=min(self.ct_slices_len,amount_w_mask)
                    window_size = int(np.random.randint(7, max_amount_slices, 1))
                    start_slice_index = int(np.random.randint(b_mask, f_mask-window_size))
                    end_slice_index = int(start_slice_index + window_size)
                else:
                    start_slice_index=b_mask
                    end_slice_index = f_mask
        else:
            start_slice_index = patient["begin"]
            end_slice_index = start_slice_index+patient["divisor"]

        
        all_images[0]=all_images[0][:,:,start_slice_index:end_slice_index]
        all_masks[0]=all_masks[0][:,:,start_slice_index:end_slice_index]
        all_images[1]=all_images[1][:,:,start_slice_index:end_slice_index]
        all_masks[1]=all_masks[1][:,:,start_slice_index:end_slice_index]
                
        labels = np.array(label)
        labels = np.expand_dims(labels, axis=-1)
        labels = self.label_encoder.transform(labels.reshape(-1, 1)).toarray()
        labels = torch.as_tensor(labels, dtype=torch.float32)
        return all_images, all_masks, all_big_masks,labels ,patient_id, all_spatial_res