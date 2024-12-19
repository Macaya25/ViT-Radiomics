# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 07:53:47 2024

@author: Mico
"""
import os
import pandas as pd
import numpy as np
import json
import h5py
from tqdm import tqdm
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from models_archs import (TransformerNoduleBimodalClassifier,
                          NoduleClassifier,
                          TransformerNoduleClassifier,
                          save_checkpoint)
from config_manager import load_conf
from datasets import PETCTDataset3D_onlineV1, PETCTDataset3D_onlineV2
from tfds_dense_descriptor import get_voxels, flip_image, rotate_image, apply_window_ct,load_model,get_dense_descriptor
from visualization_utils import extract_roi
from all_medsams import medsam_normvit_neck, LoRA_sam

def positional_encoding_3d(x, y, z, D, scale=10000):
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    n_points = x.shape[0]
    encoding = np.zeros((n_points, D))

    for i in range(D // 6):
        exponent = scale ** (6 * i / D)
        encoding[:, 2*i] = np.sin(x / exponent)
        encoding[:, 2*i + 1] = np.cos(x / exponent)
        encoding[:, 2*i + D // 3] = np.sin(y / exponent)
        encoding[:, 2*i + 1 + D // 3] = np.cos(y / exponent)
        encoding[:, 2*i + 2 * D // 3] = np.sin(z / exponent)
        encoding[:, 2*i + 1 + 2 * D // 3] = np.cos(z / exponent)

    return encoding



def generate_features(model, img_3d, mask_3d, bigger_mask):
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

    features_list = []
    mask_list = []
    for slice_i in range(0, img_3d.shape[2]):
        mask = mask_3d[:, :, slice_i] > 0
        img = img_3d[:, :, slice_i]
        features = get_dense_descriptor(model, img)
        crop_features = extract_roi(features, bigger_mask)
        crop_mask = extract_roi(mask, bigger_mask)
        features_list.append(crop_features)
        mask_list.append(crop_mask)
    return features_list, mask_list


def print_classification_report(report, global_metrics=None):
    """ Display a sklearn like classification_report with extra metrics

    Args:
        report (dict): Sklearn classification_report .
        global_metrics (list[str], optional): list of the extra global metrics.

    """
    df = pd.DataFrame(report)
    df = df.round(3)

    if global_metrics is None:
        global_metrics = ['accuracy', 'ROC AUC', 'kfold', 'loss', 'epoch', 'split']
    headers = df.index.to_list()

    df = df.T.astype(str)
    support = df.loc['macro avg'].iloc[-1]

    for row in global_metrics:
        metric_val = df.loc[row].iloc[-2]
        new_row = [' ']*len(headers)
        new_row[-2] = metric_val
        new_row[-1] = support
        df.loc[row] = new_row

    local_metrics = [col for col in df.T.columns if col not in global_metrics]
    df_local = df.loc[local_metrics]
    df_global = df.loc[global_metrics].T[-2:-1]
    df_global.index = ['   ']

    final_str = f'\n{df_global}\n\n{df_local}\n\n'
    print(final_str)
    return final_str


def plot_loss_metrics(df_loss, title):
    """ Plot loss vs epoch with the standard desviation between kfolds

    Args:
        df_loss (pd.dataframe): dataframe with the training loss metrics.

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure.

    """
    metric_names = ['Loss', 'AUC', 'F1', 'Target_metric']
    plot_grid = [[1, 1], [1, 2], [2, 1], [2, 2]]
    fig = make_subplots(rows=2,
                        shared_xaxes=True,
                        cols=2,
                        subplot_titles=metric_names)
    for plot_i, metric_name in enumerate(metric_names):
        metric_name = metric_name.lower()
        if f'train_{metric_name}' in df_loss.columns:
            fig.append_trace(go.Scatter(x=df_loss['epoch'],
                                        y=df_loss[f'train_{metric_name}'],
                                        mode='lines+markers',
                                        marker_color='red',
                                        name=f'train_{metric_name}',
                                        hovertext=df_loss['train_report']
                                        ),
                             row=plot_grid[plot_i][0], col=plot_grid[plot_i][1])
            fig.append_trace(go.Scatter(x=df_loss['epoch'],
                                        y=df_loss[f'test_{metric_name}'],
                                        mode='lines+markers',
                                        marker_color='blue',
                                        name=f'test_{metric_name}',
                                        hovertext=df_loss['test_report']
                                        ),
                             row=plot_grid[plot_i][0], col=plot_grid[plot_i][1])
        else:
            fig.append_trace(go.Scatter(x=df_loss['epoch'],
                                        y=df_loss[f'{metric_name}'],
                                        mode='lines+markers',
                                        marker_color='green',
                                        name=f'{metric_name}',
                                        hovertext=df_loss['is_improvement']),
                             row=plot_grid[plot_i][0], col=plot_grid[plot_i][1])
    fig.update_layout(title_text=title.capitalize(), xaxis_title="Epochs",)
    return fig


def create_labelmap(label_names):
    """ Create a labelmap from a list labels

    Args:
        label_names (list): list of labels.

    Returns:
        labelmap (dict): to convert label_id to label_name.
        labelmap_inv (dict): to convert label_name to label_id.

    """
    labelmap = dict(zip(np.arange(0, len(label_names)), label_names))
    labelmap_inv = dict(zip(label_names, np.arange(0, len(label_names))))
    return labelmap, labelmap_inv


def get_y_true_and_pred(y_true, y_pred, cpu=False):
    """ Check tensor sizes and apply softmax to get y_score

    Args:
        y_true (torch.tensor): batch of one-hot encoding true labels.
        y_pred (torch.tensor): batch of prediction logits.
        cpu (bool, optional): return tensors as numpy arrays. Defaults to False.

    Returns:
        y_true (torch.tensor or np.array): true labels.
        y_score (torch.tensor or np.array): pred labels probabilities.

    """
    y_true = torch.squeeze(y_true)
    y_pred = torch.squeeze(y_pred)
    assert y_pred.size() == y_true.size()

    if len(y_true.shape) == 1:
        y_pred = torch.unsqueeze(y_pred, 0)
        y_true = torch.unsqueeze(y_true, 0)

    y_score = F.softmax(y_pred, dim=1)
    y_true = torch.argmax(y_true, dim=1)

    if cpu:
        y_true = y_true.detach().cpu().numpy()
        y_score = y_score.detach().cpu().numpy()

    return y_true, y_score


def get_sampler_weights(train_labels):
    """ Compute the sampler weights of the train dataloader

    Args:
        train_labels (np.array): labels of the train dataloader.

    Returns:
        weights (list): a weight for each element in the dataloader.

    """

    label_value, counts = np.unique(train_labels, return_counts=True)
    labels_counts = dict(zip(label_value, counts))
    weights = [1/labels_counts[lb] for lb in train_labels]

    return weights


class CrossModalFocalLoss(nn.Module):
    """
     Multi-class Cross Modal Focal Loss
    """
    def __init__(self, gamma_bimodal=0, gamma_unimodal=2, alpha=None, beta=0.5):
        super(CrossModalFocalLoss, self).__init__()
        self.gamma_bimodal = gamma_bimodal
        self.gamma_unimodal = gamma_unimodal
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8

    def forward(self, inputs_petct, targets, inputs_ct, inputs_pet=None):
        """
        inputs_petct: [N, C], float32
        inputs_ct: [N, C], float32
        inputs_pet: [N, C], float32
        target: [N, ], int64
        """
        if len(inputs_petct.shape) == 1:
            inputs_petct = torch.unsqueeze(inputs_petct, 0)
            inputs_ct = torch.unsqueeze(inputs_ct, 0)
            if inputs_pet!=None:      
                inputs_pet = torch.unsqueeze(inputs_pet, 0)
            targets = torch.unsqueeze(targets, 0)
        class_indices = torch.argmax(targets, dim=1)

        logpt_petct = F.log_softmax(inputs_petct, dim=1)
        logpt_ct = F.log_softmax(inputs_ct, dim=1)

        pt_petct = torch.exp(logpt_petct)
        logpt_petct = (1-pt_petct)**self.gamma_bimodal * logpt_petct
        loss_petct = F.nll_loss(logpt_petct, class_indices, self.alpha, reduction='mean')

        pt_ct = torch.exp(logpt_ct)
        
        if inputs_pet!=None:       
            logpt_pet = F.log_softmax(inputs_pet, dim=1)
            pt_pet = torch.exp(logpt_pet)
            pt_mean = (2*pt_ct*pt_pet) / (pt_ct + pt_pet + self.eps)
            logpt_pet = (1-pt_mean*pt_pet)**self.gamma_unimodal * logpt_pet
            loss_pet = F.nll_loss(logpt_pet, class_indices, self.alpha, reduction='mean')
        else:
            pt_mean = (2*pt_ct*pt_ct) / (pt_ct + pt_ct + self.eps)
        
        logpt_ct = (1-pt_mean*pt_ct)**self.gamma_unimodal * logpt_ct
        loss_ct = F.nll_loss(logpt_ct, class_indices, self.alpha, reduction='mean')

        if inputs_pet==None:
            loss = (self.beta*loss_petct + (1-self.beta)*(loss_ct))
        else:
            loss = (self.beta*loss_petct + (1-self.beta)*(loss_ct + loss_pet))
        return loss


class FocalLoss(nn.Module):
    """
     Multi-class Focal Loss
    """
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = alpha

    def forward(self, inputs, targets):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        if len(inputs.shape) == 1:
            inputs = torch.unsqueeze(inputs, 0)
            targets = torch.unsqueeze(targets, 0)
        class_indices = torch.argmax(targets, dim=1)

        logpt = F.log_softmax(inputs, dim=1)

        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, class_indices, self.weight, reduction='sum')
        return loss


def find_divisor(slice_count, modality):
    if modality == 'ct' or modality == 'chest':
        desired_slices = 13
    else:
        desired_slices = 2
    return np.clip(desired_slices, 1, slice_count)



def get_number_of_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    return param_count

def build_model(cfg, feature_dim,arch, modality, modality_a, modality_b,mono_train ,num_classes=2):
    cfg_model = cfg['models'][arch]
    if modality == 'petct' or modality == 'petchest':
        print("\nUsing Bimodal Classifier\n")
        mlp_ratio_ct = cfg_model[modality_b]['mlp_ratio']
        mlp_ratio_pet = cfg_model[modality_a]['mlp_ratio']

        num_heads_ct = cfg_model[modality_b]['num_heads']
        num_heads_pet = cfg_model[modality_a]['num_heads']

        num_layers_ct = cfg_model[modality_b]['num_layers']
        num_layers_pet = cfg_model[modality_a]['num_layers']

        model = TransformerNoduleBimodalClassifier(feature_dim,
                                                   mlp_ratio_ct, mlp_ratio_pet,
                                                   num_heads_ct, num_heads_pet,
                                                   num_layers_ct, num_layers_pet,
                                                   mono_train,num_classes=num_classes)
    elif arch == 'conv':
        print("\nUsing Nodule Classifier\n")
        div = cfg['models'][arch][modality]['div']
        model = NoduleClassifier(input_dim=feature_dim, num_classes=num_classes, div=div)
    else:
        print("\nUsing Monomodal Classifier\n")
        mlp_ratio = cfg_model[modality]['mlp_ratio']
        num_heads = cfg_model[modality]['num_heads']
        num_layers = cfg_model[modality]['num_layers']
        dim_feedforward = int(feature_dim*mlp_ratio)
        model = TransformerNoduleClassifier(input_dim=feature_dim,
                                            dim_feedforward=dim_feedforward,
                                            num_heads=num_heads,
                                            num_classes=num_classes,
                                            num_layers=num_layers)
    return model

def add_mode(param,mode):
    words = param.split(".")  
    words[0] = words[0]+f"_{mode}"
    new_name = ".".join(words)
    return new_name

def load_model_mono(cfg,mono_train,ct_path,pet_path,feature_dim, arch, modality, modality_a, modality_b, froze_trans=False,num_classes=2):
    model=build_model(cfg, feature_dim,arch, modality, modality_a, modality_b,froze_trans,num_classes)
    if mono_train:
        ct_weight=torch.load(ct_path, map_location="cpu",weights_only=True)
        pet_weight=torch.load(pet_path, map_location="cpu",weights_only=True)
        params = {add_mode(k,"ct"):v for k, v in ct_weight.items() if k.startswith(('cls_token','transformer_encoder','norm'))}
        params_pet = {add_mode(k,"pet"):v for k, v in pet_weight.items() if k.startswith(('cls_token','transformer_encoder','norm'))}
        params.update(params_pet)
        model.load_state_dict(params, strict=False)
        print("\n\n Loaded Monomodal weights \n\n")
    else:
        print("\n\n Without pretrained monomodal weights \n\n")
    return model

def get_label_encoder(df):
    EGFR_names = list(df['label'].unique())
    EGFR_names.sort()
    EGFR_lm, EGFR_lm_inv = create_labelmap(EGFR_names)
    
    EGFR_encoder = OneHotEncoder(handle_unknown='ignore')
    EGFR_encoder.fit(np.array(list(EGFR_lm.keys())).reshape(-1, 1))
    return EGFR_encoder

def process(features,feature_dim,arch,spatial_res,use_augmentation):
        noise_val = 10
        noise = np.random.random(3) * noise_val - noise_val/2
        scale_noise = np.random.uniform(0.85, 1.15)
        if not use_augmentation:
            noise = noise * 0
            scale_noise = 1.0
            
        spatial_res = np.abs(spatial_res) * scale_noise

        features = np.transpose(np.stack(features, axis=0), axes=(3, 0, 1, 2))  # (slice, h, w, feat_dim) -> (feat_dim, slice, h, w)
        if arch == 'transformer':
            h_orig, w_orig = features.shape[0:2]
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
    
            pe = positional_encoding_3d(x, y, z, D=feature_dim, scale=10000)

            features = features.reshape(-1, feature_dim) + pe / 4 
        return features

def load_model_backbone(backbone):
    model_path_medsam="../medsam_vit_b.pth"
    if backbone=="medsam":
        feature_dim = 256
        model_backbone = load_model(backbone, model_path_medsam)
        model_backbone=model_backbone.eval()
        for param in model_backbone.parameters():
            param.requires_grad = False
    
    elif backbone=="medsam_normvit":
        feature_dim = 256
        model_backbone = load_model(backbone, model_path_medsam)
        model_backbone=model_backbone.eval()

    elif backbone=="medsam_lora":
        feature_dim = 256
        model_backbone = load_model(backbone, model_path_medsam)
        model_backbone = model_backbone.eval()
        
    elif backbone=="rad_dino":
        feature_dim = 768
        model_backbone = load_model(backbone)
        model_backbone=model_backbone.eval()
        for param in model_backbone.parameters():
            param.requires_grad = False
    return feature_dim,model_backbone

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D transoformer or CNN for lung nodules clasification")
    parser.add_argument("-a", "--arch", type=str, default="transformer",
                        help="'transformer' or 'conv'")
    parser.add_argument("-d", "--dataset", type=str, default="stanford",
                        help="dataset 'stanford' or 'santa_maria'")
    parser.add_argument("-b", "--backbone", type=str, default="medsam",
                        help="backbone ViT encoder 'medsam' or 'dinov2'")
    parser.add_argument("-m", "--modality", type=str, default="petchest",
                        help="'ct', 'pet', 'chest', 'petct' or 'petchest' ")
    parser.add_argument("-gpu", "--gpu", type=int, default=0,
                        help="id of gpu device, default is cuda:0")
    parser.add_argument("-l", "--loss", type=str, default='focal',
                        help="'focal' 'crossmodal'")
    parser.add_argument("-e", "--experiment", type=str, default='petct_online',
                        help="experiment name")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="threshold for classification")
    parser.add_argument("-pm", "--pretrain_mono", type=bool, default=False,
                        help="using checkpoint from a monomodal training")
    parser.add_argument("-ft", "--froze_trans", type=bool, default=False,
                        help="train transformer")
    parser.add_argument("-wm", "--weight_mono", type=str, default=None,
                        help="weight of mono pretraining")
    args = parser.parse_args()

    arch = args.arch
    backbone = args.backbone
    modality = args.modality
    gpu_id = args.gpu
    use_sampler = False
    arg_dataset = args.dataset
    loss_func = args.loss
    experiment_name = args.experiment
    threshold=args.threshold
    
    pretrain_mono=args.pretrain_mono
    froze_trans=args.froze_trans
    weight_mono=args.weight_mono

    modality_a = 'pet'
    if 'chest' in modality:
        modality_b = 'chest'
    else:
        modality_b = 'ct'
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

            
    df_metdata_path="../../../Data/PET-CT/lung_radiomics_datasets_mod.csv"
    hdf5_path = "../../../Data/PET-CT/lung_radiomics_datasets.hdf5"
    models_save_dir = os.path.join('..', 'models', experiment_name, f'{backbone}_{arch}_{arg_dataset}')

    cfg = load_conf()
    df_metadata = pd.read_csv(df_metdata_path)
    df_metadata['label'] = (df_metadata['egfr'] == 'Mutant').astype(int)
    # load dataframe and apply some filter criteria
    # create labelmap and onehot enconder for nodule EGFR mutation
    EGFR_encoder = get_label_encoder(df_metadata)
    train_metrics = {'kfold': [],
                     'epoch': [],
                     'train_loss': [],
                     'test_loss': [],
                     'train_auc': [],
                     'test_auc': [],
                     'train_f1': [],
                     'test_f1': [],
                     'train_report': [],
                     'test_report': []}

    # use KFold to split patients stratified by label
    print("begin training!!\n")
    pd_datas=pd.DataFrame(columns=["kfold","epoch","patient","patient_slices","class_real","class_1_prob"])
    folds = list(cfg['kfold_patients'][modality_b][arg_dataset].keys())
    for kfold in tqdm(folds, desc='kfold', leave=False, position=0):
        save_dir = os.path.join(models_save_dir, modality, f'kfold_{kfold}')
        os.makedirs(save_dir, exist_ok=True)


        # filter dataframes based on the split patients
        cfg_model = cfg['models'][arch]
        learning_rate = cfg_model['learning_rate']
        
        batch_size = cfg_model['batch_size']  # TODO: add support for bigger batches using zero padding to create batches of the same size
        virtual_batch_size = cfg_model['virtual_batch_size']
        start_epoch = 0 # TODO: make it able to start from a checkpoint
        num_epochs = cfg_model['num_epochs']
        best_roc_auc_test=0
        # Create model instance
        feature_dim,model_backbone=load_model_backbone(backbone)

        if pretrain_mono:
            ct_path = os.path.join(weight_mono, f'{backbone}_{arch}_{arg_dataset}',"ct",f"kfold_{kfold}","best_model_epoch.pth")
            pet_path = os.path.join(weight_mono, f'{backbone}_{arch}_{arg_dataset}',"pet",f"kfold_{kfold}","best_model_epoch.pth")

        model = load_model_mono(cfg,pretrain_mono,ct_path,pet_path,feature_dim, arch, modality, modality_a, modality_b,froze_trans=froze_trans)
        
        if froze_trans:
            for param in model.transformer_encoder_ct.parameters():
                param.requires_grad = False
            for param in model.transformer_encoder_pet.parameters():
                param.requires_grad = False
            
        print(model)
        print(model_backbone)
        print(get_number_of_params(model))
        model = model.to(device)
        model_backbone = model_backbone.to(device)

        if loss_func == 'crossmodal':
            print("crossmodal loss function!!")
            criterion = CrossModalFocalLoss(alpha=torch.tensor([0.25, 0.75]).to(device),
                                            gamma_unimodal=2.0,
                                            gamma_bimodal=1.0,
                                            beta=0.6)
        else:
            print("focal loss function!!")
            criterion = FocalLoss(alpha=torch.tensor([0.25, 0.75]).to(device), gamma=2.0)

        img_enc_params = [p for p in model_backbone.parameters() if p.requires_grad]
        img_encdec_params = list(img_enc_params) + list(model.parameters())
        optimizer = torch.optim.AdamW(img_encdec_params, lr=learning_rate, weight_decay=0.01, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*0.8, eta_min=0.0001)

        # create datasets
        train_dataset = PETCTDataset3D_onlineV2(cfg,
                                       df_metdata_path,
                                       label_encoder=EGFR_encoder,
                                       dataset_name=arg_dataset,
                                       hdf5_path=hdf5_path,
                                       modality_a=modality_a,
                                       modality_b=modality_b,
                                       use_augmentation=True,
                                       mode="train",
                                       kfold=kfold)

        test_dataset = PETCTDataset3D_onlineV2(cfg,
                                       df_metdata_path,
                                       label_encoder=EGFR_encoder,
                                       dataset_name=arg_dataset,
                                       hdf5_path=hdf5_path,
                                       modality_a=modality_a,
                                       modality_b=modality_b,
                                       use_augmentation=False,
                                       mode="test",
                                       kfold=kfold)

        # create a sampler to balance training classes proportion
        if use_sampler:
            train_labels = np.array(list(train_dataset.dataframe.label.values))
            sampler_weights = get_sampler_weights(train_labels)
            sampler = WeightedRandomSampler(sampler_weights, len(train_dataset), replacement=True)

            # create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with tqdm(total=num_epochs, desc='epoch', position=1, leave=False) as batch_pbar:
            for epoch in range(start_epoch, start_epoch+num_epochs):
                # reset loss, labels and predictions to compute epoch metrics
                total_train_loss = 0
                total_test_loss = 0

                y_true_train = []
                y_score_train = []
                patient_ids_train = []

                #y_true_test = []
                #y_score_test = []
                patient_ids_test = []

                # train loop
                model.train()
                optimizer.zero_grad()
                i = 0
                iters_to_accumulate = min(virtual_batch_size, len(train_loader))
                for images_batchs, masks_batchs,big_masks_batch,labels_batch, patient_id_batch,all_spatial_res,_ in tqdm(train_loader, position=2, desc='train batch'):
                    
                    labels_batch = torch.squeeze(labels_batch).to(device)
                    if modality == 'petct' or modality == 'petchest':
                                            
                        features_pet, _ = generate_features(model=model_backbone,
                                            img_3d=np.array(images_batchs[0][0]),
                                            mask_3d=np.array(masks_batchs[0][0]),
                                            bigger_mask=np.array(big_masks_batch[0][0]))
    
                        features_ct, _ = generate_features(model=model_backbone,
                                            img_3d=np.array(images_batchs[1][0]),
                                            mask_3d=np.array(masks_batchs[1][0]),
                                            bigger_mask=np.array(big_masks_batch[1][0]))
                        pet_batch=process(features_pet,feature_dim,arch,np.array(all_spatial_res[0][0]),use_augmentation=True)
                        pet_batch = torch.as_tensor(pet_batch, dtype=torch.float32)
                        ct_batch=process(features_ct,feature_dim,arch,np.array(all_spatial_res[1][0]),use_augmentation=True)
                        ct_batch = torch.as_tensor(ct_batch, dtype=torch.float32)
                    
                    
                        ct_batch = ct_batch.to(device).unsqueeze(0)
                        pet_batch = pet_batch.to(device).unsqueeze(0)
                        outputs = model(ct_batch, pet_batch)
                    elif modality == 'pet':
                        features_pet, _ = generate_features(model=model_backbone,
                                            img_3d=np.array(images_batchs[0][0]),
                                            mask_3d=np.array(masks_batchs[0][0]),
                                            bigger_mask=np.array(big_masks_batch[0][0]))
    
                        pet_batch=process(features_pet,feature_dim,arch,np.array(all_spatial_res[0][0]),use_augmentation=True)
                        pet_batch = torch.as_tensor(pet_batch, dtype=torch.float32)
                
                        
                        pet_batch = pet_batch.to(device).unsqueeze(0)
                        outputs = model(pet_batch)
                    elif modality == 'ct' or modality == 'chest':
                        features_ct, _ = generate_features(model=model_backbone,
                                            img_3d=np.array(images_batchs[1][0]),
                                            mask_3d=np.array(masks_batchs[1][0]),
                                            bigger_mask=np.array(big_masks_batch[1][0]))
                        
                        ct_batch=process(features_ct,feature_dim,arch,np.array(all_spatial_res[1][0]),use_augmentation=True)
                        ct_batch = torch.as_tensor(ct_batch, dtype=torch.float32)
                    
                        ct_batch = ct_batch.to(device).unsqueeze(0)
                        outputs = model(ct_batch)
                    if loss_func == 'crossmodal':
                            if modality == 'petct' or modality == 'petchest':
                                loss = criterion(torch.squeeze(outputs[0]),
                                             labels_batch,    
                                             torch.squeeze(outputs[2]),
                                             torch.squeeze(outputs[3]),
                                             )/ iters_to_accumulate
                                
                            else:
                                loss = criterion(torch.squeeze(outputs[0]),
                                             labels_batch,
                                             torch.squeeze(outputs[1]),)/ iters_to_accumulate
                    else:
                        loss = criterion(torch.squeeze(outputs[0]), labels_batch) / iters_to_accumulate
                    y_true, y_score = get_y_true_and_pred(y_true=labels_batch, y_pred=outputs[0], cpu=True)

                    y_true_train.append(y_true)
                    y_score_train.append(y_score)
                    patient_ids_train.append(np.array(patient_id_batch))

                    total_train_loss += loss.item() * iters_to_accumulate

                    loss.backward()

                    if (i + 1) % iters_to_accumulate == 0 or i + 1 == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                    i += 1
                # test loop
                model.eval()
                epoch_data={"kfold":[],"epoch":[],"patient":[],"patient_slices":[],"class_real":[],"class_1_prob":[]}
                with torch.no_grad():
                    for images_batchs, masks_batchs, big_masks_batch,labels_batch, patient_id_batch,all_spatial_res,patient_id_rew in tqdm(test_loader, position=2, desc='test batch'):
                        
                        labels_batch = torch.squeeze(labels_batch).to(device)
                        if modality == 'petct' or modality == 'petchest':
                            features_pet, _ = generate_features(model=model_backbone,
                                            img_3d=np.array(images_batchs[0][0]),
                                            mask_3d=np.array(masks_batchs[0][0]),
                                            bigger_mask=np.array(big_masks_batch[0][0]))
    
                            features_ct, _ = generate_features(model=model_backbone,
                                                img_3d=np.array(images_batchs[1][0]),
                                                mask_3d=np.array(masks_batchs[1][0]),
                                                bigger_mask=np.array(big_masks_batch[1][0]))
    
    
                            pet_batch=process(features_pet,feature_dim,arch,np.array(all_spatial_res[0][0]),use_augmentation=False)
                            pet_batch = torch.as_tensor(pet_batch, dtype=torch.float32)
                            ct_batch=process(features_ct,feature_dim,arch,np.array(all_spatial_res[1][0]),use_augmentation=False)
                            ct_batch = torch.as_tensor(ct_batch, dtype=torch.float32)
                            
                            ct_batch = ct_batch.to(device).unsqueeze(0)
                            pet_batch = pet_batch.to(device).unsqueeze(0)
                            outputs = model(ct_batch, pet_batch)
                        elif modality == 'pet':
                            features_pet, _ = generate_features(model=model_backbone,
                                            img_3d=np.array(images_batchs[0][0]),
                                            mask_3d=np.array(masks_batchs[0][0]),
                                            bigger_mask=np.array(big_masks_batch[0][0]))
    
                            pet_batch=process(features_pet,feature_dim,arch,np.array(all_spatial_res[0][0]),use_augmentation=False)
                            pet_batch = torch.as_tensor(pet_batch, dtype=torch.float32)
                            
                            pet_batch = pet_batch.to(device).unsqueeze(0)
                            outputs = model(pet_batch)
                        elif modality == 'ct' or modality == 'chest':
                            features_ct, _ = generate_features(model=model_backbone,
                                                img_3d=np.array(images_batchs[1][0]),
                                                mask_3d=np.array(masks_batchs[1][0]),
                                                bigger_mask=np.array(big_masks_batch[1][0]))
    
    
                            ct_batch=process(features_ct,feature_dim,arch,np.array(all_spatial_res[1][0]),use_augmentation=False)
                            ct_batch = torch.as_tensor(ct_batch, dtype=torch.float32)

                            ct_batch = ct_batch.to(device).unsqueeze(0)
                            outputs = model(ct_batch)
                        if loss_func == 'crossmodal':
                            if modality == 'petct' or modality == 'petchest':
                                loss = criterion(torch.squeeze(outputs[0]),
                                             labels_batch,    
                                             torch.squeeze(outputs[2]),
                                             torch.squeeze(outputs[3]),
                                             )
                                
                            else:
                                loss = criterion(torch.squeeze(outputs[0]),
                                             labels_batch,
                                             torch.squeeze(outputs[1]),)
                        
                        else:
                            loss = criterion(torch.squeeze(outputs[0]), labels_batch)
                            
                        y_true, y_score = get_y_true_and_pred(y_true=labels_batch, y_pred=outputs[0], cpu=True)
                        
                        patient_ids_test.append(np.array(patient_id_batch))
                        epoch_data["kfold"].append(kfold)
                        epoch_data["epoch"].append(epoch)
                        epoch_data["patient"].append(patient_id_batch[0])
                        epoch_data["patient_slices"].append(patient_id_rew[0])
                        epoch_data["class_real"].append(y_true[0])
                        epoch_data["class_1_prob"].append(y_score[0][1])
                        total_test_loss += loss.item()
                        
                pd_epoch=pd.DataFrame(epoch_data)
                pd_epoch_patient=pd_epoch.groupby("patient").max().reset_index()
                #print(np.array(pd_epoch["class_predict"]))
                pd_datas=pd.concat([pd_datas,pd_epoch])
                pd_datas.to_csv(os.path.join(models_save_dir, modality,'epoch_results.csv'), index=False) 
                
                y_true_test=np.array(pd_epoch_patient["class_real"])
                y_score_test=np.array(pd_epoch_patient["class_1_prob"])

                
                scheduler.step()
                avg_train_loss = total_train_loss / len(train_loader)
                avg_test_loss = total_test_loss / len(test_loader)

                batch_pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss})
                batch_pbar.update()

                # generate y_true and y_pred for each split in the epoch
                # aggregate predictions and metrics per patient_id

                patient_ids_train = np.concatenate(patient_ids_train, axis=0)
                patient_ids_test = np.concatenate(patient_ids_test, axis=0)

                sample_weight_train = get_sampler_weights(patient_ids_train)
                sample_weight_test = get_sampler_weights(patient_ids_test)

                y_score_train = np.concatenate(y_score_train, axis=0)[:, 1]
                y_true_train == np.concatenate(y_true_train, axis=0)
                y_pred_train = (y_score_train >= threshold)*1
                
                #y_score_test = np.concatenate(y_score_test, axis=0)[:, 1]
                #y_true_test == np.concatenate(y_true_test, axis=0)
                y_pred_test = (y_score_test >= threshold)*1

                # create a clasification report of each split
                roc_auc_test = roc_auc_score(y_true_test, y_score_test)
                roc_auc_train = roc_auc_score(y_true_train, y_score_train, sample_weight=sample_weight_train)

                train_report = classification_report(y_true_train, y_pred_train,
                                                     output_dict=True, zero_division=0,
                                                     sample_weight=sample_weight_train)
                train_report['ROC AUC'] = roc_auc_train
                train_report['kfold'] = kfold
                train_report['loss'] = avg_train_loss
                train_report['epoch'] = epoch
                train_report['split'] = 'train'

                test_report = classification_report(y_true_test, y_pred_test,
                                                    output_dict=True, zero_division=0)
                test_report['ROC AUC'] = roc_auc_test
                test_report['kfold'] = kfold
                test_report['loss'] = avg_test_loss
                test_report['epoch'] = epoch
                test_report['split'] = 'test'

                train_report_str = print_classification_report(train_report)
                test_report_str = print_classification_report(test_report)

                # save train and test clasification reports into a json file
                with open(os.path.join(save_dir, f'train_metrics_{epoch}.json'), 'w') as file:
                    json.dump(train_report, file)

                with open(os.path.join(save_dir, f'test_metrics_{epoch}.json'), 'w') as file:
                    json.dump(test_report, file)

                # save a plot of the train an test loss
                train_metrics['kfold'].append(kfold)
                train_metrics['epoch'].append(epoch)
                train_metrics['train_loss'].append(avg_train_loss)
                train_metrics['test_loss'].append(avg_test_loss)
                train_metrics['train_auc'].append(roc_auc_train)
                train_metrics['test_auc'].append(roc_auc_test)
                train_metrics['train_f1'].append(train_report['macro avg']['f1-score'])
                train_metrics['test_f1'].append(test_report['macro avg']['f1-score'])
                train_metrics['train_report'].append(train_report_str.replace('\n', '<br>').replace(' ', '  '))
                train_metrics['test_report'].append(test_report_str.replace('\n', '<br>').replace(' ', '  '))

                df_loss = pd.DataFrame(train_metrics)
                df_loss = df_loss[df_loss['kfold'] == kfold]

                # early stoping
                patience = cfg_model['patience']

                #df_loss['target_metric'] = df_loss['test_auc'] * np.sqrt(df_loss['test_auc'] * df_loss['train_auc']) * np.sqrt(df_loss['test_f1'] * df_loss['train_f1'])
                df_loss['target_metric'] = df_loss['test_auc'] * df_loss['test_auc'] * np.sqrt(df_loss['test_f1'])
                df_loss['is_improvement'] = df_loss['target_metric'] >= df_loss['target_metric'].max()

                fig = plot_loss_metrics(df_loss, title=f'{arg_dataset} fold {kfold}')
                fig.write_html(os.path.join(save_dir, 'losses.html'))

                df_loss.sort_values(by='epoch', ascending=True, inplace=True)
                df_loss.reset_index(inplace=True, drop=True)
                epochs_since_improvement = epoch - df_loss.iloc[df_loss['is_improvement'].argmax()]['epoch']

                # save .pth model checkpoint
                if roc_auc_test > best_roc_auc_test:
                    best_roc_auc_test=roc_auc_test
                    save_checkpoint(model, save_dir, epoch=epoch)

                if epochs_since_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
