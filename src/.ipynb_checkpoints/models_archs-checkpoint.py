# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:22:10 2024

@author: Mico
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def save_checkpoint(model, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epoch_str = str(epoch).zfill(4)
    model_path = os.path.join(save_dir, f'model_epoch_{epoch_str}.pth')
    model_path = os.path.join(save_dir, f'best_model_epoch.pth')
    save(model, model_path)


def load_checkpoint(model, save_dir, epoch):
    epoch_str = str(epoch).zfill(4)
    model_path = os.path.join(save_dir, f'model_epoch_{epoch_str}.pth')
    return load(model, model_path)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


class TransformerNoduleBimodalClassifier(nn.Module):
    def __init__(self, input_dim,
                 mlp_ratio_ct, mlp_ratio_pet,
                 num_heads_ct, num_heads_pet,
                 num_layers_ct, num_layers_pet,
                 froze_trans,num_classes):
        super(TransformerNoduleBimodalClassifier, self).__init__()

        self.transformer_encoder_ct = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim,
                                                                                       dim_feedforward=int(mlp_ratio_ct*input_dim),
                                                                                       nhead=num_heads_ct,
                                                                                       activation="gelu",
                                                                                       batch_first=True,
                                                                                       dropout=0.5),
                                                            num_layers=num_layers_ct)
        self.transformer_encoder_pet = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim,
                                                                                        dim_feedforward=int(mlp_ratio_pet*input_dim),
                                                                                        nhead=num_heads_pet,
                                                                                        activation="gelu",
                                                                                        batch_first=True,
                                                                                        dropout=0.5),
                                                             num_layers=num_layers_pet)

        self.norm_ct = nn.LayerNorm(input_dim)
        self.norm_pet = nn.LayerNorm(input_dim)

        self.cls_token_ct = nn.Parameter(torch.randn(1, 1, input_dim))
        self.cls_token_pet = nn.Parameter(torch.randn(1, 1, input_dim))

        self.classifier_ct = MLPLayer(input_dim, input_dim*2, num_classes, dropout_rate=0.1)
        self.classifier_pet = MLPLayer(input_dim, input_dim*2, num_classes, dropout_rate=0.1)

        self.projection_petct = MLPLayer(input_dim*2, input_dim, input_dim, dropout_rate=0.1)

        self.cross_attention_ct = CrossAttentionLayer(input_dim, num_heads_ct)
        self.cross_attention_pet = CrossAttentionLayer(input_dim, num_heads_ct)
        self.classifier_petct = MLPLayer(input_dim, input_dim*2, num_classes, dropout_rate=0.1)

        self.froze_trans=froze_trans

    def forward(self, x_ct=None, x_pet=None):
        use_ct = x_ct is not None
        use_pet = x_pet is not None
        assert use_ct or use_pet, "At least one modality should be used"
        # add cls token and norm to each pet ct seq
        if use_ct:
            batch, seq_len, feature_dim = x_ct.shape
            if self.froze_trans:
                with torch.no_grad():
                    x_ct = torch.cat([self.cls_token_ct.repeat(batch, 1, 1), x_ct], dim=1)
                    x_ct = self.norm_ct(x_ct)
                    x_ct = self.transformer_encoder_ct(x_ct)
            else:
                x_ct = torch.cat([self.cls_token_ct.repeat(batch, 1, 1), x_ct], dim=1)
                x_ct = self.norm_ct(x_ct)
                x_ct = self.transformer_encoder_ct(x_ct)
            ct_cls_token = x_ct[:, 0, :]
        else:
            ct_cls_token = self.cls_token_ct.repeat(1, 1, 1)

        if use_pet:
            batch, seq_len, feature_dim = x_pet.shape
            if self.froze_trans:
                with torch.no_grad():
                    x_pet = torch.cat([self.cls_token_pet.repeat(batch, 1, 1), x_pet], dim=1)
                    x_pet = self.norm_pet(x_pet)
                    x_pet = self.transformer_encoder_pet(x_pet)
            else:
                x_pet = torch.cat([self.cls_token_pet.repeat(batch, 1, 1), x_pet], dim=1)
                x_pet = self.norm_pet(x_pet)
                x_pet = self.transformer_encoder_pet(x_pet)
            pet_cls_token = x_pet[:, 0, :]
        else:
            pet_cls_token = self.cls_token_pet.repeat(1, 1, 1)

        # cross attention between pet-ct and ct-pet
        if use_ct and use_pet:
            x_ct_attn = self.cross_attention_ct(query=x_ct, key=x_pet, value=x_pet)
            x_pet_attn = self.cross_attention_pet(query=x_pet, key=x_ct, value=x_ct)
            ct_cls_token = x_ct_attn[:, 0, :]
            pet_cls_token = x_pet_attn[:, 0, :]

            logits_ct = self.classifier_ct(ct_cls_token)
            logits_pet = self.classifier_pet(pet_cls_token)

            petct_cls_token = torch.cat([ct_cls_token, pet_cls_token], dim=1)
            petct_cls_token = self.projection_petct(petct_cls_token)
            logits_petct = self.classifier_petct(petct_cls_token)

        elif use_ct:
            logits_ct = self.classifier_ct(ct_cls_token)
            logits_pet = logits_ct
            logits_petct = logits_ct
            petct_cls_token = ct_cls_token
        else:
            logits_pet = self.classifier_pet(pet_cls_token)
            logits_ct = logits_pet
            logits_petct = logits_pet
            petct_cls_token = pet_cls_token

        return logits_petct, petct_cls_token, logits_ct, logits_pet

#Modified
class TransformerNoduleClassifier(nn.Module):
    def __init__(self, input_dim, dim_feedforward, num_heads, num_classes, num_layers,):
        super(TransformerNoduleClassifier, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   dim_feedforward=dim_feedforward,
                                                   nhead=num_heads,
                                                   activation="gelu",
                                                   batch_first=True,
                                                   dropout=0.1)
        self.norm = nn.LayerNorm(input_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.cross_attention = CrossAttentionLayer(input_dim, num_heads) #added
        self.classifier_prev = MLPLayer(input_dim, input_dim*2, num_classes, dropout_rate=0.1)
        self.classifier_final = MLPLayer(input_dim, input_dim*2, num_classes, dropout_rate=0.1)
        
    def forward(self, x):
        batch, seq_len, feature_dim = x.shape
        cls_token = self.cls_token.repeat(batch, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.norm(x)
        x = self.transformer_encoder(x)
        ###added
        cls_token = x[:, 0, :]    
        logits_prev = self.classifier_prev(cls_token)

        x_attn = self.cross_attention(query=x, key=x, value=x)
                
        logits_final = self.classifier_final(x_attn[:, 0, :])
        ###added
        return logits_final, logits_prev


class NoduleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, div=2):
        super(NoduleClassifier, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_dim, out_channels=input_dim//div, kernel_size=3, padding=1)
        self.se1 = SELayer(input_dim//div)
        self.conv2 = nn.Conv3d(in_channels=input_dim//div, out_channels=input_dim//(div*div), kernel_size=3, padding=1)
        self.se2 = SELayer(input_dim//(div*div))

        self.fc1 = nn.Linear(input_dim//(div*div), input_dim)
        self.classifier = MLPLayer(input_dim, input_dim*2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.se1(x)
        x = F.relu(self.conv2(x))
        x = self.se2(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        return self.classifier(x), x


class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        # query = [batch size, query len, features]
        # key, value = [batch size, key/value len, features]
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        return attn_output


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_features, out_features, dropout_rate=0.1):
        super(MLPLayer, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_features, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
