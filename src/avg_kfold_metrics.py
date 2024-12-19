# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:49:59 2024

@author: Mico
"""

import os
import numpy as np
import pandas as pd
import json
import plotly.express as px
import warnings
warnings.simplefilter(action='ignore')

def harmonic_mean(metric_a, metric_b):
    harmonic = (2* metric_a * metric_b) / (metric_a + metric_b)
    return harmonic

def geometric_mean(metric_a, metric_b, metric_c):
    harmonic =  np.cbrt(metric_a * metric_b, metric_c)
    return harmonic

def load_json(json_path: str):
    """ Load a json file from path """
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    return data


if __name__ == "__main__":
    folder = 'petct_online_rad_dino_thr0'

    metrics_per_fold = []
    metrics_sumary = {}
    metrics_sumary['Dataset'] = []
    metrics_sumary['Split'] = []
    metrics_sumary['Model'] = []
    metrics_sumary['Modality'] = []

    metrics_sumary['Accuracy'] = []
    metrics_sumary['AUC'] = []
    metrics_sumary['Precision'] = []
    metrics_sumary['Recall'] = []

    metrics_sumary['Specificity'] = []
    metrics_sumary['Sensivity'] = []

    metrics_sumary['Best Kfold'] = []
    metrics_sumary['Best Epoch'] = []
    experiments = os.listdir(os.path.join('..', 'models', folder))
    #experiments = ['medsam_transformer_santa_maria', 'medsam_transformer_stanford']
    
    os.mkdir(os.path.join('..', 'plots', 'training',folder))
    for experiment in experiments:
        modalities = os.listdir(os.path.join('..', 'models', folder, experiment))
        #modalities = ['petct']
        for modality in modalities:
            modality_folder = os.path.join('..', 'models', folder, experiment, modality)
            json_metrics = []
            folds = [int(fn.split('_')[-1]) for fn in os.listdir(modality_folder) if 'kfold' in fn]
            for k in folds:
                kfold_dir = os.path.join('..', 'models', folder, experiment, modality, f'kfold_{k}')
                if os.path.exists(kfold_dir):
                    json_fns = [fn for fn in os.listdir(kfold_dir) if '.json' in fn]
                    for fn in json_fns:
                        json_path = os.path.join(kfold_dir, fn)
                        data = load_json(json_path)
                        epoch = int(fn.split('.json')[0].split('_')[-1])
                        data['epoch'] = epoch
                        if 'test' in fn:
                            data['split'] = 'test'
                        else:
                            data['split'] = 'train'
                        json_metrics.append(data)

            # save the training loss and accuracy curves with an animation slider to select the Kfold
            df_metrics = [pd.DataFrame(split_metrics).reset_index(drop=False) for split_metrics in json_metrics]
            df_metrics = pd.concat(df_metrics, axis=0)
            df_metrics = df_metrics.rename(columns={'0.0': '0', '1.0': '1'})
            df_metrics_group = df_metrics.set_index(['kfold', 'epoch'])
            fig = px.line(df_metrics[df_metrics['index'] == 'recall'].sort_values('epoch'),
                          x='epoch', y='loss', color='split', animation_frame='kfold',
                          title=experiment, markers=True)
            fig.update_yaxes(range=[0, df_metrics['loss'].max()+0.1])
            fig.update_xaxes(range=[0, df_metrics['epoch'].max()+1])
            fig.write_html(os.path.join('..', 'plots', 'training',folder, f'{experiment}-{modality}-training_loss.html'))

            fig = px.line(df_metrics[df_metrics['index'] == 'recall'].sort_values('epoch'),
                          x='epoch', y='ROC AUC', color='split', animation_frame='kfold',
                          title=experiment, markers=True)
            fig.update_yaxes(range=[0, df_metrics['accuracy'].max()+0.1])
            fig.update_xaxes(range=[0, df_metrics['epoch'].max()+1])
            fig.write_html(os.path.join('..', 'plots', 'training',folder, f'{experiment}-{modality}-training_auc.html'))

            # select the best model of each kfold based on a target metric
            df_best = df_metrics.copy()
            df_best = df_best[df_best['index'] == 'f1-score']
            df_train = df_best[df_best['split'] == 'train']
            df_test = df_best[df_best['split'] == 'test']
            df_best = df_test
            df_best['target_metric'] = df_test['ROC AUC']
            df_best = df_best.sort_values('target_metric', ascending=False)
            df_best_kfolds = df_best.groupby('kfold').first()
            dataset = ' '.join(experiment.split('_')[2:])
            df_best_kfolds['dataset'] = dataset
            df_best_kfolds['experiment'] = experiment
            df_best_kfolds['modality'] = modality
            metrics_per_fold.append(df_best_kfolds)
            
            columns=["0","1","accuracy","macro avg","weighted avg","ROC AUC","loss"  ,"epoch","target_metric"]
            
            model_avg_metrics = df_best_kfolds[columns].mean(axis=0)
            print(f'\n---------------{experiment} {modality}---------------')
            print(df_best_kfolds.round(3).T)
            best_metrics = []
            for best_k, row in df_best_kfolds.iterrows():
                best_epoch = row['epoch']
                best_metrics.append(df_metrics_group.loc[(best_k, best_epoch)])
                best_metrics[-1]['target_metric'] = row['target_metric']

            best_metrics = pd.concat(best_metrics, axis=0)
            #best_metrics.round(3).to_csv('best_metrics.csv')
            # mean across all kfolds of the best models
            model_avg_metrics = best_metrics.groupby(['split', 'index']).mean()
            model_std_metrics = best_metrics.groupby(['split', 'index']).std()   # TODO: show the std of each metric

            print("lol",best_metrics.groupby(level=[0,1])['target_metric'].mean())
            combined_metrics = model_avg_metrics.copy()
            for column in combined_metrics.columns:
                avg = model_avg_metrics[column]
                std = model_std_metrics[column]
                combined_metrics[column] = avg.map('{:,.2f}'.format) + " ± " + std.map('{:,.2f}'.format)
            if df_best_kfolds.shape[0] > 1:
                best_k = best_metrics.groupby(level=[0,1])['target_metric'].mean().argmax()
                best_epoch = best_metrics.loc[best_k].index[0]
            else:
                best_k, best_epoch = best_metrics.iloc[best_metrics['target_metric'].argmax()].name
            # Store the kfold avg metrics of each 'Dataset', 'Model', 'Modality', 'Split'
            for split in ['train', 'test']:
                
                auc = combined_metrics.loc[split, 'recall']['ROC AUC']
                accuracy = combined_metrics.loc[split, 'recall']['accuracy']
                recall_neg = combined_metrics.loc[split, 'recall']['0']
                precision = combined_metrics.loc[split, 'precision']['1']
                recall = combined_metrics.loc[split, 'recall']['1']
                model_name = ' '.join(experiment.split('_')[0:2])
                dataset = ' '.join(experiment.split('_')[2:])

                metrics_sumary['Dataset'].append(dataset)
                metrics_sumary['Split'].append(split)
                metrics_sumary['Model'].append(model_name)
                metrics_sumary['Modality'].append(modality)

                metrics_sumary['Accuracy'].append(accuracy)
                metrics_sumary['AUC'].append(auc)
                metrics_sumary['Precision'].append(precision)
                metrics_sumary['Recall'].append(recall)

                metrics_sumary['Specificity'].append(recall)
                metrics_sumary['Sensivity'].append(recall_neg)

                metrics_sumary['Best Kfold'].append(best_k)
                metrics_sumary['Best Epoch'].append(best_epoch)

            print(f'\n{dataset} {model_name} {modality}')
            print(model_avg_metrics[['0', '1', 'accuracy', 'ROC AUC', 'loss']].round(2))

    print("Results of ", folder)
    
    df_metrics_sumary = pd.DataFrame(metrics_sumary)
    sumary_index = ['Dataset', 'Model', 'Modality', 'Split']
    df_metrics_sumary = df_metrics_sumary.set_index(sumary_index).sort_index()
    df_metrics_sumary = df_metrics_sumary.sort_index(level=[0, 1, 2, 3], ascending=[True, True, True, False])
    df_metrics_sumary.round(3).to_csv(os.path.join('..', 'metrics', f'{folder}_metrics_sumary.csv'), encoding='utf-8-sig')
    print(df_metrics_sumary.round(3).T)

    metrics_per_fold = pd.concat(metrics_per_fold, axis=0)
    metrics_per_fold.round(3).to_csv(os.path.join('..', 'metrics', f'{folder}_metrics_per_fold.csv'), encoding='utf-8-sig')
