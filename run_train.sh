#!/bin/bash

cd ./src/
# python ./train_models.py --arch "transformer" -b "none" -r "medsam" --dataset "stanford" --modality "pet" --gpu 1
# python ./train_models.py --arch "transformer" -b "none" --dataset "stanford" --modality "petct" --gpu 1

python ./train_models.py --arch "transformer" -b "none" -r "medsam" -wt True --dataset "stanford" --modality "pet" --gpu 1
# python ./train_models.py --arch "transformer" -b "none" -r "medsam" -wt True --dataset "stanford" --modality "petct" --gpu 1

# python ./train_models.py --arch "conv" --dataset "stanford" --modality "pet" --gpu 1 --loss "focal"
# python ./train_models.py --arch "conv" --dataset "stanford" --modality "ct" --gpu 1 --loss "focal"
# python ./train_models.py --arch "conv" --dataset "santa_maria" --modality "pet" --gpu 1 --loss "focal"
# python ./train_models.py --arch "conv" --dataset "santa_maria" --modality "ct" --gpu 1 --loss "focal"
# python ./train_models.py --arch "transformer" --dataset "stanford" --modality "pet" --gpu 1 --loss "focal"
# python ./train_models.py --arch "transformer" --dataset "stanford" --modality "ct" --gpu 1 --loss "focal"
# python ./train_models.py --arch "transformer" --dataset "santa_maria" --modality "pet" --gpu 1 --loss "focal"
# python ./train_models.py --arch "transformer" --dataset "santa_maria" --modality "ct" --gpu 1 --loss "focal"
# python ./train_models.py --arch "transformer" --dataset "stanford" --modality "petct" --gpu 1 --loss "crossmodal"
# python ./train_models.py --arch "transformer" --dataset "santa_maria" --modality "petct" --gpu 1 --loss "crossmodal"