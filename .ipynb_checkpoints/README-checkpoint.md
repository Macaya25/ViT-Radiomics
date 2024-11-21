# ViT-Deep-Radiomics

## Installation
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. Enter the MedSAM folder `cd src/MedSAM` and run `pip install -e .`
3. Enter the main folder `cd ../..` and run `pip install -r requirements.txt`

## Dataset
To get started, download the [CSM and Stanford Radiogenomics PET-CT datasets](https://drive.google.com/drive/folders/12v969-5JwiREUnyZno69H0iVOkdgSjqj?usp=sharing) and place them in the `data/lung_radiomics/` directory as follows:
```
data/lung_radiomics/
├── lung_radiomics_datasets.csv
└── lung_radiomics_datasets.hdf5
```

## Train Models
Model training is based on precalculated patch embeddings. Training parameters such as the number of layers and learning rate can be found in `conf/parameters_models.yaml`, while the model architectures are defined in `src/model_archs.py`. Use the `train_models.py` script to train the models:

## Backbones architectures
The differents backbones to select are: "medsam", "medsam_normvit" , "medsam_lora" , "rad_dino" and ~~"medical_mae"~~

```bash
cd ./src/
python ./train_models.py --arch "transformer" --backbone "medsam" --dataset "stanford" --modality "petct" --gpu 0 --loss "crossmodal" --experiment name_exp
python ./train_models.py --arch "transformer" --backbone "medsam" --dataset "santa_maria" --modality "petct" --gpu 0 --loss "crossmodal" --experiment name_exp
```

## Evaluate Metrics
During training and evaluation, a `.json` file containing relevant metrics is created for each epoch. To aggregate these metrics across folds and modalities, use the `avg_kfold_metrics.py` script, which will generate a `*_metrics_summary.csv` file for comparison:

```bash
python avg_kfold_metrics.py
```