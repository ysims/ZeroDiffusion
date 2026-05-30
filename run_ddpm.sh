#!/bin/bash
# Run DDPM experiments on core datasets.
# DDPM (Denoising Diffusion Probabilistic Models) variant of ZeroDiffusion

mkdir -p results/ddpm
export PYTHONUNBUFFERED=1

echo "Running DDPM experiments..."

# ESC-50: 4 validation folds and a held out test fold
echo "Running DDPM on ESC-50..."
uv run python train_ddpm.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold04.pickle --dataset ESC-50 --split fold0 --cls_dataset_size 40 > results/ddpm/escfold0.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold14.pickle --dataset ESC-50 --split fold1 --cls_dataset_size 40 > results/ddpm/escfold1.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold24.pickle --dataset ESC-50 --split fold2 --cls_dataset_size 40 > results/ddpm/escfold2.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold34.pickle --dataset ESC-50 --split fold3 --cls_dataset_size 40 > results/ddpm/escfold3.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold4.pickle --dataset ESC-50 --split test --cls_dataset_size 40 > results/ddpm/esctest.txt

# GTZAN - music classification dataset
echo "Running DDPM on GTZAN..."
uv run python train_ddpm.py ../ZeroShotESCData/pickles/gtzan/YAMNet_synonyms_test.pickle --dataset GTZAN --split test --cls_dataset_size 100 > results/ddpm/gtzan.txt

# UrbanSound8k - small number of classes but many samples per class
echo "Running DDPM on UrbanSound8k..."
uv run python train_ddpm.py ../ZeroShotESCData/pickles/urbansound8k/YAMNet_synonyms_test.pickle --dataset UrbanSound8k --split test --cls_dataset_size 873 > results/ddpm/urbansound8k.txt

# TAU 2019 - a scene classification dataset
echo "Running DDPM on TAU 2019..."
uv run python train_ddpm.py ../ZeroShotESCData/pickles/tau2019/YAMNet_synonyms_test.pickle --dataset TAU2019 --split test --cls_dataset_size 1440 > results/ddpm/tau2019.txt

# FSC22 - split into train, val, test, less classes than ESC-50 but more than GTZAN and UrbanSound8k
echo "Running DDPM on FSC22..."
uv run python train_ddpm.py ../ZeroShotESCData/pickles/fsc22/YAMNet_synonyms_val.pickle --dataset FSC22 --split val --cls_dataset_size 75 > results/ddpm/fscval.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/fsc22/YAMNet_synonyms_test.pickle --dataset FSC22 --split test --cls_dataset_size 75 > results/ddpm/fsctest.txt

# ARCA23K-FSD - larger dataset with multiple validation folds
echo "Running DDPM on ARCA23K-FSD..."
uv run python train_ddpm.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold0.pickle --dataset ARCA23K-FSD --split fold0 --cls_dataset_size 339 > results/ddpm/arcafold0.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold1.pickle --dataset ARCA23K-FSD --split fold1 --cls_dataset_size 339 > results/ddpm/arcafold1.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold2.pickle --dataset ARCA23K-FSD --split fold2 --cls_dataset_size 339 > results/ddpm/arcafold2.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold3.pickle --dataset ARCA23K-FSD --split fold3 --cls_dataset_size 339 > results/ddpm/arcafold3.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold4.pickle --dataset ARCA23K-FSD --split fold4 --cls_dataset_size 339 > results/ddpm/arcafold4.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold5.pickle --dataset ARCA23K-FSD --split fold5 --cls_dataset_size 339 > results/ddpm/arcafold5.txt
uv run python train_ddpm.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold6.pickle --dataset ARCA23K-FSD --split test --cls_dataset_size 339 > results/ddpm/arcatest.txt

echo "DDPM experiments complete! Results saved to results/ddpm/ directory."
