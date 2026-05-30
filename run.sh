#!/bin/bash
# Run baseline ZeroDiffusion experiments on core datasets.
# Best method: MSE + Interpolated Noise

mkdir -p results
export PYTHONUNBUFFERED=1

echo "Running baseline MSE+Interp experiments..."

# # ESC-50: 4 validation folds and a held out test fold
uv run python train.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold04.pickle --dataset ESC-50 --split fold0 --cls_dataset_size 40 > results/escfold0.txt
uv run python train.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold14.pickle --dataset ESC-50 --split fold1 --cls_dataset_size 40 > results/escfold1.txt
uv run python train.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold24.pickle --dataset ESC-50 --split fold2 --cls_dataset_size 40 > results/escfold2.txt
uv run python train.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold34.pickle --dataset ESC-50 --split fold3 --cls_dataset_size 40 > results/escfold3.txt
uv run python train.py ../ZeroShotESCData/pickles/esc50/ESC_50_synonyms/fold4.pickle --dataset ESC-50 --split test --cls_dataset_size 40 > results/esctest.txt

# GTZAN - music classification dataset
uv run python train.py ../ZeroShotESCData/pickles/gtzan/YAMNet_synonyms_test.pickle --dataset GTZAN --split test --cls_dataset_size 100 > results/gtzan.txt

# UrbanSound8k - small number of classes but many samples per class
uv run python train.py ../ZeroShotESCData/pickles/urbansound8k/YAMNet_synonyms_test.pickle --dataset UrbanSound8k --split test --cls_dataset_size 873 > results/urbansound8k.txt

# TAU 2019 - a scene classification dataset
uv run python train.py ../ZeroShotESCData/pickles/tau2019/YAMNet_synonyms_test.pickle --dataset TAU2019 --split test --cls_dataset_size 1440 > results/tau2019.txt

# FSC22 - split into train, val, test, less classes than ESC-50 but more than GTZAN and UrbanSound8k
uv run python train.py ../ZeroShotESCData/pickles/fsc22/YAMNet_synonyms_val.pickle --dataset FSC22 --split val --cls_dataset_size 75 > results/fscval.txt
uv run python train.py ../ZeroShotESCData/pickles/fsc22/YAMNet_synonyms_test.pickle --dataset FSC22 --split test --cls_dataset_size 75 > results/fsctest.txt

# ARCA23K-FSD - larger dataset with multiple validation folds
uv run python train.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold0.pickle --dataset ARCA23K-FSD --split fold0 --cls_dataset_size 339 > results/arcafold0.txt
uv run python train.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold1.pickle --dataset ARCA23K-FSD --split fold1 --cls_dataset_size 339 > results/arcafold1.txt
uv run python train.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold2.pickle --dataset ARCA23K-FSD --split fold2 --cls_dataset_size 339 > results/arcafold2.txt
uv run python train.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold3.pickle --dataset ARCA23K-FSD --split fold3 --cls_dataset_size 339 > results/arcafold3.txt
uv run python train.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold4.pickle --dataset ARCA23K-FSD --split fold4 --cls_dataset_size 339 > results/arcafold4.txt
uv run python train.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold5.pickle --dataset ARCA23K-FSD --split fold5 --cls_dataset_size 339 > results/arcafold5.txt
uv run python train.py ../ZeroShotESCData/pickles/arca23kfsd/YAMNet_normal_fold6.pickle --dataset ARCA23K-FSD --split test --cls_dataset_size 339 > results/arcatest.txt

echo "Experiments complete! Results saved to results/ directory."
