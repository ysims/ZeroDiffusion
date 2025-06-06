mkdir results
export PYTHONUNBUFFERED=1
python3 train.py ../pickles/urbansound8k/YAMNet_synonyms_test.pickle --dataset UrbanSound8k --split test --cls_dataset_size 873 > results/urbansound8k.txt
python3 train.py ../pickles/gtzan/YAMNet_synonyms_test.pickle --dataset GTZAN --split test --cls_dataset_size 100 > results/gtzan.txt
python3 train.py ../pickles/esc50/ESC_50_synonyms/fold04.pickle --dataset ESC-50 --split fold0 --cls_dataset_size 40 > results/escfold0.txt
python3 train.py ../pickles/esc50/ESC_50_synonyms/fold14.pickle --dataset ESC-50 --split fold1 --cls_dataset_size 40 > results/escfold1.txt
python3 train.py ../pickles/esc50/ESC_50_synonyms/fold24.pickle --dataset ESC-50 --split fold2 --cls_dataset_size 40 > results/escfold2.txt
python3 train.py ../pickles/esc50/ESC_50_synonyms/fold34.pickle --dataset ESC-50 --split fold3 --cls_dataset_size 40 > results/escfold3.txt
python3 train.py ../pickles/esc50/ESC_50_synonyms/fold4.pickle --dataset ESC-50 --split test --cls_dataset_size 40 > results/esctest.txt
python3 train.py ../pickles/fsc22/YAMNet_synonyms_val.pickle --dataset FSC22 --split val --cls_dataset_size 75 > results/fscval.txt
python3 train.py ../pickles/fsc22/YAMNet_synonyms_test.pickle --dataset FSC22 --split test --cls_dataset_size 75 > results/fsctest.txt
python3 train.py ../pickles/arca23kfsd/YAMNet_normal_fold0.pickle --dataset ARCA23K-FSD --split fold0 --cls_dataset_size 339 > results/arcafold0.txt
python3 train.py ../pickles/arca23kfsd/YAMNet_normal_fold1.pickle --dataset ARCA23K-FSD --split fold1 --cls_dataset_size 339 > results/arcafold1.txt
python3 train.py ../pickles/arca23kfsd/YAMNet_normal_fold2.pickle --dataset ARCA23K-FSD --split fold2 --cls_dataset_size 339 > results/arcafold2.txt
python3 train.py ../pickles/arca23kfsd/YAMNet_normal_fold3.pickle --dataset ARCA23K-FSD --split fold3 --cls_dataset_size 339 > results/arcafold3.txt
python3 train.py ../pickles/arca23kfsd/YAMNet_normal_fold4.pickle --dataset ARCA23K-FSD --split fold4 --cls_dataset_size 339 > results/arcafold4.txt
python3 train.py ../pickles/arca23kfsd/YAMNet_normal_fold5.pickle --dataset ARCA23K-FSD --split fold5 --cls_dataset_size 339 > results/arcafold5.txt
python3 train.py ../pickles/arca23kfsd/YAMNet_normal_fold6.pickle --dataset ARCA23K-FSD --split test --cls_dataset_size 339 > results/arcatest.txt
