# Dataset Curation

Two scripts, `load_esc50.py` and `load_fsc22.py` are provided to generate pickles of the datasets with audio embeddings and word embeddings. 

The main arguments to parse are `save_path` and `model_path`. `save_path` is the full path of the pickle you want to create, eg `ESC_50_synonyms/fold04.pickle`. `model_path` is the path to the audio embedding model, which will determine which partition of the dataset this pickle is for. 

Both ESC-50 and FSC22 need to be added to this folder, as well as word embedding vectors.

ESC-50 can be obtained using the following command-line argument using git. 

```
git clone https://github.com/karolpiczak/ESC-50
```

FSC22 can be downloaded at https://www.kaggle.com/datasets/irmiot22/fsc22-dataset.

Word embedding vectors for Word2Vec trained on GoogleNews are downloaded at https://code.google.com/archive/p/word2vec/, the direct download link is https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing.

Examples of creating the pickles:

```
python load_esc50.py ./ESC_50_synonyms/fold04.pickle ../audio_embeddings/checkpoint/YAMNet_ESC_50_fold04.pt 
```

```
python load_fsc22.py ./FSC22_synonyms.pickle ../audio_embeddings/checkpoint/YAMNet_FSC22.pt
``` 