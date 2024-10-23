# ZeroDiffusion

A zero-shot learning diffusion method.

Before training, train an audio embedding model in the [audio embedding folder](audio_embeddings/README.md). 

Next, create a pickle of the embeddings in the [data folder](data/README.md).

Now you can run the ZeroDiffusion training.

Example:

```
python train.py ../data/ESC_50_synonyms/fold04.pickle --dataset ESC-50 --split fold0
```