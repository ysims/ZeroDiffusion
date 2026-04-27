# ZeroDiffusion

A zero-shot learning diffusion method.

Before training, train an audio embedding model in the [audio embedding folder](audio_embeddings/README.md). 

Next, create a pickle of the embeddings in the [data folder](data/README.md).

Now you can run the ZeroDiffusion training.

Example:

```
python train.py ../data/ESC_50_synonyms/fold04.pickle --dataset ESC-50 --split fold0
```

## Tried Variants

The repo has been simplified to the MSE + interpolation-based denoising autoencoder baseline because the following variants were tried and did not improve performance in practice:

- Additive noise instead of interpolation noise
- L1 reconstruction loss instead of MSE
- LayerNorm in the denoising autoencoder
- Contrastive DAE variants
- Conditional VAE variants
- Adaptive-noise variants

The current codebase keeps the best-performing path only: MSE loss with interpolation-based noise in the denoising autoencoder.