# Audio Embedding Training for Zero-Shot Learning

This folder contains Python code using PyTorch to train artificial neural network models to produce audio embeddings for the purpose of zero-shot learning.

See setup.py for a list of options.

Examples:

```
python train.py --split fold04 --patience 200
```

```
python train.py --dataset FSC22
```

In this study, YAMNet was the chosen model. FSC22 and ESC-50 were used. ESC-50 was used with fold4, fold04, fold14, fold24 and fold34 for four-fold cross validation with the last random partition held out as a test set. Patience was set to 200. Other parameters were default values. 

## VGGish

[VGGish is a model created by Google from VGG.](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)

This is a PyTorch implementation designed for zero-shot learning.

It uses [VGG Configuration A with 11 weight layers](https://arxiv.org/pdf/1409.1556.pdf). The last group of convolutional and maxpool layers were dropped, so there are four groups of convolution/maxpool layers. The last 1000-wide fully connected layer is changed to 128-wide and is the embedding layer.

## YAMNet

The YAMNet implementation is taken mostly from [this repository](https://github.com/w-hc/torch_audioset), based on Google's YAMNet implementation in TensorFlow. This architecture has been slightly changed to allow for retrieval of a 128-dimensional audio embedding vector during inference, similar to VGGish.

- The last two separable convolutional layers are removed 
- Three fully connected layers similar to the final layers of VGGish are added to the end. The first takes the last convolutional layer to a fully connected layer of size 4096. The second is a fully connected layer from the previous layer to 4096 nodes. The final fully connected layer is of size 128, the size of the audio embedding. 
- Two more fully connected layers are used during training for classification. The first layer goes to 100 nodes and the size of the final layer is equal to the number of classes being classified during training.

## Other Sources

If you aren't interested in ZSL or would like to find some different PyTorch code for these models, [head over to this repository](https://github.com/w-hc/torch_audioset). 
