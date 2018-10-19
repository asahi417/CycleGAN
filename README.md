# Cycle GAN
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/WassersteinGAN/blob/master/LICENSE)

Tensorflow implementation of Cycle GAN.
- Dataset are converted to TFRecord format
- Tensorboard visualization (metrics, output generated image)

## Get started
Clone repository
```
git clone https://github.com/asahi417/CycleGAN
cd CycleGAN
pip install .
```

**setup dataset** 

```bash
mkdir datasets
bash bin/download_cyclegan_dataset.sh monet2photo
```
You can choose dataset from following list: monet2photo, horse2zebra, vangogh2photo, ukiyoe2photo, cezanne2photo

***convert dataset to TFRecord format***

```bash
python bin/build_tfrecord.py --data monet2photo 
```

***train cycle GAN***

```bash
python bin/train.py --data monet2photo -e 200
```

You can tune hyperparameter by modifying [this file](./bin/hyperparameter.toml).

***visualize by tensorboard***

```bash
tensorboard logdir=./checkpoint --port 555
```



