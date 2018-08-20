# CycleGAN

## Environment
- python 3.6
- TensorFlow 1.9.0

## Usage

1. clone this repo: `git clone https://github.com/DTennant/CycleGAN.git && cd CycleGAN`
2. make a dir for dataset: `mkdir datasets && cd datasets`
3. download the horse2zebra dataset from my [BaiduYun](https://pan.baidu.com/s/1hLuoP523-1ZtXDNSUDqEsA) and unzip it , or you can manually collect a dataset
4. start training(only support one GPU): `CUDA_VISIBLE_DEVICES=0 python main.py`

```
usage: main.py [-h] [--mode {train,test}] [--batch_size BATCH_SIZE]
               [--dataset DATASET] [--load_size LOAD_SIZE]
               [--crop_size CROP_SIZE] [--epoch EPOCH] [--lr LR]
               [--beta1 BETA1] [--gpu GPU] [--sample_interval SAMPLE_INTERVAL]
               [--save_interval SAVE_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,test}
  --batch_size BATCH_SIZE
                        number of batch_size
  --dataset DATASET     the dataset to use
  --load_size LOAD_SIZE
                        resize the input img to this size
  --crop_size CROP_SIZE
                        crop the resized img
  --epoch EPOCH         the number of epoches
  --lr LR               init lr
  --beta1 BETA1         beta1 in AdamOptimizer
  --gpu GPU             use which gpu to compute
  --sample_interval SAMPLE_INTERVAL
                        sample during # epoch
  --save_interval SAVE_INTERVAL
                        save during # epoch
```

## Sample Images

To be uploaded (still training)