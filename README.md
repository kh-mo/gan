# Generative Adversarial Nets

This repository is implemented by Pytorch

**GAN: [Paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

## Result

## Architecture

## Loss

## Getting Start
### Download data
- Download in current space
- Input : MNIST URL
- Output : raw_data folder, MNIST file
```shell
python dataset_download.py \
    --download_path = .
```

### Preprocessing
- training_set = {60000 * 28 * 28 size, 0~255 value, 0~9 label}, test_set = {10000 * 28 * 28 size, 0~255 value, 0~9 label} 
- Input : raw_data folder, MNIST file
- Output : preprocess folder, test.pt, training.pt
```shell
python preprocessing.py
```

### modeling
- Input : preprocess folder, test.pt, training.pt
- Output : 