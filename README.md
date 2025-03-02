# Japaneese-Handwritten-images-prediction-using-FeedForwardNN-# Kuzushiji Kanji (KKanji) Classification

## Overview

This project trains a feedforward neural network (FNN) on the Kuzushiji Kanji (KKanji) dataset, which consists of 140,426 grayscale images of Kanji characters. The model is evaluated based on training and validation accuracy, with various hyperparameter configurations tested to determine optimal performance.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch torchvision matplotlib numpy seaborn scikit-learn deeplake
```

## Dataset

The Kuzushiji Kanji dataset is loaded directly using Deep Lake:

```python
import deeplake
ds = deeplake.load("hub://activeloop/kuzushiji-kanji")
```

- **Number of classes**: 3,831 (highly imbalanced)
- **Image Size**: 64x64 grayscale images
- **Training Samples**: 140,426 images

## Model Training

The training script supports:

- Different hidden layer configurations ([32, 64], [64, 128, 256], [128, 64, 32])
- Activation functions: ReLU and Sigmoid
- Optimizers: SGD, Momentum, Nesterov, RMSprop, Adam
- Batch sizes: 16, 32, 64
- Weight Initialization: Random, Xavier
- Learning rates: 1e-3, 1e-4

### To train the model, run:

```bash
python train.py
```

The script will train the model with different configurations and track accuracy over epochs.

## Evaluation

The model is evaluated on both validation and test datasets.

- Training and validation accuracy are plotted over epochs.
- The best configuration is selected based on validation accuracy.
- Test set evaluation is performed on the best model.

### To evaluate the model, run:

```bash
python evaluate.py
```

## Results & Inferences

### Training Performance by Optimizer

| Optimizer | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Test Accuracy |
| --------- | ------- | ------- | ------- | ------- | ------- | ------------- |
| SGD       | 30.82%  | 39.53%  | 51.82%  | 69.38%  | 77.28%  | 77.84%        |
| Momentum  | 92.72%  | 94.97%  | 94.46%  | 95.97%  | 95.62%  | 95.99%        |
| Nesterov  | 93.05%  | 94.85%  | 95.77%  | 95.64%  | 96.50%  | 96.82%        |
| RMSprop   | 92.92%  | 94.23%  | 94.99%  | 95.07%  | 96.22%  | 96.26%        |
| Adam      | 92.08%  | 94.21%  | 95.48%  | 96.18%  | 95.48%  | 95.54%        |

### Observations:

As Hidden layers are increased from 3 to 4,5 The performance increased.

As epochs /fitting the model increased(iterations) then loss is decreased

- Deeper networks generally perform better but may overfit.
- ReLU activation outperforms Sigmoid due to better gradient flow.
- Sigmoid activation struggles in deeper networks due to vanishing gradients.



## Conclusion

Based on extensive experimentation on the Kuzushiji Kanji dataset, the best-performing hyperparameter configurations were identified. These configurations demonstrated strong generalization and are recommended for use on similar datasets:

### 1

**Hidden Layers:** [128, 64] | **Activation:** ReLU | **Optimizer:** RMSprop\
**Test Accuracy:** 96.26%\
RMSprop adapts learning rates effectively, making it suitable for different datasets.

### 2

**Hidden Layers:** [128, 64, 32] | **Activation:** ReLU | **Optimizer:** Nesterov Momentum\
**Test Accuracy:** 96.82%\
Nesterov Momentum enhances stability and improves convergence.

### 3

**Hidden Layers:** [64, 32] | **Activation:** Sigmoid | **Optimizer:** Adam\
**Test Accuracy:** 95.54%\
Sigmoid provides smoother outputs, but ReLU often performs better in deeper networks.

##


