# MNIST-Digit-Generator-using-GAN

A Generative Adversarial Network (GAN) built with TensorFlow/Keras that learns to generate realistic handwritten digits from random noise, trained on the MNIST dataset.

## Overview

A GAN consists of two neural networks trained against each other:

- **Generator**: takes a random 100-dimensional noise vector and produces a 28×28 grayscale image that looks like a handwritten digit.
- **Discriminator**: takes an image and predicts whether it is a real MNIST digit or a fake one produced by the generator.

During training, the discriminator gets better at spotting fakes, and the generator gets better at fooling it. Over time, the generator learns to produce convincing digits.

## Features

- DCGAN-style architecture using transposed convolutions and batch normalization
- Custom training loop with `tf.GradientTape` and `@tf.function` for speed
- Efficient input pipeline with `tf.data` (shuffle, batch, prefetch)
- Image grid visualization of generated digits after training

## Model Architecture

### Generator

| Layer | Output Shape | Notes |
|-------|--------------|-------|
| Input (noise) | 100 | Random normal vector |
| Dense + BatchNorm + LeakyReLU | 7×7×256 | `use_bias=False` |
| Conv2DTranspose (128, 5×5, stride 1) | 7×7×128 | BatchNorm + LeakyReLU |
| Conv2DTranspose (64, 5×5, stride 2) | 14×14×64 | BatchNorm + LeakyReLU |
| Conv2DTranspose (1, 5×5, stride 2) | 28×28×1 | `tanh` activation |

### Discriminator

| Layer | Output Shape | Notes |
|-------|--------------|-------|
| Input image | 28×28×1 | |
| Conv2D (64, 5×5, stride 2) | 14×14×64 | LeakyReLU + Dropout(0.3) |
| Conv2D (128, 5×5, stride 2) | 7×7×128 | LeakyReLU + Dropout(0.3) |
| Flatten | 6272 | |
| Dense (1) | 1 | `sigmoid` activation |

## Training Details

| Setting | Value |
|---------|-------|
| Dataset | MNIST (60,000 training images) |
| Image preprocessing | Normalized to the range [-1, 1] |
| Batch size | 128 |
| Noise dimension | 100 |
| Optimizer | Adam (learning rate 1e-4) for both networks |
| Loss function | Binary Cross-Entropy |
| Epochs | 10 (adjustable) |

**Loss logic:**

- The discriminator is trained to label real images as `1` and generated images as `0`.
- The generator is trained to make the discriminator label its fake images as `1`.

## Requirements

- Python 3.9+ (developed with Python 3.11)
- TensorFlow 2.x
- NumPy
- Matplotlib

Install the dependencies:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

1. Clone the repository:

   ```bash
   git clone <your-repository-url>
   cd <your-project-folder>
   ```

2. Open the notebook:

   ```bash
   jupyter notebook
   ```

3. Run all cells. The notebook will:
   1. Download and preprocess the MNIST dataset
   2. Build the generator and discriminator
   3. Train the GAN and print the generator and discriminator loss each epoch
   4. Display a 4×4 grid of generated digits

To train for longer, change the `epochs` argument in the final cell:

```python
train(mnist_data, epochs=50)
```

## Example Output

During training, you will see logs like:

```
Epoch 1, Gen Loss: ..., Disc Loss: ...
Epoch 2, Gen Loss: ..., Disc Loss: ...
```

After training, a 4×4 grid of generated handwritten digits is displayed.

<!-- Add a screenshot of your generated digits here, e.g.: -->
<!-- ![Generated Digits](images/generated_digits.png) -->

## Project Structure

```
.
├── gan_mnist.ipynb    # Main notebook (model, training loop, visualization)
└── README.md
```

## How It Works

1. Random noise is fed to the generator to create a batch of fake images.
2. The discriminator scores both real MNIST images and the fake images.
3. Generator loss and discriminator loss are calculated.
4. Gradients are computed separately for each network and applied with their own Adam optimizers.
5. Steps 1–4 repeat for every batch in every epoch.

## Possible Improvements

- Save generated image samples and model checkpoints during training
- Plot generator and discriminator loss curves
- Use a fixed noise seed to track progress on the same inputs across epochs
- Add label conditioning (Conditional GAN) to generate a specific digit on demand
- Try other datasets such as Fashion-MNIST or CIFAR-10
- Experiment with WGAN or WGAN-GP for more stable training


## License

This project is licensed under the [MIT License](LICENSE).
