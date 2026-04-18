# Wasserstein GAN for Galaxy Image Generation (Galaxy Zoo)

This project implements a state-of-the-art **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** to synthesize high-resolution representations of galaxies based on the *Galaxy Zoo 2* dataset.

## 🌌 Abstract

Galaxy morphological classification is a long-standing challenge in astrophysics. Generating synthetic galaxies allows us to augment heavily imbalanced morphological datasets, improve robust feature extractors, and understand deep space image characteristics. Unlike standard DCGAN architectures, which are susceptible to *mode collapse* and vanishing gradients, this architecture employs the robust **Wasserstein Distance** with a **Gradient Penalty** to enforce the Lipschitz constraint on the Critic, yielding highly stable convergence and detailed galaxy structures.

## 📐 Mathematical Formulation

The optimization operates under solving the Minimax formulation of the Earth Mover's (Wasserstein) Distance.

**Critic Loss:**
$$ L_D = \mathbb{E}_{\tilde{x} \sim \mathbb{P}_g}[D(\tilde{x})] - \mathbb{E}_{x \sim \mathbb{P}_r}[D(x)] + \lambda \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2] $$

**Generator Loss:**
$$ L_G = -\mathbb{E}_{\tilde{x} \sim \mathbb{P}_g}[D(\tilde{x})] $$

Where:
- $\mathbb{P}_r$ is the real data distribution (Galaxy Zoo images).
- $\mathbb{P}_g$ is the model distribution implicitly defined by $\tilde{x} = G(z), z \sim p(z)$.
- $\lambda$ is the Gradient Penalty weight (default: 10).

## 🛠️ Project Structure

```text
├── src/
│   ├── config.py       # Hyperparameters and global settings
│   ├── data.py         # tf.data.Dataset pipeline & Kaggle downloading
│   ├── models.py       # Generator and Critic architectures
│   ├── train.py        # WGAN-GP custom training loop
│   └── utils.py        # GIF animations and plotting utilities
├── README.md           # This document
├── requirements.txt    # Python dependencies
└── GAN_images_of_galaxies.ipynb # Original legacy notebook
```

## ⚙️ Architecture

### Generator
- Maps a latent vector $z \in \mathbb{R}^{128}$ to a $64 \times 64 \times 3$ RGB image.
- Employs transposed convolutions, `BatchNormalization`, and `ReLU` activations.
- Final layer utilizes a `Tanh` activation, squashing pixels into $[-1, 1]$.

### Critic (Discriminator)
- Maps a $64 \times 64 \times 3$ image to a single scalar (Wasserstein distance estimator).
- Employs strided convolutions and `LeakyReLU(\alpha=0.2)` activations.
- **No BatchNormalization** is utilized to keep the gradient penalty valid.

## 🚀 Installation & Usage

### 1. Requirements

Ensure you have Python 3.9+ and TensorFlow 2.15+ installed.
```bash
pip install -r requirements.txt
```

### 2. Training the Model

Execute the training script. The script will automatically download the 3GB dataset efficiently using `kagglehub` and place it in the cache, mounting it seamlessly through `tf.data`.

```bash
python -m src.train
```

During training:
- **Model Checkpoints** will be saved inside the `checkpoints/` directory.
- **Visual Progress** will be dropped in the `output/` directory as `png` images.
- **TensorBoard Logging** will collect losses in `logs/` (Run `tensorboard --logdir logs/`).

Once the training concludes, `utils.py` will automatically stitch the epoch snapshots into an animated GIF showcasing the GAN learning over time.

## 📈 TensorBoard Metrics

Tracking the convergence can be done via TensorBoard:
```bash
tensorboard --logdir logs/
```

WGAN-GP metrics typically exhibit a noisy but strictly increasing (approaching 0 over long times) Discriminator Loss and a fluctuating Generator Loss mirroring the zero-sum nature.

## 📄 License
This project is an open source scientific endeavor. Datasets depend on the Galaxy Zoo 2 standard licensing.