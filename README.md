# MNIST Classifier

A simple convolutional neural network (CNN) for handwritten digit classification using PyTorch and the MNIST dataset.

## Features

- Downloads and preprocesses the MNIST dataset automatically.
- Trains a lightweight CNN with two convolutional layers.
- Uses CUDA automatically when available.
- Evaluates the model on the MNIST test set after each epoch.
- Saves the trained weights to `mnist_cnn.pth`.

## Requirements

- Python 3.9+
- PyTorch
- torchvision

Install the dependencies with:

```bash
pip install torch torchvision
```

## Usage

Run the training script:

```bash
python mnist_pytorch.py
```

The script trains for 5 epochs by default. Training and test accuracy are printed to the console, and the trained model is saved as:

```text
mnist_cnn.pth
```

The MNIST data is downloaded into the `data/` directory on the first run.

## Model Architecture

The classifier consists of:

1. A convolutional layer with 32 filters.
2. A max-pooling layer.
3. A convolutional layer with 64 filters.
4. A second max-pooling layer.
5. A fully connected layer with 128 units and dropout.
6. A 10-class output layer for digits 0–9.

## Configuration

Training settings can be adjusted at the top of `mnist_pytorch.py`:

- `BATCH_SIZE`: Batch size used during training.
- `EPOCHS`: Number of training epochs.
- `LR`: Learning rate for the Adam optimizer.

## License

No license has been specified for this repository yet.
