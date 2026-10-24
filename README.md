# MNIST Classifier

A PyTorch convolutional neural network (CNN) that recognizes handwritten digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. The project includes a Jupyter notebook for training and evaluation, a saved model checkpoint, and a small Tkinter drawing application for interactive predictions.

## Features

- Downloads the MNIST training and test datasets automatically.
- Trains a lightweight CNN on 28×28 grayscale images.
- Evaluates the model with test accuracy, a classification report, and a confusion matrix.
- Saves the trained weights to `mnist_cnn.pth`.
- Provides an interactive digit-recognition GUI in `draw_digits.py`.

## Repository Contents

| File | Description |
| --- | --- |
| [`mnist.ipynb`](mnist.ipynb) | End-to-end notebook for data loading, training, evaluation, visualization, and model export. |
| [`draw_digits.py`](draw_digits.py) | Tkinter application that lets you draw a digit and view the model's prediction and confidence. |
| [`mnist_cnn.pth`](mnist_cnn.pth) | Saved PyTorch model weights used by the GUI. |
| `data/` | Directory where torchvision downloads the MNIST dataset. |

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- matplotlib
- seaborn
- scikit-learn
- Pillow
- Tkinter (usually included with desktop Python installations)
- Jupyter Notebook or JupyterLab

Install the Python dependencies with:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn pillow jupyter
```

> If Tkinter is not included with your Python installation, install the system package for your platform. For example, on Debian/Ubuntu:
>
> ```bash
> sudo apt-get install python3-tk
> ```

## Training and Evaluation

Launch the notebook from the repository root:

```bash
jupyter notebook mnist.ipynb
```

Run the notebook cells in order to:

1. Download and preprocess MNIST.
2. Create the training and test data loaders.
3. Train the CNN for five epochs using Adam and cross-entropy loss.
4. Evaluate the model on the 10,000-image test set.
5. Generate predictions, a classification report, and a confusion matrix.
6. Save the weights as `mnist_cnn.pth`.

The included notebook run achieved **98.67% test accuracy**. Results may vary slightly depending on the environment and training configuration.

The dataset is downloaded to `data/` automatically on the first run.

## Interactive Digit Recognizer

After `mnist_cnn.pth` is available, run:

```bash
python draw_digits.py
```

Draw a digit on the black canvas and click **Predict**. The application crops and centers the drawing, converts it to a 28×28 tensor, and displays the predicted digit with the model's confidence. Use **Clear** to try another digit.

## Model Architecture

The CNN defined in `mnist.ipynb` and `draw_digits.py` contains:

1. `Conv2d(1, 16, kernel_size=3)`
2. ReLU activation
3. 2×2 max pooling
4. `Conv2d(16, 32, kernel_size=3)`
5. ReLU activation
6. 2×2 max pooling
7. Flattening to 800 features
8. Fully connected layer with 128 units
9. Fully connected output layer with 10 units for digits 0–9

## Configuration

To change the training setup, edit the corresponding notebook cells:

- `batch_size`: number of images processed per training step; the notebook uses `64`.
- `num_epochs`: number of training epochs; the notebook uses `5`.
- Adam learning rate: the notebook uses `0.001`.

## License

No license has been specified for this repository yet.
