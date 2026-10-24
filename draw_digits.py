import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from pathlib import Path

# Define the same CNN architecture used during training
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# Load your trained model

MODEL_PATH = Path(__file__).resolve().parent / "mnist_cnn.pth"

model = CNN()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()


# Drawing canvas settings
CANVAS_SIZE = 280
BRUSH_SIZE = 18

root = tk.Tk()
root.title("MNIST Digit Recognizer")

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    cursor="cross"
)
canvas.pack(padx=10, pady=10)

# PIL image stores the drawing as white digits on black
drawing = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(drawing)

last_x = None
last_y = None


def start_draw(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y


def draw_digit(event):
    global last_x, last_y

    if last_x is not None and last_y is not None:
        canvas.create_line(
            last_x, last_y, event.x, event.y,
            fill="white",
            width=BRUSH_SIZE,
            capstyle=tk.ROUND,
            smooth=True
        )

        draw.line(
            (last_x, last_y, event.x, event.y),
            fill=255,
            width=BRUSH_SIZE
        )

        # Draw rounded ends
        r = BRUSH_SIZE // 2
        draw.ellipse(
            (event.x-r, event.y-r, event.x+r, event.y+r),
            fill=255
        )

    last_x = event.x
    last_y = event.y


def stop_draw(event):
    global last_x, last_y
    last_x = None
    last_y = None


def predict_digit():
    # Invert if necessary: MNIST uses white digits on black
    img = drawing.copy()

    # Crop the digit to its bounding box
    bbox = img.getbbox()

    if bbox is None:
        result_label.config(text="Draw a digit first!")
        return

    img = img.crop(bbox)

    # Preserve proportions and center the digit in a 28x28 image
    img.thumbnail((20, 20))

    digit_canvas = Image.new("L", (28, 28), 0)
    x = (28 - img.width) // 2
    y = (28 - img.height) // 2
    digit_canvas.paste(img, (x, y))

    # Convert pixels to a tensor in the same range as MNIST
    tensor = torch.tensor(
        list(digit_canvas.getdata()),
        dtype=torch.float32
    ).reshape(1, 1, 28, 28) / 255.0

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item() * 100

    result_label.config(
        text=f"Prediction: {prediction} ({confidence:.1f}% confidence)"
    )


def clear_canvas():
    global drawing, draw, last_x, last_y

    canvas.delete("all")
    drawing = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(drawing)
    last_x = None
    last_y = None
    result_label.config(text="Draw a digit and click Predict")


canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", stop_draw)

result_label = tk.Label(
    root,
    text="Draw a digit and click Predict",
    font=("Arial", 14)
)
result_label.pack(pady=5)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    width=12
).pack(side=tk.LEFT, padx=5)

tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    width=12
).pack(side=tk.LEFT, padx=5)

root.mainloop()
