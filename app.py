from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps, ImageFilter
import torch
import torchvision.transforms as transforms
from CNNmodel import CNNModel

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Keep normalization, but no grayscale here (we’ll handle manually)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert('L')

    # Invert to match MNIST
    image = ImageOps.invert(image)

    # Enhance contrast slightly
    image = image.filter(ImageFilter.SHARPEN)

    # Resize and center
    image = image.resize((28, 28))

    # Apply transforms
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True).item()

    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True)
