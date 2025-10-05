import os
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps, ImageFilter
import torch
import torchvision.transforms as transforms
from CNNmodel import CNNModel

# Explicit template path
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))


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
    image = Image.open(file).convert('L')  # Already white-on-black

    # Resize and normalize
    image = image.resize((28, 28))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True).item()

    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True)
