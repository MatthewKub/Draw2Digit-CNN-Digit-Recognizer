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

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert('L')

    # Normalizes to MNIST format (white on black)
    image = image.point(lambda x: 255 if x > 50 else 0, 'L')
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # Resizes proportionally
    w, h = image.size
    scale = 20.0 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Pastes onto 28×28 black canvas, centered
    new_img = Image.new('L', (28, 28), (0))
    new_img.paste(image, ((28 - new_w) // 2, (28 - new_h) // 2))

    # Applies standard transforms
    image = transform(new_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True).item()

    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True)
