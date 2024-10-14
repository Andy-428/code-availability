import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import numpy as np
from oiodnet import OIODNet

# Define the path of the pre-trained weights
alexnet_pretrained_path = 'oiodnet/alexnet-owt-7be5be79.pth'
shufflenet_pretrained_path = 'oiodnet/shufflenetv2_x2_0-8be3c8ee.pth'

# Image enhancement function
def enhance_image(image):
    # The image is converted to grayscale and enhanced by CLAHE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

# Load and preprocess images
def load_and_preprocess_image(image_path, transform, enhance=False):
    image = cv2.imread(image_path)
    if enhance:
        image = enhance_image(image)
    
    # Convert an OpenCV image to a PIL image and apply the conversion
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Prediction function
def predict_image(image_tensor, model, device, class_names):
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Forward propagation results in model output
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)  # Softmax normalization
        confidence, predicted = torch.max(probabilities, 1)  # The class of maximum confidence
    
    predicted_class = class_names[predicted.item()]
    predicted_confidence = confidence.item() * 100
    return predicted_class, predicted_confidence, probabilities

if __name__ == "__main__":
    # Check CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define a class name
    class_names = ['class1', 'class2']  # Change the value based on the actual category
    
    # Define a preprocessing transformation of an image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalization
    ])
    
    # Loading model
    model = OIODNet(alexnet_pretrained_path=alexnet_pretrained_path, 
                        shufflenet_pretrained_path=shufflenet_pretrained_path).to(device)
    
    # Load model weight
    model.load_state_dict(torch.load('oiodnet/oiodnet_average_weights.pth', map_location=device))

    # Predicted picture path
    image_path = ''
    
    # Prediction of raw and enhanced images
    original_image_tensor = load_and_preprocess_image(image_path, transform, enhance=False)
    enhanced_image_tensor = load_and_preprocess_image(image_path, transform, enhance=True)

    # The original image is predicted
    original_class, original_confidence, _ = predict_image(original_image_tensor, model, device, class_names)
    
    # The enhanced image is predicted
    enhanced_class, enhanced_confidence, _ = predict_image(enhanced_image_tensor, model, device, class_names)

    # Output result
    print(f"Raw image classification: {original_class}, Confidence degree: {original_confidence:.2f}%")
    print(f"Enhanced image classification: {enhanced_class}, Confidence degree: {enhanced_confidence:.2f}%")