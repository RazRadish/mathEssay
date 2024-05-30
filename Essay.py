import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import spacy


# 加载自然语言处理模型
nlp = spacy.load("en_core_web_sm")

# 加载和预处理图像
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

image1 = load_and_preprocess_image('image1.jpg')
image2 = load_and_preprocess_image('image2.jpg')

# 像素相似性计算
def pixel_similarity(image1, image2):
    return np.linalg.norm(image1 - image2)

pixel_sim = pixel_similarity(image1, image2)
print(f"Pixel Similarity: {pixel_sim}")

# 余弦相似性计算
def extract_features(image):
    model = resnet50(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features

def cosine_similarity(feature1, feature2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(feature1, feature2).item()

features1 = extract_features(image1)
features2 = extract_features(image2)
cosine_sim = cosine_similarity(features1, features2)
print(f"Cosine Similarity: {cosine_sim}")

def rgb_vector_analysis(image):
    r_mean = np.mean(image[:,:,0])
    g_mean = np.mean(image[:,:,1])
    b_mean = np.mean(image[:,:,2])
    return np.array([r_mean, g_mean, b_mean])

def rgb_similarity(image1, image2):
    mean_rgb1 = np.mean(image1, axis=(0, 1))
    mean_rgb2 = np.mean(image2, axis=(0, 1))
    return np.linalg.norm(mean_rgb1 - mean_rgb2)

rgb_sim = rgb_similarity(image1, image2)
print(f"RGB Similarity: {rgb_sim}")

# 自然语言生成
def generate_description(image, pixel_sim, cosine_sim, rgb_sim):
    doc = nlp("This image has been analyzed for object recognition.")
    description = f"Image Analysis Report:\nPixel Similarity: {pixel_sim}\nCosine Similarity: {cosine_sim}\nRGB Similarity: {rgb_sim}\n"
    description += "Detected objects and their attributes will be described here."
    return description

description = generate_description(image1, pixel_sim, cosine_sim, rgb_sim)
print(description)

def extract_features(image):
    model = resnet50(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features

# Test the module with two images
def test_module(image_path1, image_path2):
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')

    # Compute pixel similarity
    pixel_sim = pixel_similarity(img1, img2)

    # Extract features and compute cosine similarity
    features1 = extract_features(img1)
    features2 = extract_features(img2)
    cosine_sim = cosine_similarity(features1, features2).item()

    # Compute RGB vector similarity
    rgb_vector1 = rgb_vector_analysis(img1)
    rgb_vector2 = rgb_vector_analysis(img2)
    rgb_sim = rgb_similarity(rgb_vector1, rgb_vector2)

    # Compile similarities into a dictionary
    similarities = {
        'pixel': pixel_sim,
        'cosine': cosine_sim,
        'rgb': rgb_sim
    }

    # Generate description
    description = generate_description(similarities)
    return description

# Example usage
image_path1 = 'path/to/first/image.jpg'
image_path2 = 'path/to/second/image.jpg'
result = test_module(image_path1, image_path2)
print(result)