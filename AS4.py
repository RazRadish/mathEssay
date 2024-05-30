import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

def image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(image)
    tensor = tensor.unsqueeze(0)  # 添加批次维度
    return tensor

image_path1 = r'C:\Users\IBUKI\Desktop\Musume\Musume2.png'
image_path2 = r'C:\Users\IBUKI\Desktop\Musume\Musume2-better.png'

tensor1 = image_to_tensor(image_path1)
tensor2 = image_to_tensor(image_path2)


tensor1_flat = tensor1.view(1, -1)
tensor2_flat = tensor2.view(1, -1)

cosine_similarity = F.cosine_similarity(tensor1_flat, tensor2_flat).item()

cosine_similarity = max(min(cosine_similarity, 1.0), -1.0)

angle = torch.acos(torch.tensor(cosine_similarity)) * (180.0 / 3.141592653589793)

print("Cosine Similarity:", cosine_similarity)
print("Angle (degrees):", angle.item())
