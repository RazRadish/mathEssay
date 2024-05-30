import cv2
import numpy as np

def calculate_image_similarity(image1_path, image2_path):
    # 读取图像
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 调整图像大小以匹配
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # 将图像转换为灰度图像
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算图像相似度（利用像素级别的比较）
    pixel_similarity = np.sum(np.abs(gray_image1 - gray_image2)) / (gray_image1.shape[0] * gray_image1.shape[1])

    # 计算图像的余弦相似度
    dot_product = np.sum(gray_image1 * gray_image2)
    norm_image1 = np.sqrt(np.sum(gray_image1 ** 2))
    norm_image2 = np.sqrt(np.sum(gray_image2 ** 2))
    cosine_similarity = dot_product / (norm_image1 * norm_image2)

    return pixel_similarity, cosine_similarity

if __name__ == "__main__":
    image1_path = r"C:\Users\IBUKI\Desktop\Musume\Musume2.png"
    image2_path = r"C:\Users\IBUKI\Desktop\Musume\Musume2-better.png"

    pixel_similarity, cosine_similarity = calculate_image_similarity(image1_path, image2_path)

    print("Pixel Similarity:", pixel_similarity)
    print("Cosine Similarity:", cosine_similarity)
