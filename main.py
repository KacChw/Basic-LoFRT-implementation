///
import cv2
import torch
from kornia.feature import LoFTR
import matplotlib.pyplot as plt

# sprawdzic czy CUDA jest dostępne
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# inicjalizacja zalecanego LoFTR
matcher = LoFTR(pretrained="outdoor").to(device)

# funkcja do wczytania obrazu i przygotowania go do LoFTR
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Wczytanie obrazu w skali szarości
    img_tensor = torch.from_numpy(img / 255.0).float()[None, None].to(device)  # Normalizacja i konwersja do tensora
    return img, img_tensor

# przyklad
image_path_1 = "image1.jpg"  #  ścieżka do pierwszego obrazu
image_path_2 = "image2.jpg"  #  ścieżka do drugiego obrazu

img1, img1_tensor = preprocess_image(image_path_1)
img2, img2_tensor = preprocess_image(image_path_2)

# Dopasowanie kluczowych punktów za pomocą LoFTR
with torch.no_grad():
    input_dict = {"image0": img1_tensor, "image1": img2_tensor}
    correspondences = matcher(input_dict)

# Pobranie dopasowań i konwersja do np
mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()

# Wizualizacja dopasowań
def visualize_matches(img1, img2, mkpts0, mkpts1):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Łączenie obrazów w jeden
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    combined_img = cv2.hconcat([img1, img2])

    # Przesunięcie punktów z drugiego obrazu
    mkpts1_shifted = mkpts1.copy()
    mkpts1_shifted[:, 0] += w1

    # Rysowanie dopasowań
    ax.imshow(combined_img, cmap="gray")
    for pt1, pt2 in zip(mkpts0, mkpts1_shifted):
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "r-", alpha=0.5)
    ax.scatter(mkpts0[:, 0], mkpts0[:, 1], c="blue", s=5, label="Image 1")
    ax.scatter(mkpts1_shifted[:, 0], mkpts1_shifted[:, 1], c="green", s=5, label="Image 2")

    ax.axis("off")
    ax.legend()
    plt.show()

visualize_matches(img1, img2, mkpts0, mkpts1)
