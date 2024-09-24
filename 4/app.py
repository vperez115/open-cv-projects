import cv2
import numpy as np
import requests
from io import BytesIO
import random

def extract_patch(texture, patch_size):
    h, w, _ = texture.shape
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    return texture[y:y + patch_size, x:x + patch_size]

def create_texture_from_patches(sample_texture, patch_size, new_width, new_height):
    new_texture = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(0, new_height, patch_size):
        for j in range(0, new_width, patch_size):
            patch_height = min(patch_size, new_height - i)
            patch_width = min(patch_size, new_width - j)
            patch = extract_patch(sample_texture, patch_size)[:patch_height, :patch_width]
            new_texture[i:i + patch_height, j:j + patch_width] = patch

    return new_texture

def main():
    url = 'https://storage.googleapis.com/webdesignledger.pub.network/LaT/edd/2017/01/DSC00712-1560x1075.jpg'

    response = requests.get(url)
    image_bytes = BytesIO(response.content)

    image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    sample_texture = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    patch_size = 20  # Size of each patch
    new_width, new_height = 640, 480  # Size of the new texture
    synthesized_texture = create_texture_from_patches(sample_texture, patch_size, new_width, new_height)

    cv2.imshow('Synthesized Texture', synthesized_texture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
