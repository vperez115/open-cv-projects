import cv2
import numpy as np
import random
from tkinter import Tk, filedialog

def upload_image_file():
    """Open a file dialog to upload an image."""
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

def extract_patch(texture, patch_size):
    """Extract a random patch from the given texture."""
    h, w, _ = texture.shape
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    return texture[y:y + patch_size, x:x + patch_size]

def create_texture_from_patches(sample_texture, patch_size, new_width, new_height):
    """Create a new texture by stitching patches from the sample texture."""
    new_texture = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(0, new_height, patch_size):
        for j in range(0, new_width, patch_size):
            patch_height = min(patch_size, new_height - i)
            patch_width = min(patch_size, new_width - j)
            patch = extract_patch(sample_texture, patch_size)[:patch_height, :patch_width]
            new_texture[i:i + patch_height, j:j + patch_width] = patch

    return new_texture

def main():
    # Upload the image file
    image_path = upload_image_file()

    if image_path:
        # Read the uploaded image using OpenCV
        sample_texture = cv2.imread(image_path)

        if sample_texture is None:
            print("Failed to load the image. Please try again.")
            return

        patch_size = 20  # Size of each patch
        new_width, new_height = 640, 480  # Size of the new texture

        # Create a synthesized texture from patches of the uploaded image
        synthesized_texture = create_texture_from_patches(sample_texture, patch_size, new_width, new_height)

        # Display the synthesized texture
        cv2.imshow('Synthesized Texture', synthesized_texture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
