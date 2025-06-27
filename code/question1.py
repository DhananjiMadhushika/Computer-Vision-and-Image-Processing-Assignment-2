import numpy as np
import cv2
import os

# ------------------- Step 1: Create image with two objects -------------------
def create_image(width, height):
    # Start with white background
    img = np.full((height, width), 255, dtype=np.uint8)

    center_y = height // 2

    # Object 1: Ellipse
    ellipse_center = (width // 4, center_y)
    axes_lengths = (60, 38)
    cv2.ellipse(img, ellipse_center, axes_lengths, 0, 0, 360, 128, -1)

    # Object 2: Triangle
    tri_center_x = 3 * width // 4
    tri_size = 90

    # Define triangle vertices
    pts = np.array([
        [tri_center_x, center_y - tri_size // 2],
        [tri_center_x - tri_size // 2, center_y + tri_size // 2],
        [tri_center_x + tri_size // 2, center_y + tri_size // 2]
    ], np.int32)

    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], 0)

    return img




# ------------------- Step 2: Add Gaussian noise -------------------
def apply_gaussian_noise(img, mean=0, stddev=50):
    noise = np.random.normal(mean, stddev, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# ------------------- Step 3: Apply Otsu's Thresholding -------------------
def apply_otsu_threshold(img):
    _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

# ------------------- Main Flow -------------------
if __name__ == "__main__":
    width, height = 300, 300

    # Save path
    output_dir = r"../result"
    os.makedirs(output_dir, exist_ok=True)

    # Create and process images
    base_img = create_image(width, height)
    noisy_img = apply_gaussian_noise(base_img)
    otsu_img = apply_otsu_threshold(noisy_img)

    # Save results
    cv2.imwrite(os.path.join(output_dir, "originalImage.png"), base_img)
    cv2.imwrite(os.path.join(output_dir, "noisyImage.png"), noisy_img)
    cv2.imwrite(os.path.join(output_dir, "otsuImage.png"), otsu_img)

    # Optional: Display results
    cv2.imshow("Original Image", base_img)
    cv2.imshow("Noisy Image", noisy_img)
    cv2.imshow("Otsu Threshold", otsu_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
