## ---------- 1. Reduce the number of intensity levels ----------

import cv2
import numpy as np
import os

def reduce_intensity_levels(img, levels):
    factor = 256 // levels
    return (img // factor) * factor

def task1_reduce_levels(image_path, levels):
    # Validate if levels is a power of 2
    if levels < 2 or levels > 256 or (levels & (levels - 1)) != 0:
        raise ValueError("Levels must be a power of 2 between 2 and 256.")

    # Load grayscale and color images
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(image_path)

    if gray is None or color is None:
        print(f"Error: Image not found at {image_path}")
        return

    # Create output subfolder for this level
    level_folder = f"task1_output/level_{levels}"
    os.makedirs(level_folder, exist_ok=True)

    # Apply reduction
    reduced_gray = reduce_intensity_levels(gray, levels)
    reduced_color = reduce_intensity_levels(color, levels)

    # Save output images
    cv2.imwrite(f"{level_folder}/original_gray.jpg", gray)
    cv2.imwrite(f"{level_folder}/original_color.jpg", color)
    cv2.imwrite(f"{level_folder}/gray_reduced_{levels}.jpg", reduced_gray)
    cv2.imwrite(f"{level_folder}/color_reduced_{levels}.jpg", reduced_color)

    print(f"Saved intensity reduction for {levels} levels in '{level_folder}'.")

    # Display the images (optional: comment out if running many at once)
    cv2.imshow("Original Gray", gray)
    cv2.imshow(f"Gray {levels} Levels", reduced_gray)
    cv2.imshow("Original Color", color)
    cv2.imshow(f"Color {levels} Levels", reduced_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------- Main Loop for Multiple Levels ----------
image_path = 'lena.jpg'
levels_list = [2, 4, 8, 16, 32, 64, 128]

for lvl in levels_list:
    task1_reduce_levels(image_path, lvl)







## ---------- 2. Average Blur with 3x3, 10x10, 20x20 ----------

# Make output directory
os.makedirs("task2_output", exist_ok=True)

# Load grayscale image
image = cv2.imread('lena.jpg')

# Apply 3x3 average filter
blur_3 = cv2.blur(image, (3, 3))

# Apply 10x10 average filter
blur_10 = cv2.blur(image, (10, 10))

# Apply 20x20 average filter
blur_20 = cv2.blur(image, (20, 20))

# Save output images
cv2.imwrite(f"task2_output/original.jpg", image)
cv2.imwrite(f"task2_output/3x3_blur.jpg", blur_3)
cv2.imwrite(f"task2_output/10x10_blur.jpg", blur_10)
cv2.imwrite(f"task2_output/20x20_blur.jpg", blur_20)

# Show results
cv2.imshow("Original", image)
cv2.imshow("3x3 Blur", blur_3)
cv2.imshow("10x10 Blur", blur_10)
cv2.imshow("20x20 Blur", blur_20)
cv2.waitKey(0)
cv2.destroyAllWindows()





## ---------- 3. Rotate Image by 45 and 90 Degrees ----------

# Create output folder
os.makedirs("task3_output", exist_ok=True)

# Load image
image = cv2.imread('lena.jpg')

if image is None:
    print("Error: Image not found.")
    exit()

# Get image dimensions and center
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# ---------- Rotate 45 degrees CLOCKWISE ----------
# Negative angle = clockwise rotation
M_45 = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated_45 = cv2.warpAffine(image, M_45, (w, h))

# ---------- Rotate 90 degrees CLOCKWISE ----------
rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# ---------- Save rotated images ----------
cv2.imwrite("task3_output/original.jpg", image)
cv2.imwrite("task3_output/rotated_45_clockwise.jpg", rotated_45)
cv2.imwrite("task3_output/rotated_90_clockwise.jpg", rotated_90)

# ---------- Display for verification ----------
cv2.imshow("Original", image)
cv2.imshow("Rotated 45° Clockwise", rotated_45)
cv2.imshow("Rotated 90° Clockwise", rotated_90)
cv2.waitKey(0)
cv2.destroyAllWindows()




## ---------- 4. Block Averaging Function ----------


def block_average(image, block_size):
    (h, w) = image.shape
    output = np.zeros_like(image)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image[y:y+block_size, x:x+block_size]
            avg = np.mean(block, dtype=np.uint8)
            output[y:y+block_size, x:x+block_size] = avg

    return output

# Create output directory
os.makedirs("task4_output", exist_ok=True)

# Load image in grayscale
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found.")
    exit()

# Apply block averaging
avg_3 = block_average(image, 3)
avg_5 = block_average(image, 5)
avg_7 = block_average(image, 7)

# Save output images
cv2.imwrite("task4_output/original_gray.jpg", image)
cv2.imwrite("task4_output/block_avg_3x3.jpg", avg_3)
cv2.imwrite("task4_output/block_avg_5x5.jpg", avg_5)
cv2.imwrite("task4_output/block_avg_7x7.jpg", avg_7)

# Show results
cv2.imshow("Original", image)
cv2.imshow("3x3 Block Averaging", avg_3)
cv2.imshow("5x5 Block Averaging", avg_5)
cv2.imshow("7x7 Block Averaging", avg_7)
cv2.waitKey(0)
cv2.destroyAllWindows()