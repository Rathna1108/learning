import cv2
import matplotlib.pyplot as plt

# Load and convert to grayscale
image = cv2.imread('images/Dhingra ENT 8th Edition_Split_page-0006.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply denoising BEFORE thresholding
# denoised = cv2.fastNlMeansDenoising(
#     gray,
#     None,           # Output image (None = create new)
#     h=10,           # Filter strength (higher = more smoothing)
#     templateWindowSize=7,   # Size of template patch
#     searchWindowSize=21     # Size of search area
# )

denoised_aggressive = cv2.fastNlMeansDenoising(gray, None, h=20, templateWindowSize=7, searchWindowSize=21)


# Now apply adaptive threshold to denoised image
binary_clean = cv2.adaptiveThreshold(
    denoised_aggressive,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# Compare: without denoising vs with denoising
binary_noisy = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

binary_inv = cv2.bitwise_not(binary_clean)

contours, hierarchy = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f'total contours found: {len(contours)}')


image_with_contours = image.copy()

cv2.drawContours(image_with_contours, contours, -1, (0,255,0), 3)

# Display comparison
# fig, axes = plt.subplots(1, 1, figsize=(15, 15))

# axes[0, 0].imshow(gray, cmap='gray')
# axes[0, 0].set_title('1. Original Grayscale')
# axes[0, 0].axis('off')

# axes[0, 1].imshow(denoised, cmap='gray')
# axes[0, 1].set_title('2. After Denoising')
# axes[0, 1].axis('off')

# axes[1, 0].imshow(binary_noisy, cmap='gray')
# axes[1, 0].set_title('3. Threshold WITHOUT Denoising')
# axes[1, 0].axis('off')

# axes.imshow(denoised_aggressive, cmap='gray')
# axes.set_title('4. Threshold WITH Denoising (Cleanest!)')
# axes.axis('off')


fig, axes = plt.subplots(1, 3, figsize=(18, 8))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(binary_inv, cmap='gray')
axes[1].set_title('Binary Inverted (for contour detection)')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'Contours Found: {len(contours)}')
axes[2].axis('off')

plt.tight_layout()
plt.show()