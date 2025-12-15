import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('images/Dhingra ENT 8th Edition_Split_page-0003.jpg')
image2 = cv2.imread('images/Dhingra ENT 8th Edition_Split_page-0006.jpg')

if image is None:
    print("Error: Could not load image!")
else:
    print("Image loaded successfully!")
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height} pixels")
    
    # Convert to grayscale
    gray_img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(f"Grayscale shape: {gray.shape}")  # Notice: only 2 dimensions now!
    

    gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


    # Display original and grayscale side by side
    fig, axes = plt.subplots(1,1,figsize=(15, 8))
    
    # Original image
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # axes[0].imshow(image_rgb)
    # axes[0].set_title('Original Image (Color)')
    # axes[0].axis('on')
    
    # Grayscale image
    # axes[1].imshow(th, cmap='gray')
    # axes[1].set_title('Grayscale Image')
    # axes[1].axis('on')

    _, th = cv2.threshold(gray_img1, 177, 255, cv2.THRESH_BINARY)
    _, th2 = cv2.threshold(gray_img2, 97, 255, cv2.THRESH_BINARY)

    # axes[0].axis('on')
    binary_adaptive = cv2.adaptiveThreshold(
    gray_img1, 
    255,                              # Max value (white)
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # Method
    cv2.THRESH_BINARY,                # Type
    11,                               # Block size (neighborhood area)
    2                                 # Constant subtracted from mean
)


    def rescaleFrame(frame, scale = 0.7):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)

        dimension = (width, height)

        return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    rescaled_image = rescaleFrame(th)

    # cv2.imshow('Thresholded Image',rescaled_image)
   
  

    axes.imshow(binary_adaptive,cmap = 'gray')
    axes.set_title('adaptive Image')
    axes.axis('off')

    
    plt.tight_layout()
    plt.show()

 