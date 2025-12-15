import cv2 


image = cv2.imread('textbook_images/page_001.jpg')

#rescale the image size
def rescaleFrame(frame, scale = 0.4):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimension = (width, height)

    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

rescaled_image = rescaleFrame(image)

cv2.imshow('rescaled_image', rescaled_image)

cv2.waitKey(0)

capture = cv2.VideoCapture('videos/test.mp4')

while True:
    isTrue, frame = capture.read()
    if isTrue == False:
        break

    resized_frame = rescaleFrame(frame)
    cv2.imshow('video', resized_frame)

    if cv2.waitKey(20) & 0xFF == ord('v'):
        break

capture.release()
cv2.destroyAllWindows()


