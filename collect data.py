import math
import cv2
from datetime import datetime

color = (0, 0, 255)
count = 0
date = datetime.now()
date = date.strftime("%d.%m.%Y %H.%M.%S")


def get_face_image(image, x, y, side_length, scale):
    left = x
    top = y
    right = len(image[0]) - x - side_length
    bottom = len(image) - y - side_length

    scale_left = left / side_length
    scale_top = top / side_length
    scale_right = right / side_length
    scale_bottom = bottom / side_length

    scale_topleft = min(scale, scale_left, scale_top)
    scale_bottomright = min(scale, scale_right, scale_bottom)

    start_x = math.floor(x - scale_topleft * side_length + 0.5)
    end_x = math.floor(
        (x + side_length) + scale_bottomright * side_length + 0.5)
    start_y = math.floor((y - scale_topleft * side_length) + 0.5)
    end_y = math.floor(
        (y + side_length) + scale_bottomright * side_length + 0.5)

    image = image[start_y:end_y, start_x:end_x]
    return image


# model that will detect faces for us
model_detection = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cv2.namedWindow("Create Data")
camera = cv2.VideoCapture(0)
ret, frame = camera.read()

while ret:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = model_detection.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, length, _) in faces:
        face_image = get_face_image(frame, x, y, length, scale=0.5)
        cv2.imwrite(f"data/{date}_{count}.png", face_image)
        count += 1
        cv2.rectangle(frame, (x, y), (x + length, y + length), color, 2)

    cv2.imshow("Create Data", frame)

    key = cv2.waitKey(20)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
