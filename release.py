import numpy as np
import cv2
from tensorflow import keras

RED = (0, 0, 255)
GREEN = (0, 255, 0)
model_name = "face_classifier 2.1.2.h5"


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
    end_x = math.floor((x + side_length) + scale_bottomright * side_length + 0.5)
    start_y = math.floor((y - scale_topleft * side_length) + 0.5)
    end_y = math.floor((y + side_length) + scale_bottomright * side_length + 0.5)

    image = image[start_y:end_y, start_x:end_x]
    return image


# model that will detect faces for us
model_detection = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model_classifier = keras.models.load_model(f"models/{model_name}")
IMAGE_SIZE = model_classifier.layers[0].input_shape[0][1:3]
class_names = ["me", "not me"]

cv2.namedWindow("MeNotMe")
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
        # for each detected face on the image
        face_image = get_face_image(frame, x, y, length, scale=0.5)

        # classify face
        result = model_classifier.predict(face_image)
        prediction = class_names[round(result[0][0])]

        if prediction == "me":
            color = GREEN
            confidence = 1 - result[0][0]
        else:
            color = RED
            confidence = result[0][0]
        # draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + length, y + length), color, 2)
        cv2.putText(frame,
                    "{:6} - {:.2f}%".format(prediction, confidence * 100),
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    cv2.imshow("MeNotMe", frame)

    key = cv2.waitKey(20)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
