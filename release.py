import numpy as np
import cv2
from tensorflow import keras

RED = (0, 0, 255)
GREEN = (0, 255, 0)
model_name = "face_classifier 2.1.2.h5"


def get_face_image(image, x, y, side_length, scale):
    scale_top = [scale] * 2
    scale_bottom = [scale] * 2

    if x - scale * side_length < 0:
        scale_top[0] = x / side_length
    if y - scale * side_length < 0:
        scale_top[1] = y / side_length
    if (x + side_length) + scale * side_length > len(image[0]):
        scale_bottom[0] = (len(image[0]) - (x + side_length)) / side_length
    if (y + side_length) + scale * side_length < len(image):
        scale_bottom[1] = (len(image) - (y + side_length)) / side_length

    scale_top = min(scale_top)
    scale_bottom = min(scale_bottom)

    start_x = int(x - scale_top * side_length)
    end_x = int((x + side_length) + scale_bottom * side_length)
    start_y = int(y - scale_top * side_length)
    end_y = int((y + side_length) + scale_bottom * side_length)

    image = image[start_y:end_y, start_x:end_x]
    image = cv2.resize(image, IMAGE_SIZE)
    '''cv2.imshow("MeNotMe", face_image)'''
    image = np.expand_dims(image, axis=0)
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
