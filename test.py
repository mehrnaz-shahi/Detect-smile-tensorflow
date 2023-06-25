import cv2
import tensorflow as tf
import numpy as np

cap = cv2.VideoCapture('test4.mp4')
facedetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vwriter = cv2.VideoWriter('out.wmv', cv2.VideoWriter_fourcc(*'WMV1'), 20, (640, 480))
model = tf.keras.models.load_model("detect_99_model2.h5")


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces


while (True):
    ret, frame = cap.read()
    if ret:

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 10, 10), 2)

            # Resize image to 224x224
            resized_image = cv2.resize(face_img, (100, 100))

            # Preprocess the image
            preprocessed_image = tf.keras.applications.vgg16.preprocess_input(resized_image)

            # Add batch dimension
            input_image = np.expand_dims(preprocessed_image, axis=0)

            # Make predictions
            predictions = model.predict(input_image)

            # Convert predictions to labels
            label = "smile" if predictions[0] > 0.5 else "not smile"

            # Print the label
            print("Prediction: ", label)
            if predictions[0] > 0.5:
                image = cv2.putText(frame, 'Smiling', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                image = cv2.putText(frame, "Not Smiling", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)

        vwriter.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    else:
        vwriter.release()

cap.release()