# Detect-smile-tensorflow
Smile detection that is trained by TensorFlow and uses vgg 16 and resnet

by vgg16 in Colab: https://colab.research.google.com/drive/1qa-52v52EAW4Dvc6jKCRQqaygBIcfnRC

by resnet in Colab: https://colab.research.google.com/drive/1jLHrUWzHhLwqwYd0PNkqVepIxYph30Vt

You can see the image files at https://drive.google.com/drive/folders/1gMNWK7vKVklwk63MaGMZ0il23CszCyfB?usp=drive_link


Running:

-first, run face_detect.py to detect faces then adjust actions in argumentation.py and run it to achieve more images for the train.

-splitImages.py split the images based on labels to smile and not smile faces to separate folders.

-main.py is used to train.

-test.py is for testing the model. If you want to use the camera for testing, change cap = cv2.VideoCapture('test4.mp4') to cap = cv2.VideoCapture(0).
