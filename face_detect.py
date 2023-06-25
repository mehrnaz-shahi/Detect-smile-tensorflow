import os
import cv2

# Directory path for the images
images_dir = "dataset/smile"

# Directory path for saving the extracted faces
faces_dir = "faces/smile_faces"

# Create a CascadeClassifier object for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create the directory for saving faces if it doesn't exist
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)

# Get the list of image files in the directory
image_files = os.listdir(images_dir)

# For each image file
for image_file in image_files:
    # Path of the image file
    image_path = os.path.join(images_dir, image_file)

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    for i, (x, y, w, h) in enumerate(faces):
        # Extract the face image from the grayscale image
        face_image = gray_image[y:y + h, x:x + w]
        # Create a directory path to save the face with the same file name
        face_path = os.path.join(faces_dir, image_file)

        # If there are multiple faces, add the face number to the file name
        if len(faces) > 1:
            filename, ext = os.path.splitext(face_path)
            face_path = f"{filename}_face{i + 1}{ext}"

        # Save the face to a new image file
        cv2.imwrite(face_path, face_image)