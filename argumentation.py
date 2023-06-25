import cv2
import os


def augment_data(image_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)

    # Create a new output file name based on the original file name
    file_name = os.path.basename(image_path)
    file_name = os.path.splitext(file_name)[0]

    # Rotation at different angles
    # angle = 60
    # rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    # rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    # cv2.imwrite(os.path.join(output_dir, file_name + '_rotated_' + str(angle) + '.jpg'), rotated_image)

    # Brightness and contrast adjustment
    alpha = 0.8
    beta = -30
    brightness_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    cv2.imwrite(os.path.join(output_dir, file_name + '_brightness_' + str(alpha) + '_' + str(beta) + '.jpg'),
                brightness_image)


# Path to the input images directory
input_dir = 'faces/not_smile_faces'

# Path to the output images directory
output_dir = 'argu/not_smile'

# Check if the output directory exists and create it if necessary
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Traverse the input images
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)
        augment_data(image_path, output_dir)
