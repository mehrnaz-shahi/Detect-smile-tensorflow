import os
import shutil
import numpy as np

# Specify the path  files directory
files_directory = 'genki/files'

# Read the labels.txt file
lables = np.loadtxt("genki/labels.txt", usecols=(0,))
lables = lables.astype(int)

# Create new directories
smile_directory = "dataset/smile"
not_smile_directory = "dataset/not_smile"
os.makedirs(smile_directory, exist_ok=True)
os.makedirs(not_smile_directory, exist_ok=True)

# Read the images and move them to the appropriate directories
image_files = os.listdir(files_directory)
for image_file in image_files:
    image_path = os.path.join(files_directory, image_file)
    print(f'image_file.split(".")[0][4: 7] : {image_file.split(".")[0][4: 8]}')
    index = int(image_file.split(".")[0][4: 8])
    print(f'index : {index}')
    label = lables[int(image_file.split(".")[0][4: 8]) - 1]
    print(label)
    if label == 1:
        destination_directory = smile_directory
    else:
        destination_directory = not_smile_directory
    print(destination_directory)
    print('--------------------------')
    shutil.copy(image_path, os.path.join(destination_directory, image_file))
