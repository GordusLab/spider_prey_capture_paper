import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, array_to_img
import os
from PIL import Image

# Function to load images
def load_images(image_paths):
    images = np.load(image_paths)
    images = images['arr_0']
    return np.array(images)

# Directory containing your spider web images

# Load images into a NumPy array
images = load_images('X.npz')
Y_truth = load_images('Y.npz')
images  = images[:, int(1280/2)-425: int(1280/2)+425 , int(1024/2)-425: int(1024/2)+425,:]
Y_truth  = Y_truth[:, int(1280/2)-425: int(1280/2)+425 , int(1024/2)-425: int(1024/2)+425,:]
Y_truth = Y_truth*255
Y_truth = Y_truth.astype(np.uint8)


images = images[86:91]
Y_truth = Y_truth[86:91]


# Custom function to apply the same transformation to both image and label
import random
def augment_image_and_label(image, label, shear_range=0.2):
    # Random rotation
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        image = image.rotate(angle)
        label = label.rotate(angle)

    # Random horizontal shift
    if random.random() > 0.5:
        shift_x = random.uniform(-0.1, 0.1) * image.width
        image = image.transform(image.size, Image.AFFINE, (1, 0, shift_x, 0, 1, 0))
        label = label.transform(label.size, Image.AFFINE, (1, 0, shift_x, 0, 1, 0))

    # Random vertical shift
    if random.random() > 0.5:
        shift_y = random.uniform(-0.1, 0.1) * image.height
        image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, shift_y))
        label = label.transform(label.size, Image.AFFINE, (1, 0, 0, 0, 1, shift_y))

    # # Random shear
    # if random.random() > 0.5:
    #     shear_factor = random.uniform(-shear_range, shear_range)
    #     shear_matrix = (1, shear_factor, 0, shear_factor, 1, 0)
    #     image = image.transform(image.size, Image.AFFINE, shear_matrix)
    #     label = label.transform(label.size, Image.AFFINE, shear_matrix)

    # Random flip horizontally
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    # Random zoom
    if random.random() > 0.5:
        zoom_factor = random.uniform(0.8, 1.2)
        new_size = (int(image.width * zoom_factor), int(image.height * zoom_factor))
        image = image.resize(new_size)
        label = label.resize(new_size)
    image = np.array(image)
    label = np.array(label)



    ### Randomly adjust "light" on web
    disappear_size = 30
    N = 10
    rand_vals = np.random.randint(1, len(np.where(label == 255)[0]) - N - 1, size=N)

    for k in range(N):
        idx_x = np.where(label == 255)[0][k]
        idx_y = np.where(label == 255)[1][k]

        image[idx_x:idx_x + disappear_size, idx_y:idx_y + disappear_size] = 0



    return np.array(image), np.array(label)


# Generate augmented images and labels and save them
from PIL import Image


output_image_dir = "output/output_images"
output_label_dir = "output/output_labels"

if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)
if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)

# Number of augmented images to generate
num_generated = 2500
X_syn =[]
Y_syn = []
for i in range(num_generated):
    idx = i % len(images)  # Cycle through the original images
    img = Image.fromarray(images[idx,:,:,0].astype(np.uint8))
    lbl = Image.fromarray(Y_truth[idx,:,:,0].astype(np.uint8))

    # Apply the same augmentation to both image and label
    augmented_img, augmented_lbl = augment_image_and_label(img, lbl)
    augmented_lbl = (augmented_lbl/255).astype(int)
    if augmented_img.shape[0]>=850:
        resize_img = np.zeros((1280, 1024))
        resize_img[int(1280 / 2) - 425: int(1280 / 2) + 425, int(1024 / 2) - 425: int(1024 / 2) + 425] = augmented_img[0:850, 0:850]
        resize_lbl = np.zeros((1280, 1024))
        resize_lbl[int(1280 / 2) - 425: int(1280 / 2) + 425, int(1024 / 2) - 425: int(1024 / 2) + 425] = augmented_lbl[0:850, 0:850]

    elif augmented_img.shape[0]%2==1:
        continue
    else:
        resize_img = np.zeros((1280, 1024))
        resize_img[int(1280 / 2) - int(augmented_img.shape[0]/2): int(1280 / 2) + int(augmented_img.shape[0]/2), int(1024 / 2) - int(augmented_img.shape[1]/2): int(1024 / 2) + int(augmented_img.shape[1]/2)] = augmented_img
        resize_lbl = np.zeros((1280, 1024))
        resize_lbl[int(1280 / 2) - int(augmented_lbl.shape[0]/2): int(1280 / 2) + int(augmented_lbl.shape[0]/2), int(1024 / 2) - int(augmented_lbl.shape[1]/2): int(1024 / 2) + int(augmented_lbl.shape[1]/2)] = augmented_lbl



    resize_img = resize_img.astype(np.uint8)
    resize_lbl = resize_lbl.astype(int)
    resize_img = resize_img.reshape(1280, 1024, 1)
    resize_lbl = resize_lbl.reshape( 1280, 1024, 1)
    X_syn.append(resize_img)
    Y_syn.append(resize_lbl)

    # Save the augmented images and labels
X_syn = np.array(X_syn)
Y_syn = np.array(Y_syn)
np.savez('X_test_syn.npz', X_syn)
np.savez('Y_test_syn.npz', Y_syn)

print(f"Generated {num_generated} augmented images and labels.")

