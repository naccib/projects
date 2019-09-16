import numpy as np

import matplotlib.pyplot as plt
from skimage.io import imshow, imshow_collection

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_SIZE = 23262
BATCH_SIZE = 32

#subjs = load_subjects(group_filter=['CONTROL', 'SCHZ'])
#images, labels = preprocess(subjs)
# X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, shuffle=True)

def number_to_label(number: int) -> str:
    return ['Cat', 'Dog'][number]


def preview_images(dataset: tf.data.Dataset):
    for n, tensor in enumerate(dataset.take(8)):
        image = tensor[0]

        plt.subplot(4, 2, n + 1)
        plt.imshow(image)
        plt.xlabel(number_to_label(tensor[1]))

        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
    plt.show()


def preprocess_image_label_pair(image, label):
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image, label


data_dict, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

dataset = data_dict['train']
dataset = dataset.map(preprocess_image_label_pair)

dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

# split the dataset

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

# defining the model

model = tf.keras.models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(192, 192, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))