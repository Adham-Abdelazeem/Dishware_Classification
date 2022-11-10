# Auther: Group 14
# Data: Septemper/2022
# Task_name: Improve the accuracy of our supervised learning project based on our kitchenware dataset and classification algorithm.

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

data_path_train = r"D:\Master\M.L & A.I\Tasks\Task 8\Filter_Yes_Aug_Yes\train"
data_path_valid = r"D:\Master\M.L & A.I\Tasks\Task 8\Filter_Yes_Aug_Yes\valid"
data_path_test = r"D:\Master\M.L & A.I\Tasks\Task 8\Filter_Yes_Aug_Yes\test"

img_height = 255
img_width = 255

train_ds = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    data_path_train,
    target_size=(img_height, img_width),
)

# For seed,guarantee the same set of randomness [e.g. initializing weights of ANN, if not set, very different results can arrise]
val_ds = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    data_path_valid,
    target_size=(img_height, img_width),
)


# Building NN model
# __NOTE__ : we can use another type of layers specially for images from here ( https://keras.io/api/layers/ )

model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 3)),
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3)
])


model.summary()  # summary of the model (Type, shape, parameters number for each layer)

"""
input shape pic 28 pixels * 28 pixels 
Dense (neuron numbers )
"""
# Compile the built model with imported Fasion MNIST dataset
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
"""
loss = the way how good or bad the guess  of the random values
optimizer = to geuss another related values to see if it would be better
"""
epochs_num = 10
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs_num)  #
# model.fit(data to learn from , labels to learn from)
# Trains the model for a fixed number of epochs

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs_num)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save(r"D:\Master\M.L & A.I\Tasks\Task 8\models\Filter_Yes_Aug_Yes\CNN_model\CNN_model.hdf5", include_optimizer=True)