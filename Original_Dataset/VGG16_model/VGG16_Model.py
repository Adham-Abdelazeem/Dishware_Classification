# Auther: Group 14
# Data: Septemper/2022
# Task_name: Improve the accuracy of our supervised learning project based on our kitchenware dataset and classification algorithm.

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

data_path_train = r"D:\Master\M.L & A.I\Tasks\Task 8\DataSet\train"
data_path_valid = r"D:\Master\M.L & A.I\Tasks\Task 8\DataSet\valid"
data_path_test = r"D:\Master\M.L & A.I\Tasks\Task 8\DataSet\test"

img_height = 255
img_width = 255

train_ds = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    data_path_train,
    target_size=(img_height, img_width),
    batch_size=10,
    classes=['bowls', 'cups', 'plates']
)

# For seed,guarantee the same set of randomness [e.g. initializing weights of ANN, if not set, very different results can arrise]
val_ds = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    data_path_valid,
    target_size=(img_height, img_width),
    classes=['bowls', 'cups', 'plates'],
    batch_size=10
)
test_ds = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    data_path_test,
    target_size=(img_height, img_width),
    classes=['bowls', 'cups', 'plates'],
    batch_size=10
)

# Building NN model


vgg16_model = tf.keras.applications.vgg16.VGG16()

vgg16_model.summary()  # summary of the model (Type, shape, parameters number for each layer)

# Model Fine-Tunning
model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 3))
])
for layer in vgg16_model.layers[1:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False
model.add(layers.Dense(3, activation='softmax'))

"""
input shape pic 28 pixels * 28 pixels 
Dense (neuron numbers )
"""
# Compile the built model with imported Fasion MNIST dataset
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
"""
loss = the way how good or bad the guess  of the random values
optimizer = to geuss another related values to see if it would be better
"""
epochs_num = 10
history = model.fit(x=train_ds,
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
model.save(r"D:\Master\M.L & A.I\Tasks\Task 8\models\Original_Dataset\VGG16_model\vgg16.hdf5", include_optimizer=True)
