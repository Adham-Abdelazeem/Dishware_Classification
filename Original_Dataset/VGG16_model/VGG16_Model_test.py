import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

model = tf.keras.models.load_model(r"D:\Master\M.L & A.I\Tasks\Task 8\models\Original_Dataset\VGG16_model\vgg16.hdf5")

data_path_test = r"D:\Master\M.L & A.I\Tasks\Task 8\DataSet\test"
img_height = 255
img_width = 255
test_ds = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    data_path_test,
    target_size=(img_height, img_width),
    classes=['bowls', 'cups', 'plates'],
    batch_size=10,
    shuffle=False
)


def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(test_ds.classes, y_pred, target_names=['bowls', 'cups', 'plates'], ))


predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)
print(test_ds.classes)
print(y_pred)
print_confusion_matrix(test_ds.classes, y_pred)
clas_report = classification_report(test_ds.classes, y_pred, target_names=['bowls', 'cups', 'plates'], output_dict=True)
print("Accuracy =  ", 100 * round(clas_report.get('accuracy'), 2), '%')
