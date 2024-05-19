from helper_function import load_and_prep_image
# import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import csv
import tensorflow as tf
import os

# import keras

# print(tf.__version__)
# print("Keras version:", keras.__version__)

# csv_file_path = r"C:\class_names.csv"
# class_names = []
# with open(csv_file_path, mode='r', newline='') as file:
#     reader = csv.reader(file)
#     # Skip the header row
#     next(reader)
#     # Iterate over rows and append class names to the list
#     for row in reader:
#         class_names.append(row[1])
#
# print("Loaded classes: ", class_names)

# model = load_model("C:\EfficientNetB1 (1).hdf5")


def pred_plot_custom(folder_path):
    custom_food_images = [folder_path + img_path for img_path in os.listdir(folder_path)]
    i = 0
    # fig, a = plt.subplots(len(custom_food_images), 2, figsize=(15, 5 * len(custom_food_images)))

    for img in custom_food_images:
        print(img)
        img = load_and_prep_image(img, scale=False)
        pred_prob = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[pred_prob.argmax()]
        top_5_i = (pred_prob.argsort())[0][-5:][::-1]
        values = pred_prob[0][top_5_i]
        labels = []
        # for x in range(5):
        #     labels.append(class_names[top_5_i[x]])

        # Plotting Image
        # a[i][0].imshow(img / 255.)
        print(f"Prediction: {pred_class}")

        # a[i][0].set_title(f"Prediction: {pred_class}   Probability: {pred_prob.max():.2f}")
        # a[i][0].axis(False)

        # Plotting Models Top 5 Predictions
        # a[i][1].bar(labels, values, color='orange');
        # a[i][1].set_title('Top 5 Predictions')
        i = i + 1


# pred_plot_custom("C:/Images/")
# convert keras model to tflite
# def get_file_size(file_path):
#     size = os.path.getsize(file_path)
#     return size
#

# def convert_bytes(size, unit=None):
#     if unit == "KB":
#         return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
#     elif unit == "MB":
#         return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
#     else:
#         return print('File size: ' + str(size) + ' bytes')
#

import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = tf.keras.models.load_model("C:\EfficientNetB1 (1).hdf5")
print("Model loaded successfully.")
print("Coverting.")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open("efficientnetb1.tflite", "wb") as f:
    f.write(tflite_model)

print("Done saving tflite")
# import os
# import csv
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
#
# # Function to load and preprocess image
# def load_and_prep_image(img_path, scale=False):
#     img = tf.io.read_file(img_path)
#     img = tf.image.decode_image(img, channels=3)
#     img = tf.image.resize(img, [224, 224])
#     if scale:
#         img /= 255.0  # normalize to [0,1] range
#     return img
#
#
# # Load class names from CSV file
# def load_class_names(csv_file_path):
#     class_names = []
#     with open(csv_file_path, mode='r', newline='') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip the header row
#         for row in reader:
#             class_names.append(row[1])
#     return class_names
#
#
# # Function to predict and plot images
# def pred_plot_custom(folder_path):
#   import os
#
#   custom_food_images = [folder_path + img_path for img_path in os.listdir(folder_path)]
#   i=0
#   fig,a =  plt.subplots(len(custom_food_images),2, figsize=(15, 5*len(custom_food_images)))
#
#   for img in custom_food_images:
#     img = load_and_prep_image(img, scale=False)
#     pred_prob = loaded_model.predict(tf.expand_dims(img, axis=0))
#     pred_class = class_names[pred_prob.argmax()]
#     top_5_i = (pred_prob.argsort())[0][-5:][::-1]
#     values = pred_prob[0][top_5_i]
#     labels = []
#     for x in range(5):
#       labels.append(class_names[top_5_i[x]])
#
#     # Plotting Image
#     a[i][0].imshow(img/255.)
#     a[i][0].set_title(f"Prediction: {pred_class}   Probability: {pred_prob.max():.2f}")
#     a[i][0].axis(False)
#
#     # Plotting Models Top 5 Predictions
#     a[i][1].bar(labels, values, color='orange');
#     a[i][1].set_title('Top 5 Predictions')
#     i=i+1
# # Define paths
#
# csv_file_path = r"C:\class_names.csv"
# #"C:\class_names.csv"
# #"C:\FoodVisiontest.keras"
# #"C:\FinalModel.hdf5"
# #"C:\EfficientNetB1 (1).hdf5"
# model_path = r"C:\FinalModel.hdf5"
# image_folder_path = r"C:\Images"
#
# # Load class names
# class_names = load_class_names(csv_file_path)
