import numpy as np
from helper_function import load_and_prep_image
# import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import csv
import tensorflow as tf
import os

print(tf.__version__)

csv_file_path = r"C:\class_names.csv"
class_names = []
with open(csv_file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    # Skip the header row
    next(reader)
    # Iterate over rows and append class names to the list
    for row in reader:
        class_names.append(row[1])

print("Loaded classes: ", class_names)

model = load_model("C:\EfficientNetB1 (1).hdf5")

def pred_plot_custom(folder_path):
    custom_food_images = [folder_path + img_path for img_path in os.listdir(folder_path)]
    i = 0
    # fig, a = plt.subplots(len(custom_food_images), 2, figsize=(15, 5 * len(custom_food_images)))

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="C:\\amodeltflite.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for img in custom_food_images:
        print(img)
        img = load_and_prep_image(img, scale=False)
        pred_prob = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[pred_prob.argmax()]
        # top_5_i = (pred_prob.argsort())[0][-5:][::-1]
        # values = pred_prob[0][top_5_i]
        # labels = []
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


        # Add batch dimension and convert to float32
        img1 = np.expand_dims(img, axis=0).astype(np.float32)

        # Set the tensor to point to the input data
        interpreter.set_tensor(input_details[0]['index'], img1)

        # Run inference
        interpreter.invoke()

        # Get the results
        pred_prob1 = interpreter.get_tensor(output_details[0]['index'])
        pred_class1 = class_names[pred_prob1.argmax()]

        print(f"tflite Prediction: {pred_class1}")








        i = i + 1


pred_plot_custom("C:/Images/")

import tensorflow as tf
# model = tf.keras.models.load_model("C:\EfficientNetB1 (1).hdf5")
# print("Model loaded successfully.")
# print("Coverting.")
#
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# tflite_model = converter.convert()
#
# # Save the converted model to a .tflite file
# with open("efficientnetb1.tflite", "wb") as f:
#     f.write(tflite_model)

# Define the function to make predictions and plot the results
def pred_plot_custom_tflite(folder_path):
    # Load the TFLite model and allocate tensors
    tflite_model_path = "C:\\amodeltflite.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    print("Input")
    print(input_details)
    print("OUTPUT")

    output_details = interpreter.get_output_details()
    print(output_details)


# pred_plot_custom_tflite("C:\Images\\")

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
