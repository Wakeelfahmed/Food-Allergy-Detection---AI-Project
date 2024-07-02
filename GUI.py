import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

from tensorflow.keras.models import load_model
import tensorflow as tf
import os

import csv
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("food_ingredients_and_allergens.csv")

def recognize_allergens(user_input, food_column='Food Product'):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(df[food_column])
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_vectorizer.transform(df[food_column]))
    most_similar_index = np.argmax(cosine_similarities)
    food_data = df.iloc[most_similar_index]
    return food_data

class_names = [
    'apple pie', 'baby back ribs', 'Baklava', 'Beef carpaccio', 'beef tartare', 'beet salad', 'beignets', 'bibimbap',
    'bread pudding', 'Breakfast burrito', 'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'Carrot Cake',
    'ceviche', 'cheesecake', 'cheese plate', 'Bhicken curry', 'chicken quesadilla', 'Chicken Wings', 'Chocolate Cake',
    'Chocolate mousse', 'churros', 'Blam chowder', 'club sandwich', 'crab cakes', 'creme brulee', 'croque madame',
    'Cup cakes', 'Deviled eggs', 'Donuts', 'dumplings', 'edamame', 'eggs benedict', 'escargots', 'falafel',
    'filet mignon', 'Fish and chips', 'foie gras', 'french fries', 'french onion soup', 'french toast',
    'fried calamari', 'fried rice', 'frozen yogurt', 'garlic bread', 'gnocchi', 'greek salad',
    'grilled cheese sandwich', 'Grilled salmon', 'guacamole', 'gyoza', 'hamburger', 'hot and sour soup', 'hot dog',
    'huevos rancheros', 'hummus', 'Ice Cream', 'lasagna', 'lobster bisque', 'lobster roll sandwich',
    'macaroni and cheese', 'macarons', 'miso soup', 'mussels', 'nachos', 'omelette', 'onion rings', 'oysters',
    'pad thai', 'paella', 'pancakes', 'Panna cotta', 'peking duck', 'pho', 'pizza', 'pork chop', 'poutine', 'prime rib',
    'pulled pork sandwich', 'ramen', 'Ravioli', 'red velvet cake', 'risotto', 'Samosa', 'sashimi', 'Scallops',
    'seaweed salad', 'shrimp and grits', 'Spaghetti bolognese', 'spaghetti carbonara', 'Spring rolls', 'Steak',
    'strawberry shortcake', 'sushi', 'tacos', 'takoyaki', 'Tiramisu', 'Tuna tartare', 'Waffles'
]

print("Loaded classes: ", class_names)

model = load_model("EfficientNetB1 Model.hdf5")  #org

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img / 255.
    else:
        return img

def pred_plot_custom(folder_path):
    interpreter = tf.lite.Interpreter(model_path="Finalmodeltflite_2.4.1_fullepoch.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = load_and_prep_image(folder_path, scale=False)
    pred_prob = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[pred_prob.argmax()]

    img1 = np.expand_dims(img, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img1)
    interpreter.invoke()
    pred_prob1 = interpreter.get_tensor(output_details[0]['index'])
    pred_class1 = class_names[pred_prob1.argmax()]

    food_data = recognize_allergens(pred_class, food_column='Food Product')

    # Display predictions in the GUI using Treeview
    tree.delete(*tree.get_children())
    tree.insert('', 'end', values=('Prediction', pred_class))
    tree.insert('', 'end', values=('TFLite Prediction', pred_class1))
    tree.insert('', 'end', values=('Food Product', food_data['Food Product']))
    tree.insert('', 'end', values=('Main Ingredient', food_data['Main Ingredient']))
    tree.insert('', 'end', values=('Sweetener', food_data['Sweetener']))
    tree.insert('', 'end', values=('Fat/Oil', food_data['Fat/Oil']))
    tree.insert('', 'end', values=('Seasoning', food_data['Seasoning']))
    tree.insert('', 'end', values=('Allergens', food_data['Allergens']))
    tree.insert('', 'end', values=('Prediction', food_data['Prediction']))

def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")]
    )
    if not file_path:
        return

    image = Image.open(file_path)
    image = image.resize((400, 400))
    image_tk = ImageTk.PhotoImage(image)

    image_label.config(image=image_tk)
    image_label.image = image_tk

    pred_plot_custom(file_path)

root = tk.Tk()
root.title("Food Detection and Allergy Prediction")

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack(padx=10, pady=10)

# Create a button to open the file dialog
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

# Create a Treeview widget to display predictions
tree = ttk.Treeview(root, columns=('Attribute', 'Value'), show='headings')
tree.heading('Attribute', text='Attribute')
tree.heading('Value', text='Value')
tree.pack(pady=10)

# Run the application
root.mainloop()
