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


print(tf.__version__)  # must be 2.4.1 (python 3.8)

# class_names = []      #use csv or loading or hardcoded below

# with open("C:\class_names.csv", mode='r', newline='') as file:
#     reader = csv.reader(file)
#     # Skip the header row
#     next(reader)
#     # Iterate over rows and append class names to the list
#     for row in reader:
#         class_names.append(row[1])
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
    """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img / 255.
    else:
        return img


def pred_plot_custom(folder_path):
    custom_food_images = [folder_path + img_path for img_path in os.listdir(folder_path)]
    # i = 0
    # fig, a = plt.subplots(len(custom_food_images), 2, figsize=(15, 5 * len(custom_food_images)))

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="Finalmodeltflite_2.4.1_fullepoch.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for img in custom_food_images:
        print(img)  # show image path & name
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
        food_data = recognize_allergens(pred_class, food_column='Food Product')
        print(food_data, "\n")

pred_plot_custom("C:/Images/")  # Run the model and make allergen predictions on all the images in the folder