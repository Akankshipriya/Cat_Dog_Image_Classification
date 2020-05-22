import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt

C = ["Dog", "Cat"]


def img_arr(path):
    SIZE = 100  # 100 in txt-based
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (SIZE, SIZE)) # resize image to match model's expected sizing
    return new_array.reshape(-1, SIZE, SIZE, 1)  # return the image w

model = tf.keras.models.load_model("dog_cat-CNN.h5")

path=input("Enter the path of image to classify\n")

image_array = cv2.imread(os.path.join(path) ,cv2.IMREAD_GRAYSCALE)  # convert to array
plt.imshow(image_array, cmap='gray')  # graph it
plt.show()

pre = model.predict([img_arr(path)])
print("It is a: -",end=" ")
print(C[int(pre[0][0])])
