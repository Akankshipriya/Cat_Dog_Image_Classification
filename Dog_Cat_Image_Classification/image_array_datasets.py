import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

DIR = "PetImages"

C = ["Dog", "Cat"]

SIZE = 100
data = []

def create_data():
    for c in C:  # do dogs and cats

        data_path = os.path.join(DIR,c)  # create path to dogs and cats
        c_number = C.index(c)  # get the classification  (0 or a 1). 0=dog 1=cat

        for image in tqdm(os.listdir(data_path)):  # iterate over each image per dogs and cats
            try:
                image_arr = cv2.imread(os.path.join(data_path,image) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(image_arr, (SIZE, SIZE))  # resize to normalize data size
                data.append([new_array, c_number])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
create_data()
random.shuffle(data)
for sample in data[:10]:
    print(sample[1])
X = []
y = []

for features,label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, SIZE, SIZE, 1)
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()