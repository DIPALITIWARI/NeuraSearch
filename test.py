import pickle
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
#1->
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


    # 1. Load image
img = image.load_img(
                    "sample/ball.png",
                    target_size=(224, 224)
                )

                # 2. Convert image → numpy array
img_array = image.img_to_array(img)

                # 3. ResNet50 preprocessing
img_array = preprocess_input(img_array)

                # 4. Add batch dimension
                # 
img_array = np.expand_dims(img_array, axis=0)

                # 5. Extract features
features = model.predict(img_array, verbose=0)

                # 6. Convert to 1D
features = features.flatten()

                # 7. Normalize
final_features = features / np.linalg.norm(features)

# 8. ab hume final_features and feature_list ke beech mein distance nikalna hai
neighbors=NearestNeighbors(n_neighbors=5,algorithm="brute",metric="euclidean")# 6 isliye kyuki pehla vala khud vahi ata h
neighbors.fit(feature_list) # input data
# neighbors.kneighbors([final_features]) -> this returns.2 things distances and feature list
distances,indices=neighbors.kneighbors([final_features]) 
print(indices)

for index in indices[0][1:6]:
    temp_img = cv2.imread(filenames[index])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)


