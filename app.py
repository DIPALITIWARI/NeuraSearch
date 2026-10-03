import os
import tensorflow
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm

# 1 ->
model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
# 2 ->
model.trainable = False

# 3 ->
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# 4 ->
def extract_features(img_path, model):

    # 1. Load image
    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    # 2. Convert image → numpy array
    img_array = image.img_to_array(img)

    # 3. ResNet50 preprocessing
    img_array = preprocess_input(img_array)

    # 4. Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Extract features
    features = model.predict(img_array, verbose=0)

    # 6. Convert to 1D
    features = features.flatten()

    # 7. Normalize
    final_features = features / np.linalg.norm(features)

    # 8. VERY IMPORTANT
    return final_features


filenames = []
# 5 ->
for img in os.listdir('images'):
    filenames.append(os.path.join('images', img))

# 6 ->
feature_list = []

for file in tqdm(filenames):
    curfeatures = extract_features(file, model)
    feature_list.append(curfeatures)


print(np.array(feature_list).shape)
# 7 ->
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))