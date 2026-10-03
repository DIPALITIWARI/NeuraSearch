import streamlit as st
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from PIL import Image

st.title("FASHION RECOMMENDER SYSTEM")



feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))


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


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0    

def feature_extraction(img_path,model):
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
neighbors = NearestNeighbors(
    n_neighbors=5,
    algorithm="brute",
    metric="euclidean"
)

# final -> 
def recommend(final_features,feature_list):
    neighbors=NearestNeighbors(n_neighbors=5,algorithm="brute",metric="euclidean")
    neighbors.fit(feature_list) 
    distances,indices=neighbors.kneighbors([final_features]) 
    return indices

neighbors.fit(feature_list)
uploaded_file= st.file_uploader("Choose any of your fav image")

if uploaded_file is not None:
   if save_uploaded_file(uploaded_file):
        display_img=Image.open(uploaded_file)
        st.header("upload hogyaa oyy")
        st.image(display_img)
        features=feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        st.text(features)
        indices=recommend(features,feature_list)

        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4 ]])



   else:
        st.header("some error occured while uploading the file")


# import streamlit as st
# import os
# import numpy as np
# import pickle
# import tensorflow
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from sklearn.neighbors import NearestNeighbors
# from PIL import Image


# # -----------------------------
# # PAGE
# # -----------------------------

# st.set_page_config(
#     page_title="Fashion Recommender",
#     layout="wide"
# )

# st.title("👗 FASHION RECOMMENDER SYSTEM")

# st.write("App started successfully!")


# # -----------------------------
# # LOAD DATA
# # -----------------------------

# st.write("Loading embeddings...")

# feature_list = pickle.load(
#     open("embeddings.pkl", "rb")
# )

# filenames = pickle.load(
#     open("filenames.pkl", "rb")
# )

# st.write("Embeddings loaded!")
# st.write("Number of images:", len(filenames))
# st.write("Feature shape:", np.array(feature_list).shape)


# # -----------------------------
# # LOAD MODEL
# # -----------------------------

# @st.cache_resource
# def load_model():

#     base_model = ResNet50(
#         weights="imagenet",
#         include_top=False,
#         input_shape=(224, 224, 3)
#     )

#     base_model.trainable = False

#     model = tensorflow.keras.Sequential([
#         base_model,
#         GlobalMaxPooling2D()
#     ])

#     return model


# st.write("Loading ResNet50...")

# model = load_model()

# st.write("ResNet50 loaded successfully!")


# # -----------------------------
# # NEAREST NEIGHBORS
# # -----------------------------

# neighbors = NearestNeighbors(
#     n_neighbors=5,
#     algorithm="brute",
#     metric="euclidean"
# )

# neighbors.fit(feature_list)

# st.write("Nearest Neighbors ready!")


# # -----------------------------
# # SAVE UPLOADED FILE
# # -----------------------------

# def save_uploaded_file(uploaded_file):

#     try:

#         os.makedirs("uploads", exist_ok=True)

#         file_path = os.path.join(
#             "uploads",
#             uploaded_file.name
#         )

#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         return file_path

#     except Exception as e:

#         st.error(f"Upload error: {e}")

#         return None


# # -----------------------------
# # FEATURE EXTRACTION
# # -----------------------------

# def feature_extraction(img_path, model):

#     img = image.load_img(
#         img_path,
#         target_size=(224, 224)
#     )

#     img_array = image.img_to_array(img)

#     img_array = preprocess_input(img_array)

#     img_array = np.expand_dims(
#         img_array,
#         axis=0
#     )

#     features = model.predict(
#         img_array,
#         verbose=0
#     )

#     features = features.flatten()

#     final_features = features / np.linalg.norm(features)

#     return final_features


# # -----------------------------
# # UPLOAD IMAGE
# # -----------------------------

# uploaded_file = st.file_uploader(
#     "Upload your favourite clothes image",
#     type=["jpg", "jpeg", "png"]
# )


# # -----------------------------
# # RECOMMENDATION
# # -----------------------------

# if uploaded_file is not None:

#     st.write("Image uploaded!")

#     file_path = save_uploaded_file(uploaded_file)

#     if file_path is not None:

#         display_img = Image.open(uploaded_file)

#         st.subheader("Your Image")

#         st.image(
#             display_img,
#             width=300
#         )

#         st.write("Extracting features...")

#         features = feature_extraction(
#             file_path,
#             model
#         )

#         st.success("Feature extraction completed!")

#         st.write(
#             "Feature shape:",
#             features.shape
#         )

#         st.write("Finding similar clothes...")

#         distances, indices = neighbors.kneighbors(
#             [features]
#         )

#         st.success("Recommendations found!")

#         st.subheader("✨ Recommended Clothes")

#         cols = st.columns(5)

#         for i, index in enumerate(indices[0]):

#             with cols[i]:

#                 st.image(
#                     filenames[index],
#                     width="stretch"
#                 )

#                 st.caption(
#                     f"Distance: {distances[0][i]:.2f}"
#                 )
