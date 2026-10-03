# 🛍️ NeuraSearch — Fashion Recommendation System

NeuraSearch is an **AI-powered fashion recommendation system** that recommends visually similar fashion products based on an input image.

The system uses **Deep Learning and Computer Vision** to extract image features and find similar products from the Myntra fashion dataset.

## ✨ Features

- 🖼️ Upload a fashion/product image
- 🤖 Extract image features using **ResNet50**
- 🔍 Find visually similar fashion products
- ⚡ Fast similarity search using **K-Nearest Neighbors**
- 👕 Supports different fashion categories such as clothing, footwear, sarees, jerseys, etc.
- 🌐 Simple web interface using **Streamlit**

## 🧠 How It Works

```text
Input Image
     ↓
Preprocessing
     ↓
ResNet50 Feature Extraction
     ↓
Feature Vector
     ↓
Nearest Neighbors Search
     ↓
Similar Fashion Products
```

### 1. Image Input
The user uploads an image of a fashion product.

### 2. Feature Extraction
The uploaded image is passed through a pretrained **ResNet50** model to extract meaningful visual features.

### 3. Similarity Search
The extracted feature vector is compared with the stored feature vectors using **Nearest Neighbors**.

### 4. Recommendations
The system returns the most visually similar products from the dataset.

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **ResNet50**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **Streamlit**
- **Pickle**

## 📂 Project Structure

```text
NeuraSearch/
│
├── app.py
├── main.py
├── test.py
├── Untitled.ipynb
├── styles.csv
├── img.png
│
├── myntradataset/
│   └── styles.csv
│
└── sample/
    ├── ball.png
    ├── jersey.png
    ├── jj.png
    ├── saree.png
    └── sneaker.png
```

> **Note:** The original Myntra image dataset and generated `embeddings.pkl` file are not stored in this repository because of their large size.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/DIPALITIWARI/NeuraSearch.git
cd NeuraSearch
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies:

```bash
pip install tensorflow streamlit numpy pandas scikit-learn pillow
```

## ▶️ Run the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📸 Example

The user uploads a fashion image:

```text
       Input Image
           ↓
     ResNet50 Model
           ↓
    Feature Extraction
           ↓
 Similarity Calculation
           ↓
  Recommended Products
```

## 🔬 Model

The project uses **ResNet50**, a pretrained convolutional neural network originally trained on ImageNet.

Instead of using ResNet50 directly for classification, the model is used as a **feature extractor**. These feature representations are then used to find visually similar images.

## 📊 Dataset

The project uses the **Myntra fashion dataset**, containing thousands of fashion product images and associated product information.

Due to GitHub's file-size limitations, the complete image dataset and generated embedding files are not included in this repository.

## 🔮 Future Improvements

- Add category-aware recommendations
- Improve similarity ranking
- Add text-based product search
- Combine image and text features
- Add product metadata filtering
- Deploy the application online
- Add personalized recommendations based on user preferences

## 👩‍💻 Author

**Dipali Tiwari**

B.Tech — Computer Science Engineering

---

⭐ If you find this project useful, consider giving the repository a star!
