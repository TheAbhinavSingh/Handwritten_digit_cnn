# Handwritten Digit Recognition using CNN

This project demonstrates **handwritten digit recognition** using Convolutional Neural Networks (CNN) on the MNIST dataset. The app is built with **Streamlit** so you can interactively draw or upload digits for prediction.

---

## Dataset
The MNIST dataset consists of:
- **60,000 training images** and **10,000 testing images** of handwritten digits (0–9).  
- Each image is **28x28 pixels**, grayscale.

---

## Model Architecture

| Layer Type | Details |
|------------|---------|
| Conv2D | 64 filters, kernel size (3,3), ReLU activation |
| BatchNormalization | - |
| MaxPooling2D | Pool size (2,2) |
| Conv2D | 64 filters, kernel size (3,3), ReLU activation |
| BatchNormalization | - |
| MaxPooling2D | Pool size (2,2) |
| Flatten | - |
| Dense | 128 units, ReLU activation |
| Dropout | Rate 0.2 |
| Dense | 64 units, ReLU activation |
| Dropout | Rate 0.2 |
| Dense | 10 units, Softmax activation |

---

## Training Details
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Epochs:** 10  
- **Validation Split:** 20%  

**Results:** ~98% accuracy on the MNIST test dataset.

---

## How to Run the Streamlit App

1. **Clone the repository:**
```bash
git clone https://github.com/AKSHATV25/Handwritten_digit_cnn.git
cd Handwritten_digit_cnn

Set up a virtual environment:

python -m venv venv
# Activate it:
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Ensure the model file is present:
digit_model.h5 must be in the same folder as app.py.

Run the Streamlit app:

streamlit run app.py


Open the URL shown in your browser to draw or upload digits for prediction.

Notes

Large files like digit_model.h5 may not display in GitHub but are included in the repository.

.gitignore is used to ignore the venv folder and other unnecessary files.
