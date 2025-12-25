# 🕵️ Instagram Fake Account Detector

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fakeinstagramaccountdetection-hlvyrjcm4avpuqxjsdsbpt.streamlit.app/)

A Deep Learning-based web application to detect fake Instagram accounts using profile metadata. This project utilizes an Artificial Neural Network (ANN) to classify accounts as **Real** or **Fake** based on 11 specific features.

🔗 **[Live Demo: Click here to try the App](https://fakeinstagramaccountdetection-hlvyrjcm4avpuqxjsdsbpt.streamlit.app/)**

## 📂 Data Source
The models were trained using the **`nahiar/instagram_bot_detection`** dataset sourced from **Hugging Face**.

## 🚀 Features

-   **Deep Learning Model:** Uses a trained ANN (Keras/TensorFlow) for high-accuracy prediction (~96%).
-   **Dual Input Methods:**
    1.  **Manual Entry:** Manually input profile details (followers, bio length, etc.).
    2.  **Auto-Fetch (Instaloader):** Enter a username to automatically scrape data and predict.
-   **Smart Logic:** Automatically detects "Verified" (Blue Tick) accounts and flags them as Real without needing the model.
-   **Research Notebooks:** Includes three separate notebooks exploring Custom Logistic Regression, Lasso Regularization, and Deep Learning (ANN).

## 📊 Model Performance

Three models were trained and evaluated during the research phase. The **ANN** was selected for production due to its superior accuracy.

| Model | Accuracy | Notes |
| :--- | :--- | :--- |
| **Artificial Neural Network (ANN)** | **97%** | *Selected for App.* 2-layer dense network. Low loss (0.098). |
| **Logistic Regression (Lasso)** | 90.0% | Scikit-learn implementation with L1 regularization. |
| **Logistic Regression (Custom)** | ~90.0% | Custom NumPy implementation from scratch. |

*Note: The ANN achieved a validation accuracy of ~97% by Epoch 50/50.*

## 📂 Repository Structure

The repository consists of **9 main files**:

| File Name | Description |
| :--- | :--- |
| `app.py` | The main Streamlit application containing the UI, feature engineering logic, and prediction pipeline. |
| `InstaFakeID_Detection_v_1.ipynb` | Research notebook containing data preprocessing and a **Custom Logistic Regression** implementation using NumPy. |
| `InstaFakeID_Detection.ipynb` | Research notebook containing **Scikit-learn Lasso (L1)** Logistic Regression models. |
| `fake_insta_ANN_.ipynb` | Research notebook containing the **ANN** architecture design, training, and evaluation. |
| `instagram_fake_detector_ANN.h5` | The trained Artificial Neural Network model saved in H5 format. |
| `scaler_ANN.pkl` | The standard scaler object (Joblib) used to normalize input features before prediction. |
| `requirements.txt` | List of Python dependencies required to run the project. |
| `LICENSE` | MIT License. |
| `README.md` | Project documentation. |

## 🧠 Model Architecture (ANN)

The primary model used in the application is a Sequential Artificial Neural Network trained on extracted features.

**Input Features (11):**
*Profile Pic Presence, Username Length Ratio, Fullname Words, Fullname Length Ratio, Name==Username, Bio Length, External URL, Private Status, #Posts, #Followers, #Following.*

**Network Topology:**
1.  **Input Layer:** Dense (32 units) + BatchNormalization + ReLU + Dropout (0.2)
2.  **Hidden Layer:** Dense (64 units) + BatchNormalization + ReLU + Dropout (0.2)
3.  **Output Layer:** Dense (1 unit) + Sigmoid Activation

*Optimizer: Adam | Loss: Binary Crossentropy*

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **Deep Learning:** TensorFlow (Keras)
*   **Machine Learning:** Scikit-learn (Scaler, Lasso), NumPy (Custom LogReg)
*   **Data Processing:** Pandas, NumPy, Regular Expressions
*   **Web Scraping:** Instaloader

## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/instagram-fake-detector.git
    cd instagram-fake-detector
    ```

2.  **Install Dependencies**
    *Recommended Python Version: 3.10 or higher.*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## ⚠️ Limitations

**Instaloader & Rate Limits:**
The "Auto (Instaloader)" feature relies on Instagram's public API.
-   Instagram frequently blocks anonymous or frequent requests (HTTP 429/401 errors).
-   If the Auto-fetcher fails, the app is designed to fall back gracefully—users should use the **Manual Entry** tab in these cases.

## 📝 License

This project is licensed under the **MIT License**.
