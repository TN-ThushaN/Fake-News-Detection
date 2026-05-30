# 📰 Fake News Detection System

A Machine Learning and Natural Language Processing (NLP) web application that predicts whether a news article is **Fake** or **Real**. The system uses **TF-IDF vectorization** and a **Logistic Regression** classifier to analyze news content and provide predictions with confidence scores.

## 📌 Project Overview

The spread of misinformation has become a major challenge in the digital age. This project aims to help users identify potentially fake news articles by applying machine learning techniques to textual data.

Users can enter a news headline or article into the web application and receive an instant prediction.

---

## 🚀 Features

* Detects Fake and Real news articles
* Confidence percentage for predictions
* Machine Learning model trained on real-world datasets
* NLP text processing using TF-IDF
* Modern Flask web interface
* Responsive dark-themed UI
* Easy to deploy and extend

---

## 🛠️ Technologies Used

### Backend

* Python
* Flask

### Machine Learning

* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression

### Data Processing

* Pandas
* NumPy

### Frontend

* HTML5
* CSS3
* Bootstrap 5

---

## 📂 Project Structure

fake-news-detection/

├── dataset/

│   ├── Fake.csv

│   └── True.csv

├── templates/

│   └── index.html

├── static/

│   └── style.css

├── app.py

├── train_model.py

├── model.pkl

├── vectorizer.pkl

├── requirements.txt

└── README.md

---

## 📊 Dataset

This project uses the Fake and Real News Dataset available on Kaggle.

Dataset contains:

* Real news articles
* Fake news articles
* Thousands of training samples for machine learning

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/TN-Thushan/Fake-News-Detection.git
cd Fake-News-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_model.py
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📈 Model Performance

| Model               | Accuracy   |
| ------------------- | ---------- |
| Logistic Regression | ~95% - 98% |

*Accuracy may vary depending on preprocessing and dataset split.*

---

## 🎯 Future Improvements

* Deep Learning models (LSTM)
* Transformer models (BERT)
* News source credibility analysis
* User authentication system
* News history tracking
* API integration

---

## 👨‍💻 Author

**Thushan**

Data Science Undergraduate

Interested in:

* Machine Learning
* Artificial Intelligence
* Web Development
* Data Science

---

## 📄 License

This project is licensed under the MIT License.
