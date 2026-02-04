# Customer Churn Prediction App

This project predicts whether a customer is likely to churn (leave a service) using machine learning.  
It includes data preprocessing, model training, evaluation, and an interactive Streamlit web application for real-time predictions.

---

## Project Overview

Customer churn is an important business problem, as retaining customers is often more cost-effective than acquiring new ones.  
This application uses customer attributes such as age, gender, tenure, and monthly charges to estimate churn risk.

The project follows a realistic machine learning workflow:
1. Data exploration and preprocessing in a Jupyter notebook  
2. Model training and evaluation  
3. Saving trained models and preprocessing objects  
4. Deploying the model using a Streamlit application  

---

## Models Explored

The following models were trained and evaluated:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Classifier (SVC)  
- Decision Tree  
- Random Forest  

The final deployed model was selected based on performance metrics on the test set.

---

## Project Structure
churn project/
app.py # Streamlit application
notebook.ipynb # Data exploration and model training
customer_churn_data.csv # Dataset
model.pkl # Trained machine learning model
scaler.pkl # Feature scaler used during training
requirements.txt # Requirements for the project
README.md # Project documentation

## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/gabtlv/Customer-Churn-Prediction.git](https://github.com/gabtlv/Customer-Churn-Prediction.git)
   cd Customer-Churn-Prediction

2. **Install dependencies**
pip install -r requirements.txt

3. **Run the application**
streamlit run app.py

## Planned Enhancements

The following features are planned for future development as the project evolves:

- Display churn prediction probabilities to provide more nuanced risk assessment rather than only binary output  
- Integrate model performance metrics (accuracy, precision, recall, confusion matrix) directly into the application  
- Combine preprocessing and model inference into a single pipeline to improve robustness and maintainability  
- Allow users to compare predictions across multiple models (e.g., Logistic Regression, SVC, Random Forest)  
- Improve user interface layout and usability using Streamlit components such as sliders and multi-column layouts  
- Add basic model explainability to help users understand which features contribute most to churn predictions  
- Deploy the application publicly using Streamlit Cloud for live demonstration  

These enhancements are intended to move the project closer to a production-ready machine learning application.

