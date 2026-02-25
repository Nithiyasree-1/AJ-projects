# Anemia Prediction AI 🩺

An advanced clinical diagnostic assistant built with Machine Learning and Flask. This application predicts the probability of anemia based on clinical blood reports.

## 🚀 Features
- **Production-Level ML Pipeline**: Includes preprocessing, feature engineering, and automated model selection.
- **High Accuracy**: Logistic Regression model with **99.65% accuracy** and **1.0 ROC-AUC**.
- **Modern UI**: Dark-themed, responsive Glassmorphism design for clinical use.
- **Vercel Ready**: Pre-configured for seamless deployment to Vercel.

## 📊 ML Pipeline Details
1. **Preprocessing**: IQR outlier capping, scaling, and missing value handling.
2. **Feature Engineering**: Derived medical features including Hematocrit and Estimated RBC.
3. **Algorithms**: Trained on Logistic Regression, Random Forest, XGBoost, and SVM.
4. **Optimization**: Automated ensemble methods for performance boosting.

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **ML Libraries**: Scikit-learn, XGBoost, Pandas, NumPy, Imbalanced-learn
- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript (Fetch API)
- **Deployment**: Vercel

## 📦 Deployment
This project is configured for Vercel. 
1. Push to GitHub.
2. Connect the repository to Vercel.
3. Done!

## 🧪 Quick Start (Local)
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python app.py`
3. Visit: `http://127.0.0.1:5000`

---
*Disclaimer: This is an AI-powered tool for informational purposes. Always consult with a qualified medical professional.*
