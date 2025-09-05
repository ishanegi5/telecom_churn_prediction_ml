# 📞 Telecom Customer Churn Prediction (ML Project)

This project predicts customer churn in the telecom sector using various **Machine Learning algorithms**.  
We built and compared models, tuned hyperparameters, and identified the best-performing algorithm.

---

## 🚀 Project Highlights
- Repository: **telecom_churn_prediction_ml**
- Dataset: Telecom Churn Dataset  
- Models implemented:
  - Logistic Regression ✅ (Best Model - 82.17% accuracy)
  - Support Vector Machine (SVM)
  - XGBoost
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier
  - Decision Tree Classifier
  - Bernoulli Naive Bayes
  - Gaussian Naive Bayes
- Hyperparameter tuning performed on all models
- Logistic Regression was saved using **joblib** for deployment

---

## 📊 Results
| Model                     | Accuracy  |
|----------------------------|-----------|
| Logistic Regression        | **82.17%** |
| SVM                        | 81.88%    |
| XGBoost                    | 81.67%    |
| KNN                        | 81.46%    |
| Random Forest              | 81.03%    |
| Decision Tree              | 80.25%    |
| Bernoulli Naive Bayes      | 73.22%    |
| Gaussian Naive Bayes       | 69.60%    |

---

## ⚙️ Technologies Used
- Python
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn
- Joblib (for model saving)

---

## 📂 Project Structure
telecom_churn_prediction_ml/
│── data/ # Dataset (not included due to size)
│── notebooks/ # Jupyter notebooks
│── models/ # Saved models (joblib)
│── scripts/ # Python scripts
│── README.md # Documentation

yaml
Copy code

---

## 💾 Saving Best Model
```python
import joblib
joblib.dump(pipeline_lr, "models/logistic_regression_model.pkl")
✨ Author
👤 Isha Negi

GitHub: ishanegi5

LinkedIn: Isha Negi
