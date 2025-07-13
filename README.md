# 💓 Heart Disease Prediction Using Machine Learning

This project compares multiple machine learning classification models to predict the presence of heart disease using medical features. The project also implements hyperparameter tuning (GridSearchCV and RandomizedSearchCV) and evaluates models using metrics like Accuracy, Precision, Recall, and F1 Score.

---

## 📁 Dataset

- **Name**: Heart Disease Prediction Dataset
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Records**: 918 patients
- **Target Variable**: `HeartDisease` (1 = disease present, 0 = no disease)

---

## 🛠️ Project Workflow

### 1. Data Preprocessing
- Handled categorical features using one-hot encoding.
- Scaled all features using `StandardScaler`.

### 2. Models Trained
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### 3. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

### 4. Hyperparameter Tuning
- **Random Forest**: Tuned using `GridSearchCV`
- **SVM**: Tuned using `RandomizedSearchCV`

---

## 📊 Results Summary

| Model                    | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| SVM (default)            | 0.880    | ~0.89     | ~0.89  | **0.891** ✅  
| Random Forest (tuned)    | 0.870    | 0.895     | 0.878  | 0.887  
| SVM (tuned)              | 0.859    | 0.901     | 0.850  | 0.875  
| Logistic Regression      | ~0.85    | ~0.87     | ~0.86  | 0.870  
| Decision Tree            | ~0.84    | ~0.85     | ~0.87  | 0.865  

> 🔍 **Best Model**: SVM with default parameters (F1 Score: 0.891)

---

## 📈 Visualizations

- ✅ F1 Score Bar Chart (model comparison)
- ✅ Confusion Matrix for SVM
- ✅ Feature Importance Plot from Random Forest

---

## 📄 Conclusion

- SVM gave the best results without tuning, which shows how sometimes default settings can work very well on structured data.
- Hyperparameter tuning improved model performance and helped validate the choice of the final model.
- This project demonstrates a full ML pipeline from preprocessing to final evaluation.

---

## 💡 Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
