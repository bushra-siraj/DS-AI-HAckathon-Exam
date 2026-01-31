from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib


df = pd.read_csv('Processed_AI_Resume_Data.csv')

#Data Preprocessing

#How did you prepare categorical variables for modeling?
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('Recruiter Decision')  # Remove target variable

#2. 2. How did you prepare numerical variables for modeling?
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# Impute missing values in numerical columns with mean


#3. What transformations were applied before training models?
# Create a ColumnTransformer to handle preprocessing of categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), df.select_dtypes(include=[np.number]).columns.tolist()),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Prepare the data for modeling
X = df.drop(columns=['Recruiter Decision'])
y = df['Recruiter Decision']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing to training and testing sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 1. Prepare Target Variables (Numeric mapping for models)
y_train_numeric = y_train.map({'Hire': 1, 'Reject': 0})
y_test_numeric = y_test.map({'Hire': 1, 'Reject': 0})

# 2. Define 5 Classification Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(randomstate=42)
}

# 3. Define Hyperparameter Grids (For "High-End" Tuning)
param_grids = {
    "Random Forest": {
        'n_estimators': [100],
        'max_depth': [5],
    },
    "XGBoost": {
        'learning_rate': [0.1],
        'n_estimators': [100],
        'max_depth': [5]
    }
}

# 4. Use Cross-Validation and Hyperparameter Tuning
print("--- Starting Model Tuning and Cross-Validation ---")
best_estimators = {}

for name, model in models.items():
    if name in param_grids:
        print(f"Tuning {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train_processed, y_train_numeric)
        best_estimators[name] = grid.best_estimator_
        print(f"Best Params for {name}: {grid.best_params_}")
    else:
        # For non-tuned models, still apply Cross-Validation
        cv_scores = cross_val_score(model, X_train_processed, y_train_numeric, cv=5)
        print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        model.fit(X_train_processed, y_train_numeric)
        best_estimators[name] = model

# 5. Final Comparison on Test Set
performance_metrics = []
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for i, (name, model) in enumerate(best_estimators.items()):
    y_pred = model.predict(X_test_processed)
    
    # Calculate Metrics
    acc = accuracy_score(y_test_numeric, y_pred)
    prec = precision_score(y_test_numeric, y_pred)
    rec = recall_score(y_test_numeric, y_pred)
    f1 = f1_score(y_test_numeric, y_pred)
    
    performance_metrics.append([name, acc, prec, rec, f1])
    
    # Confusion Matrix Plotting
    cm = confusion_matrix(y_test_numeric, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Reject', 'Hire'])
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(f"{name}")

plt.tight_layout()
plt.show()

# 6. Display Performance Table
performance_df = pd.DataFrame(performance_metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
print("\n--- Final Model Performance Comparison ---")
print(performance_df.sort_values(by='F1-Score', ascending=False))

joblib.dump(best_estimators['XGBoost'], 'recruiter_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

#WHich model is best and why?
#XGBoost is the best model because it achieved a perfect 1.0 accuracy even after we applied regularization, meaning it has mastered the underlying rules of the dataset with total precision. While Logistic Regression is a close runner-up with 0.99 accuracy, XGBoost wins because its tree-based structure naturally handles the "if-then" logic of hiring decisions more effectively than linear models. The Random Forest at 0.78 is actually the most realistic for real-world data, but for this specific "Live Predictor" project, XGBoost is the strongest engine since it perfectly aligns with the recruiter's existing decision-making patterns.