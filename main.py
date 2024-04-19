import numpy as np 
import pandas as pd 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def coffee_quality_prediction():
    # Read the dataset
    df = pd.read_csv("df_arabica_clean (1).csv")
    
    # Preprocessing steps...
    # Convert categorical variables to numerical labels...
    
    # Splitting data into features and target
    X = df.drop(['Category Two Defects'], axis=1)
    y = df['Category Two Defects']
    
    # Applying SMOTE for balancing the dataset
    smote = SMOTE(sampling_strategy='minority', k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=23)
    
    # Training the model
    model = RandomForestClassifier(n_estimators=150, random_state=23)
    model.fit(X_train, y_train)
    
    # Evaluating the model
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    classification_rep = classification_report(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)
    
    # Returning results
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'classification_report': classification_rep,
        'confusion_matrix': confusion_mat,
        'model': model
    }
