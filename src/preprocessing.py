from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(df):
    """Prepara datos: escalado y split train/test"""
    X = df.drop("Class").to_pandas()
    X = X.fillna(0)
    
    y = df.select("Class").to_pandas().values.ravel()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler