from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import time

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Entrena los 3 modelos y retorna resultados"""
    results = {}
    
    # Logistic Regression
    start = time.time()
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train, y_train)
    time_lr = time.time() - start
    
    y_pred_lr = model_lr.predict(X_test)
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'auc': roc_auc_score(y_test, y_pred_lr),
        'tiempo': time_lr,
        'y_pred': y_pred_lr
    }
    
    # Random Forest
    start = time.time()
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    time_rf = time.time() - start
    
    y_pred_rf = model_rf.predict(X_test)
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'auc': roc_auc_score(y_test, y_pred_rf),
        'tiempo': time_rf,
        'y_pred': y_pred_rf
    }
    
    # XGBoost
    start = time.time()
    model_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    time_xgb = time.time() - start
    
    y_pred_xgb = model_xgb.predict(X_test)
    results['XGBoost'] = {
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'auc': roc_auc_score(y_test, y_pred_xgb),
        'tiempo': time_xgb,
        'y_pred': y_pred_xgb
    }
    
    return results