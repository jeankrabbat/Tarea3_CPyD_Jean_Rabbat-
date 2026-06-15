# Tarea 3: Polars vs Pandas

## 1. Descripción del Problema

Análisis comparativo de rendimiento entre Polars y Pandas en un pipeline de detección de fraude en tarjetas de crédito. Se evalúan operaciones de lectura, filtrado, feature engineering, agregación y joins.

---

## 2. Fuente del Dataset

**Nombre:** Credit Card Fraud Detection  
**Origen:** Kaggle (Universidad Libre de Bruselas)  
**URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
**Tamaño:** 284,807 transacciones, 31 variables

---

## 3. Requisitos de Software

- Python 3.8+
- polars, pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

```bash
pip install -r requirements.txt
```

---

## 4. Instalación

```bash
git clone https://github.com/jeankrabbat/Tarea3_CPyD_JeanRabbat.git
cd Tarea3_CPyD_JeanRabbat

# Descargar dataset de Kaggle
# Colocar creditcard.csv en data/raw/

pip install -r requirements.txt
```

---

## 5. Ejecución

```bash
jupyter notebook
# Abrir: notebooks/02_main_analysis.ipynb
# Ejecutar todas las celdas
```

---

## 6. Estructura del Repositorio

```
Tarea3_JeanRabbat/
├── data/raw/
│   └── creditcard.csv
├── src/
│   ├── polars_pipeline.py
│   ├── pandas_pipeline.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── train_models.py
├── notebooks/
│   └── analysis.ipynb
├── figures/
├── report/
│   └── Report_JeanRabbat_Tarea3.pdf
├── requirements.txt
└── README.md
```

---

## 7. Resultados

| Operación | Polars (s) | Pandas (s) | Speedup |
|-----------|-----------|-----------|---------|
| Lectura | 0.1498 | 1.2206 | 8.67x |
| Filtrado | 0.0163 | 0.0446 | 2.73x |
| Feature Eng | 0.0015 | 0.0043 | 2.95x |
| Agregación | 0.0029 | 0.0052 | 1.79x |
| Join | 0.0191 | 0.1302 | 6.82x |
| **Total** | **0.1806** | **1.4050** | **7.78x** |

**Conclusión:** Polars es 7.78x más rápido que Pandas.

---

## 8. Modelos ML

| Modelo | Accuracy | F1 | Tiempo |
|--------|----------|-----|--------|
| Logistic Regression | 100% | 100% | 0.55s |
| Random Forest | 100% | 100% | 162.41s |
| XGBoost | 99.99% | 98.40% | 1.64s |

**Mejor modelo:** XGBoost (balance rendimiento/velocidad)

---

## Referencias

- Polars: https://docs.pola-rs/
- Pandas: https://pandas.pydata.org/
- Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

**Autor:** Jean Carlo Rabbat  
**Curso:** Computación Paralela y Distribuida  
**Profesor:** Johansell Villalobos Cubillo
