import pandas as pd
import time

def load_and_process_pandas(filepath):
    """Carga y procesa datos con Pandas"""
    times = {}
    

    start = time.time()
    df = pd.read_csv(filepath)
    times['lectura'] = time.time() - start
    

    start = time.time()
    df = df[df["Amount"] > 0]
    times['filtrado'] = time.time() - start
    

    start = time.time()
    df["amount_scaled"] = df["Amount"] / df["Amount"].max()
    times['feature_eng'] = time.time() - start
    

    start = time.time()
    stats = df.groupby("Class")["Amount"].mean().reset_index()
    stats.columns = ["Class", "avg_amount"]
    times['agregacion'] = time.time() - start
    

    start = time.time()
    df = df.merge(stats, on="Class")
    times['join'] = time.time() - start
    
    return df, times