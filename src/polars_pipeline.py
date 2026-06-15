import polars as pl
import time

def load_and_process_polars(filepath, infer_schema_length=10000):
    """Carga y procesa datos con Polars"""
    times = {}
    
    # Lectura
    start = time.time()
    df = pl.read_csv(filepath, infer_schema_length=infer_schema_length, ignore_errors=True)
    times['lectura'] = time.time() - start
    
    # Filtrado
    start = time.time()
    df = df.filter(pl.col("Amount") > 0)
    times['filtrado'] = time.time() - start
    
    # Feature engineering
    start = time.time()
    df = df.with_columns(
        (pl.col("Amount") / pl.col("Amount").max()).alias("amount_scaled")
    )
    times['feature_eng'] = time.time() - start
    
    # Agregación
    start = time.time()
    stats = df.group_by("Class").agg(pl.col("Amount").mean().alias("avg_amount"))
    times['agregacion'] = time.time() - start
    
    # Join
    start = time.time()
    df = df.join(stats, on="Class")
    times['join'] = time.time() - start
    
    return df, times

def get_system_info():
    """Obtiene información del sistema"""
    import psutil
    
    cores = psutil.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    dataset_size_mb = 1.67
    
    return {
        'cores': cores,
        'ram_gb': ram_gb,
        'dataset_size_mb': dataset_size_mb
    }