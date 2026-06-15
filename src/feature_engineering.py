import polars as pl

def create_features(df):
    """Crea características adicionales"""
    df_engineered = df.with_columns(
        (pl.col("Amount") / pl.col("Amount").max()).alias("amount_scaled"),
        pl.col("Amount").std().over("Class").alias("amount_std_by_class")
    )
    
    return df_engineered