import pandas as pd


def query_to_pandas(sql: str, conn) -> pd.DataFrame:
    df = pd.read_sql(sql, conn)
    return df


def query_to_pandas_schema(sql: str, conn) -> str:
    df = query_to_pandas(sql, conn)
    return str(df.dtypes.to_markdown())
