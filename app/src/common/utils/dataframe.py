import pandas as pd


def query_to_pandas(sql: str, conn) -> pd.DataFrame:
    df = pd.read_sql(sql, conn)
    return df


def query_to_pandas_schema(sql: str, conn) -> str:
    df = query_to_pandas(sql, conn)
    return str(df.dtypes.to_markdown())


def interpolate_template(template: str, df) -> str:
    placeholders = {}
    for row_index, row in df.iterrows():
        cols = df.columns
        for col_index, col in enumerate(cols):
            f"placeholder_row_{row_index}_col_{col_index}"
            placeholders[f"placeholder_row_{row_index}_col_{col_index}"] = row[
                col
            ]
    return template.format(**placeholders)
