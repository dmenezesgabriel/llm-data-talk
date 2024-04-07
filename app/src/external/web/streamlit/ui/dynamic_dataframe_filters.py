import warnings
from typing import Union

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


def _convert_dataframe_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=UserWarning)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    warnings.resetwarnings()
    return df


def _categorical_multiselect(
    df: pd.DataFrame,
    column: str,
    st_column,
    key: Union[str, None] = None,
):
    user_cat_input = st_column.multiselect(
        f"Values for {column}",
        df[column].unique(),
        default=list(df[column].unique()),
        key=f"{column}_{key}_multiselect",
    )
    df = df[df[column].isin(user_cat_input)]
    return df


def _numeric_slider(
    df: pd.DataFrame,
    column: str,
    st_column,
    key: Union[str, None] = None,
):
    _min = float(df[column].min())
    _max = float(df[column].max())
    step = (_max - _min) / 100
    user_num_input = st_column.slider(
        f"Values for {column}",
        min_value=_min,
        max_value=_max,
        value=(_min, _max),
        step=step,
        key=f"{column}_{key}_slider",
    )
    df = df[df[column].between(*user_num_input)]
    return df


def _date_selector(
    df: pd.DataFrame,
    column: str,
    st_column,
    key: Union[str, None] = None,
):
    user_date_input = st_column.date_input(
        f"Values for {column}",
        value=(
            df[column].min(),
            df[column].max(),
        ),
        key=f"{column}_{key}_date_input",
    )
    if len(user_date_input) == 2:
        user_date_input = tuple(map(pd.to_datetime, user_date_input))
        start_date, end_date = user_date_input
        df = df.loc[df[column].between(start_date, end_date)]
    return df


def _text_search(
    df: pd.DataFrame,
    column: str,
    st_column,
    key: Union[str, None] = None,
):
    user_text_input = st_column.text_input(
        f"Substring or regex in {column}",
        key=f"{column}_{key}_text_input",
    )
    if user_text_input:
        df = df[df[column].astype(str).str.contains(user_text_input)]
    return df


def _filter_factory(
    column_type: str,
    df: pd.DataFrame,
    column: str,
    st_column,
    key: Union[str, None] = None,
) -> pd.DataFrame:
    functions = {
        "categorical": _categorical_multiselect,
        "numeric": _numeric_slider,
        "date": _date_selector,
        "default": _text_search,
    }
    return functions[column_type](df, column, st_column, key)


def _get_column_type(column: str, df: pd.DataFrame) -> str:
    column_type = "default"
    # Treat columns with < 10 unique values as categorical
    if (
        isinstance(df[column].dtype, pd.CategoricalDtype)
        or df[column].nunique() < 10
    ):
        column_type = "categorical"
    elif is_numeric_dtype(df[column]):
        column_type = "numeric"
    elif is_datetime64_any_dtype(df[column]):
        column_type = "date"
    else:
        column_type = "default"
    return column_type


def dynamic_dataframe_filter_ui(
    df: pd.DataFrame, key: Union[str, None] = None
) -> pd.DataFrame:
    add_filters = st.checkbox("Add filters", key=f"{key}_checkbox")

    if not add_filters:
        return df

    df = df.copy()
    df = _convert_dataframe_datetime_columns(df)

    with st.container():
        to_filter_columns = st.multiselect(
            "Filter dataframe on", df.columns, key=f"{key}_multiselect"
        )
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            column_type = _get_column_type(column, df)
            df = _filter_factory(
                column_type=column_type,
                df=df,
                column=column,
                st_column=right,
                key=key,
            )
    return df
