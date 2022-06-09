import pandas as pd


def make_stationary_df( data, selected_cols ):
    new_df = data.copy()
    for col in selected_cols:
        new_df[f"{col}_diff"] = new_df[col].diff()
    new_df = new_df.dropna()
    return new_df


def date_categorization( df ):
    df = df.copy()
    df = df.reset_index()

    df["Year"] = df["dates"].dt.year
    df["Month"] = df["dates"].dt.month
    df["Week"] = df["dates"].dt.week
    df["Day"] = df["dates"].dt.day
    df["Dayofweek"] = df["dates"].dt.dayofweek
    df["Dayofyear"] = df["dates"].dt.dayofyear
    df["Weekofyear"] = df["dates"].dt.weekofyear
    df["py_Year"] = df["Year"] - 1

    return df

def merge_years( df ):
    df = df.copy()
    df = df.reset_index()
    df["py_dates"] = df["dates"] + pd.offsets.DateOffset(years=-1)

    data1 = df[["dates", "id_stores", "id_products", "sales", "prices"]].copy()
    data2 = df[["py_dates", "id_stores", "id_products", "py_sales", "py_prices"]].copy()
    data2.columns = ["dates", "id_stores", "id_products", "sales", "prices"]

    aux = pd.concat([data1,data2], axis=0)
    return aux