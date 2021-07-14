import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

def get_date_range(df, include_end=True):
    if include_end:
        return pd.date_range(start=df['date'].min().replace(day=1).date(),
                           end=df['date'].max().replace(day=1).date() + relativedelta(months=1),
                           freq='MS')
    else:
        return pd.date_range(start=df['date'].min().replace(day=1).date(),
                           end=df['date'].max().replace(day=1).date() + relativedelta(months=1),
                           freq='MS')[0:-1]

def group_by_month(df):
    g = df.groupby(pd.Grouper(key=, freq='M'))

    # groups to a list of dataframes with list comprehension
    return [group for _, group in g]

def first_months(df, n, group=False):
    # group=True returns a list of DataFrames where each element is a month
    # n starts from 1,...,20
    date_range = get_date_range(df)

    if n > len(date_range) - 1:
        print('Number of months given exceeds the Date_range of the data')
        return

    months = date_range[0:n + 1]

    # greater than the start date and smaller than the end date
    mask = (df['date'] >= months[0]) & (df['date'] < months[-1])
    df2 = df.loc[mask]

    if group:
        g_months = group_by_month(df2)
        return months, df2, g_months
    else:
        return months, df2


def last_months(df, n, group=False):  # group=True returns a list of DataFrames where each element is a month
    # n starts from 1,...,20
    date_range = get_date_range(df)

    if n > len(date_range) - 1:
        print('Number of months given exceeds the Date_range of the data')
        return

    months = date_range[-n - 1:]

    # greater than the start date and smaller than the end date
    mask = (df['date'] >= months[0]) & (df['date'] < months[-1])
    df2 = df.loc[mask]

    if group:
        g_months = group_by_month(df2)
        return months, df2, g_months
    else:
        return months, df2


def time_interval(df, start, end, include_end=True, group=False):
    # start,end can be a datetime objects or integers
    # If they are integers, they describe integer months starting from 1,...,20
    # If they are datetime objects, the grouper will still group the dataframe by calandar months, not 30-day intervals
    date_range = get_date_range(df)

    if isinstance(start, int) and isinstance(end, int):
        if include_end:
            months = date_range[start - 1:end + 1]
        else:
            months = date_range[start - 1:end]

        mask = (df['date'] >= months[0]) & (df['date'] < months[-1])

    elif (isinstance(start, datetime.date) and isinstance(end, datetime.date)) or (
            isinstance(start, datetime.datetime) and isinstance(end, datetime.datetime)):
        if include_end:
            mask = (df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))
        else:
            mask = (df['date'] >= pd.to_datetime(start)) & (df['date'] < pd.to_datetime(end))

    else:
        print("'star' or 'end' are invalid object types")
        return

    df2 = df.loc[mask]

    if group:
        g_months = group_by_month(df2)
        return months, df2, g_months
    else:
        return months, df2


def random_undersampling(df, size, seed=42):
    # Returns: undersampled dataframe with length=size

    np.random.seed(seed)

    ind = np.random.choice(df.index, size=size, replace=False)
    ind = np.sort(ind)

    return df.loc[ind]