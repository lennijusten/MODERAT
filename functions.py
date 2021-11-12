import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

def get_date_range(df, include_end=True):
    if include_end:
        return pd.date_range(start=df['Date'].min().replace(day=1).date(),
                           end=df['Date'].max().replace(day=1).date() + relativedelta(months=1),
                           freq='MS')
    else:
        return pd.date_range(start=df['Date'].min().replace(day=1).date(),
                           end=df['Date'].max().replace(day=1).date() + relativedelta(months=1),
                           freq='MS')[0:-1]

def group_by_month(df):
    g = df.groupby(pd.Grouper(key='Date', freq='M'))

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
    mask = (df['Date'] >= months[0]) & (df['Date'] < months[-1])
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
    mask = (df['Date'] >= months[0]) & (df['Date'] < months[-1])
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

        mask = (df['Date'] >= months[0]) & (df['Date'] < months[-1])

    elif (isinstance(start, datetime.date) and isinstance(end, datetime.date)) or (
            isinstance(start, datetime.datetime) and isinstance(end, datetime.datetime)):
        if include_end:
            mask = (df['Date'] >= pd.to_datetime(start)) & (df['Date'] <= pd.to_datetime(end))
        else:
            mask = (df['Date'] >= pd.to_datetime(start)) & (df['Date'] < pd.to_datetime(end))

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


def chunk_by_number(df, n_chunks, method='sequential'):
    # df is a dateframe sorted by date, n_chunks is the number of chunks to split the dataframe into
    # Returns list of dataframes

    if method == 'random':
        df = df.sample(frac=1)
    elif method == 'sequential':
        pass
    else:
        print("Unknown 'method' keyword. (sequential, random)")
        return

    return np.array_split(df, n_chunks)