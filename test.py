import pandas as pd


data = pd.read_csv('data/AAPL.csv')


# "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"
data.drop(["Open", "Low", "Close", "Adj Close", "Volume"], axis=1, inplace=True)


# I want the procental change of a week to the next week
# and store it in a dataframe. So that I get something like this:
# Date, High
# 2021-01-01, 0
# 2021-01-08, -0.02
# 2021-01-15, 0.03
# 2021-01-22, 100
#...
# The first value is 0 because there is no previous week to compare to.
# The second value is -0.02 because the high of the second week is 2% lower than the high of the first week.
# The column of the week should be averaged over the days of the week, so I get a overall value for the week.
# The last value is 100 because the high of the last week is 100% higher than the high of the previous week.
def calculate_weekly_change(data: pd.DataFrame, column: str) -> pd.DataFrame:
    data["Date"] = pd.to_datetime(data["Date"])

    weekly = data.groupby(pd.Grouper(key="Date", freq="W")).agg({column: "mean"}).reset_index()
    weekly["Percentage_Change"] = weekly[column].pct_change() * 100
    weekly["Percentage_Change"] = weekly["Percentage_Change"].fillna(0)
    weekly["Percentage_Change"] = weekly["Percentage_Change"].round(4)

    return weekly

df = calculate_weekly_change(data, "High")
print(df[:4])


