import pandas as pd;
import matplotlib.pyplot as plt;

# DATA_NAME = "AAPL"
# DATA_NAME = "AMZN"
# DATA_NAME = "FB"
# DATA_NAME = "GOOGL"
# DATA_NAME = "TSLA"

DATA_NAMES = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]

for DATA_NAME in DATA_NAMES:
    print(f"Plotting {DATA_NAME} stock price")
    df = pd.read_csv(f"./data/{DATA_NAME}.csv")
    df.plot(x="Date", y="Close", title=f"{DATA_NAME} Stock Price", ylabel="Close Price", xlabel="Date")
    plt.gcf().set_size_inches(15, 7)
    plt.savefig(f"./plots/{DATA_NAME}_stock_price.png")



# df = pd.read_csv(f"./data/{DATA_NAME}.csv")
# df.plot(x="Date", y="Close", title="AAPL Stock Price", ylabel="Close Price", xlabel="Date")
# plt.gcf().set_size_inches(15, 7)
# plt.savefig(f"./plots/{DATA_NAME}_stock_price.png")