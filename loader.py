import os
import shutil
import kagglehub
import pandas as pd

# Represents a record of stock prices with various attributes such as date, open, high, low, close, volume, and adjusted close.
class StockPriceRecord:
  def __init__(self, id, date, open, high, low, close, volume, adj_close):
    self.id = id
    self.date = date
    self.open = open
    self.high = high
    self.low = low
    self.close = close
    self.volume = volume
    self.adj_close = adj_close

  def __repr__(self):
    return f"({self.id} || {self.date})"

  def __eq__(self, __o: object) -> bool:
     return self.check_if_same_stock(__o) and self.date == __o.date

  def check_if_same_stock(self, other):
    return self.id == other.id

def read_csv(path):
    df = pd.read_csv(path)
    rows_as_dicts = df.to_dict('records')
    return rows_as_dicts


download_path = kagglehub.dataset_download("ashbellett/australian-historical-stock-prices")
print("Downloaded to:", download_path)

destination_folder = "./dataset"

if os.path.exists(destination_folder):
    print("Removing existing folder:", destination_folder)
    shutil.rmtree(destination_folder)
else:
    print("Creating folder:", destination_folder)

os.makedirs(destination_folder)
print("Moving files to:", destination_folder)

destination_path = os.path.join(destination_folder, os.path.basename(download_path))
shutil.move(download_path, destination_path)

print("Moved to:", destination_path)

rootdir = destination_path


stocksRecords = {}
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        stockName = os.path.join(subdir, file).split('/')[-1].split('.')[0]
        rows = read_csv(os.path.join(subdir, file))
        stocksRecords[stockName] = []
        for row in rows:
            stocksRecords[stockName].append(StockPriceRecord(stockName, row['Date'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Adj Close']))
