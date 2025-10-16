import os
import time
import random
import akshare as ak 
import pandas as pd
import warnings



# 创建文件存储路径
def create_path(ak_code="600519"):
    return 'stock_data/' + ak_code + '_bfq' + '.csv'

# 获取所有股票的历史数据
def do_load(ak_code, ak_name, period, start_date, end_date, adj=""):
#def do_load(ak_code, ak_name, period, start_date, end_date, adj="qfq"):
    print(f"Fetching data for {ak_code} ({ak_name})")
    try:
        df = ak.stock_zh_a_hist(symbol=ak_code, period=period, start_date=start_date, end_date=end_date, adjust=adj)
        if df.empty:
            print(f"No data available for {ak_code} in the specified date range")
            return
        print(f"Retrieved {len(df)} records for {ak_code}")
        path = create_path(ak_code)
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error fetching data for {ak_code}: {e}")

if __name__ == "__main__":
    # Create stock_data directory if it doesn't exist
    os.makedirs('stock_data', exist_ok=True)
    
    # Fetch data for a longer period
    do_load("600519", "贵州茅台", "daily", "20000101", "20251010")      
  
