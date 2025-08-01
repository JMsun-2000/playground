#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:58:36 2025

@author: sunjim
"""

import pandas as pd

#import a_get_data

filedata = pd.read_csv("adsk_stock_prices.csv")
data = {
    'CLOSE': filedata['Close'],
    'HIGH': filedata['High'],
    'LOW': filedata['Low']
}

# 计算 CCI
def calculate_CCI(data, n):
    typ = (data['HIGH'] + data['LOW'] + data['CLOSE']) / 3
    typ_ma = typ.rolling(window=n).mean()
    mean_deviation = typ.rolling(window=n).apply(lambda x: (x - x.mean()).abs().mean())
    cci = (typ - typ_ma) / (0.015 * mean_deviation)
    return cci.round(2)


df = pd.DataFrame(data)

#n = 14

n = 26

# 计算CCI指标
df["CCI"] = calculate_CCI(df, n)
#print(df)

combined_df = pd.concat([filedata, df], axis=1)
print(combined_df)
combined_df.to_csv('adsk_stock_prices_with_cci.csv')