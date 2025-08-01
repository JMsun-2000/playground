#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:12:35 2025

@author: sunjim
"""

import pandas as pd

# Assume `data` is a DataFrame with columns: 'Date', 'Volume', 'Open', 'Close', 'High', 'Low'
filedata = pd.read_csv("adsk_stock_prices.csv")
data = {
    'CLOSE': filedata['Close'],
    'HIGH': filedata['High'],
    'LOW': filedata['Low']
}

df = pd.DataFrame(data)

N = 14
M = 6

# 计算 MTR
df['HL'] = df['HIGH'] - df['LOW']
df['HC'] = abs(df['HIGH'] - df['CLOSE'].shift(1))
df['LC'] = abs(df['LOW'] - df['CLOSE'].shift(1))
df['MAX1'] = df[['HL', 'HC']].max(axis=1)
df['MAX2'] = df[['MAX1', 'LC']].max(axis=1)
df['MTR'] = df['MAX2'].rolling(window=N).sum()

# 计算 HD
df['HD'] = df['HIGH'] - df['HIGH'].shift(1)

# 计算 LD
df['LD'] = df['LOW'].shift(1) - df['LOW']

# 计算 DMP
df['DMP'] = df.apply(lambda x: x['HD'] if (x['HD'] > 0 and x['HD'] > x['LD']) else 0, axis=1).rolling(window=N).sum()

# 计算 DMM
df['DMM'] = df.apply(lambda x: x['LD'] if (x['LD'] > 0 and x['LD'] > x['HD']) else 0, axis=1).rolling(window=N).sum()

# 计算 PDI、MDI 和 ADX
df['PDI'] = df['DMP'] * 100 / df['MTR']
df['MDI'] = df['DMM'] * 100 / df['MTR']
df['DX'] = (df['PDI'] - df['MDI']).abs() / (df['PDI'] + df['MDI']) * 100
df['ADX'] = df['DX'].rolling(window=M).mean()

# 计算 ADXR
df['ADXR'] = (df['ADX'] + df['ADX'].shift(M)) / 2

# 删除中间计算用的列
df.drop(['HL', 'HC', 'LC', 'MAX1', 'MAX2', 'HD', 'LD', 'DX'], axis=1, inplace=True)

# round to 2 decimal places
df = df.round(2)

combined_df = pd.concat([filedata, df], axis=1)
print(combined_df)
combined_df.to_csv('adsk_stock_prices_with_dmi.csv')