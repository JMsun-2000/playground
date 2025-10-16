import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Read the CSV file
df = pd.read_csv('maotai/600519_bfq.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date to ensure we get the latest records
df = df.sort_values('Date')

# Get the latest 30 days
df_latest = df.tail(30).reset_index(drop=True)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for plotting
dates = df_latest['Date']
opens = df_latest['Open']
closes = df_latest['Close']
highs = df_latest['High']
lows = df_latest['Low']

# Plot candlesticks
for i in range(len(df_latest)):
    date = i
    open_price = opens.iloc[i]
    close_price = closes.iloc[i]
    high_price = highs.iloc[i]
    low_price = lows.iloc[i]
    
    # Determine color (green if close > open, red if close < open)
    if close_price < open_price:
        color = 'green'
        body_color = 'lightgreen'
    else:
        color = 'red'
        body_color = 'lightcoral'
    
    # Draw the high-low line (wick)
    ax.plot([date, date], [low_price, high_price], color='black', linewidth=1)
    
    # Draw the open-close box (body)
    body_height = abs(close_price - open_price)
    body_bottom = min(open_price, close_price)
    
    # Create a rectangle for the body
    rect = mpatches.Rectangle((date - 0.3, body_bottom), 0.6, body_height,
                               linewidth=1, edgecolor=color, 
                               facecolor=body_color if body_height > 0 else color)
    ax.add_patch(rect)

# Set x-axis labels
ax.set_xticks(range(len(df_latest)))
ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')

# Labels and title
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Price', fontsize=12, fontweight='bold')
ax.set_title('Latest 30 Days Candlestick Chart - Maotai Stock (600519)', 
             fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
green_patch = mpatches.Patch(color='lightgreen', label='Close >= Open (Bullish)')
red_patch = mpatches.Patch(color='lightcoral', label='Close < Open (Bearish)')
ax.legend(handles=[green_patch, red_patch], loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the plot
plt.show()

# Print summary statistics
print("\n=== Latest 30 Days Summary ===")
print(f"Date Range: {dates.iloc[0].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}")
print(f"Highest Price: {highs.max():.2f} on {dates.iloc[highs.idxmax()].strftime('%Y-%m-%d')}")
print(f"Lowest Price: {lows.min():.2f} on {dates.iloc[lows.idxmin()].strftime('%Y-%m-%d')}")
print(f"Starting Price (Open): {opens.iloc[0]:.2f}")
print(f"Ending Price (Close): {closes.iloc[-1]:.2f}")
print(f"Change: {closes.iloc[-1] - opens.iloc[0]:.2f} ({((closes.iloc[-1] - opens.iloc[0]) / opens.iloc[0] * 100):.2f}%)")

