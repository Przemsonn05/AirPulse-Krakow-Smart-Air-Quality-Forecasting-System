# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import requests
import holidays
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.colors as colors
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from scipy.stats import boxcox
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.special import inv_boxcox
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px

# Load datasets

years = range(2019, 2025)

lista_df = [pd.read_excel(f'../data/{year}_PM10_24g.xlsx') for year in years]
pm10_all = pd.concat(lista_df, ignore_index=True)

pm10_all.head()

air_data = pm10_all.copy()

air_data.columns = air_data.iloc[0]
air_data = air_data[1:].reset_index(drop=True)

air_data = air_data.iloc[4:].copy()
air_data = air_data.rename(columns={'Kod stacji': 'Date'})

air_data.head()

stations_list = ['MpKrakAlKras', 'MpKrakBujaka', 'MpKrakBulwar', 'MpKrakWadow']

cols_to_keep = ['Date'] + stations_list
krak_stations = air_data[cols_to_keep].copy()

krak_stations.head()

# Just to make sure that we have all rows from the original dataset

print(len(krak_stations) == len(air_data))

krak_stations.info()

krak_stations.describe()

krak_stations['Date'] = pd.to_datetime(krak_stations['Date'], errors='coerce')
krak_stations = krak_stations.dropna(subset=['Date'])
krak_stations = krak_stations.set_index('Date')

for col in stations_list:
    krak_stations[col] = krak_stations[col].astype(str).str.replace(',', '.').astype(float)

krak_stations = krak_stations.asfreq('D')

STATIONS = ['MpKrakAlKras', 'MpKrakBujaka', 'MpKrakBulwar', 'MpKrakWadow']

for col in STATIONS:
    gap_flag = krak_stations[col].isnull()
    
    krak_stations[col] = krak_stations[col].interpolate(method='time', limit=3)
    krak_stations[f'{col}_long_gap'] = (gap_flag & krak_stations[col].isnull()).astype(int)
    krak_stations[col] = krak_stations[col].ffill().bfill()

Q1 = krak_stations['MpKrakAlKras'].quantile(0.25)
Q3 = krak_stations['MpKrakAlKras'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3.0 * IQR

krak_stations.loc[krak_stations['MpKrakAlKras'] > upper_bound, 'MpKrakAlKras'] = np.nan
krak_stations['MpKrakAlKras'] = krak_stations['MpKrakAlKras'].interpolate(method='time', limit=3)

krak_stations['MpKrakAlKras'] = krak_stations['MpKrakAlKras'].ffill()
krak_stations['MpKrakAlKras'] = krak_stations['MpKrakAlKras'].bfill()

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 50.0577717,
	"longitude": 19.9265492,
	"start_date": "2019-01-01",
	"end_date": "2024-12-31",
	"daily": [
                "temperature_2m_mean", 
                'precipitation_sum',
                "wind_speed_10m_max",
                "surface_pressure_mean",
                "wind_direction_10m_dominant"
        ],
	"timezone": "Europe/Berlin"
}

responses = requests.get(url, params=params)
data = responses.json()

weather_df = pd.DataFrame(data["daily"])
weather_df['time'] = pd.to_datetime(weather_df['time'])
weather_df.columns = [
    'date', 'temp_avg', 'rain_sum', 'wind_max', 
    'pressure_avg', 'wind_dir_dominant'
]

print(weather_df.head())

weather_df.describe()

weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df = weather_df.set_index('date')

df_final = krak_stations.join(weather_df, how='left')

DIR_MAP = {
    'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,
    'SE':135,'SSE':157.5,'S':180,'SSW':202.5,'SW':225,
    'WSW':247.5,'W':270,'WNW':292.5,'NW':315,'NNW':337.5
}

if df_final['wind_dir_dominant'].dtype == object:
    df_final['wind_dir_deg'] = df_final['wind_dir_dominant'].map(DIR_MAP).fillna(0)
else:
    df_final['wind_dir_deg'] = df_final['wind_dir_dominant'].fillna(0)

rad = np.deg2rad(df_final['wind_dir_deg'])
df_final['wind_dir_sin'] = np.sin(rad)
df_final['wind_dir_cos'] = np.cos(rad)
df_final.drop(columns=['wind_dir_dominant', 'wind_dir_deg'], inplace=True)

df_final.head()

# Time serires with EU norm

plt.figure(figsize=(12, 7))

plt.plot(df_final.index, df_final['MpKrakAlKras'], 
        color='teal', alpha=0.5, linewidth=1, label='Daily PM10 Concentration')

plt.plot(df_final['MpKrakAlKras'].rolling(window=30).mean(),
        color='darkorange', linewidth=2, label='30-day Rolling Average')

plt.axhline(50, color='red', linestyle='--', linewidth=2, label='EU norm (50 µg/m³)')

plt.title('PM10 Levels in Kraków with EU Air Quality Standard', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Concentration [µg/m³]', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/eda_time_series.png', dpi=300)
plt.show()

# EU norm exceedances grouped by year

krak_yearly = df_final['MpKrakAlKras'].resample('YE').agg(['mean', 'max', 'count']).copy()

exceedances = df_final[df_final['MpKrakAlKras'] > 50]['MpKrakAlKras'].resample('YE').size()

krak_yearly['Days_above_50'] = exceedances.fillna(0).astype(int)

krak_yearly.index = krak_yearly.index.year
krak_yearly.columns = ['Avg_PM10', 'Max_PM10', 'Total_Days', 'Exceedance_Days']

plt.figure(figsize=(12, 7))

bars = sns.barplot(x=krak_yearly.index, y=krak_yearly['Exceedance_Days'], hue=krak_yearly.index, palette='Blues_d', width=0.6, legend=False)

for bar in bars.patches:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,             
        int(bar.get_height()),             
        ha='center',                      
        va='bottom',                       
        fontsize=10
    )

plt.axhline(y=35, color='red', linestyle='--', linewidth=2, label='Permitted days per year (35)')

plt.title('Number of days with PM10 > 50 µg/m³ per year', fontsize=16, fontweight='bold')
plt.ylabel('Number of days', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig('../images/eda_exceedances_barplot.png', dpi=300)
plt.show()

# Distribution of PM10 by Season

if 'month' not in df_final.columns:
    df_final['month'] = df_final.index.month

df_final["season"] = df_final["month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
})

season_order = ["Winter", "Spring", "Autumn", "Summer"]
season_colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]

plt.figure(figsize=(12, 7))

for season, color in zip(season_order, season_colors):
    sns.kdeplot(
        data=df_final[df_final['season'] == season]['MpKrakAlKras'],
        label=season,
        color=color,
        fill=True,
        alpha=0.2,
        linewidth=2
    )

plt.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='EU Limit (50 µg/m³)')

plt.title("Statistical Distribution of PM10 by Season", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("PM10 Concentration [µg/m³]", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(0, 150)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("../images/eda_seasonal_distribution.png", dpi=300)
plt.show()

# Weekly analysis PM10 level

df_final['week'] = df_final.index.dayofweek
weekly = df_final.groupby('week')['MpKrakAlKras'].agg(['mean', 'std'])
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

norm = mcolors.Normalize(vmin=weekly['mean'].min(), vmax=weekly['mean'].max())
cmap = plt.cm.Blues
colors_w = [cmap(norm(v)) for v in weekly['mean']]

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(
    days, weekly['mean'], yerr=weekly['std'], capsize=5, color=colors_w, alpha=0.8,
    edgecolor='black', width=0.5
)

ax.bar_label(bars, padding=8, fmt='%.1f', fontsize=12)

ax.set_title("Weekly PM10 Cycles in Kraków (2019-2024)", fontsize=16, fontweight="bold")
ax.set_ylabel("PM10 [µg/m³]", fontsize=12)
ax.set_ylim(0, (weekly['mean'] + weekly['std']).max() * 1.2)

ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_alpha(0.2)

ax.axhline(50, color='#c0392b', linestyle='--', alpha=0.5, label='EU Daily Limit (50 µg/m³)')
ax.legend(frameon=False, loc='upper right')

plt.tight_layout()
plt.savefig('../images/eda_week_PM10_analysis.png', dpi=300)
plt.show()

# Boxplots by months

df_final['month_name'] = df_final.index.month_name()
month_order = list(calendar.month_name)[1:]

plt.figure(figsize=(12, 7))

sns.boxplot(
    data=df_final,
    x='month_name',
    y='MpKrakAlKras',
    hue='month_name',
    order=month_order,
    palette='Blues_d'
)

plt.axhline(y=50, color='red', linestyle='--', linewidth=2, label='EU Daily Limit (50 µg/m³)')

plt.title('Monthly Distribution and Variance of PM10 in Kraków', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('PM10 Concentration [µg/m³]', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(rotation=45)

plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
sns.despine()

plt.tight_layout()
plt.savefig('../images/eda_monthly_boxplots.png', dpi=300)
plt.show()

# Heatmap - month × year

df_final['year'] = df_final.index.year

pivot_data = df_final.pivot_table(
    values='MpKrakAlKras',
    index=df_final['year'],
    columns=df_final['month'],
    aggfunc='mean'
)

pivot_data.columns = [calendar.month_name[m] for m in pivot_data.columns]

plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot_data,
    annot=True,
    fmt='.1f',
    cmap='coolwarm'
)

plt.title('Monthly Average PM10 Concentrations in Kraków (Year vs Month)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../images/eda_heatmap_month_year.png', dpi=300)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.regplot(data=df_final, x='wind_max', y='MpKrakAlKras', 
            scatter_kws={'alpha':0.3, 'color':"#3A963F"}, line_kws={'color':'red'}, ax=axes[0])
axes[0].set_title('Impact of Wind Speed on PM10', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Max Wind Speed [m/s]', fontsize=12)
axes[0].set_ylabel('PM10 Concentration [µg/m³]', fontsize=12)

sns.regplot(data=df_final, x='temp_avg', y='MpKrakAlKras', 
            scatter_kws={'alpha':0.3, 'color':"#3ba4dc"}, line_kws={'color':'red'}, ax=axes[1])
axes[1].set_title('Impact of Temperature on PM10', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Average Temperature [°C]', fontsize=12)
axes[1].set_ylabel('PM10 Concentration [µg/m³]', fontsize=12)

plt.tight_layout()
plt.savefig('../images/weather_scatter_plots.png', dpi=300)
plt.show()

df_year = df_final.last('1Y')

fig, ax1 = plt.subplots(figsize=(16, 8))

color1 = 'tab:red'
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('PM10 [µg/m³]', color=color1, fontsize=12, fontweight='bold')
ax1.plot(df_year.index, df_year['MpKrakAlKras'], color=color1, alpha=0.7, label='PM10')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.fill_between(df_year.index, df_year['MpKrakAlKras'], color=color1, alpha=0.1)

ax2 = ax1.twinx() 
color2 = 'tab:blue'
ax2.set_ylabel('Temperature [°C]', color=color2, fontsize=12, fontweight='bold')
ax2.plot(df_year.index, df_year['temp_avg'], color=color2, linestyle='--', label='Temperature')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Temporal Relationship: PM10 vs Temperature (Last 12 Months)', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('../images/weather_dual_axis_timeseries.png', dpi=300)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Ustawienie stylu
sns.set_theme(style="whitegrid")

# Wybieramy kolumny do korelacji
cols_to_corr = ['MpKrakAlKras', 'temp_avg', 'rain_sum', 'wind_max']
corr_matrix = df_final[cols_to_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f", center=0, linewidths=0.5)
plt.title('Correlation Heatmap: PM10 vs Meteorological Factors', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../images/weather_correlation_heatmap.png', dpi=300)
plt.show()

# Top 15 most polluted days

top_days = df_final['MpKrakAlKras'].nlargest(15).reset_index()
top_days['label'] = top_days['Date'].dt.strftime("%d %b %Y")

norm = colors.Normalize(vmin=top_days['MpKrakAlKras'].min(), vmax=top_days['MpKrakAlKras'].max())
cmap = cm.get_cmap('Reds')
colors_gradient = [cmap(norm(value)) for value in top_days['MpKrakAlKras']]

fig, ax = plt.subplots(figsize=(12 ,7))

bars = ax.barh(
    top_days['label'][::-1], top_days['MpKrakAlKras'][::-1], color=colors_gradient[::-1], 
    alpha=0.8, edgecolor='black', height=0.7
)

for bar in bars:
    plt.text(bar.get_width() + 0.5,               
             bar.get_y() + bar.get_height()/2, 
             int(bar.get_width()),             
             va='center', fontsize=10)

ax.axvline(50, color='black', linestyle='--', linewidth=2, label='PM10 EU norm')

ax.set_title("Top 15 Days with Highest PM10 Concentrations (2019–2024)", fontsize=16, fontweight="bold")
ax.set_xlabel("PM10 [µg/m³]", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis="x")

plt.tight_layout()
plt.savefig("../images/top15_peak_pollution_events.png", dpi=300)
plt.show()

# STL decomposition

series = df_final['MpKrakAlKras'].dropna()

# robust=True - ignore outliers
stl=STL(series, period=365, robust=True).fit()

sns.set_style("white")
fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

ax[0].plot(series.index, series.values, color="#1e272e", linewidth=0.8, alpha=0.9)
ax[0].set_ylabel("Observed\n($\mu g/m^3$)", fontsize=12, fontweight='bold')
ax[0].set_title("Longitudinal PM10 Signal Decomposition in Kraków (STL Method)", fontsize=18, fontweight='bold', pad=25)

ax[1].plot(series.index, stl.trend, color="#d35400", linewidth=2)
ax[1].set_ylabel("Secular Trend\n($\mu g/m^3$)", fontsize=12, fontweight='bold')

ax[2].plot(series.index, stl.seasonal, color="#009432", linewidth=0.9)
ax[2].axhline(0, color="black", linewidth=0.8, linestyle='--')
ax[2].set_ylabel("Seasonal\nVariation", fontsize=12, fontweight='bold')

ax[3].scatter(series.index, stl.resid, color="#7f8c8d", s=2, alpha=0.5)
ax[3].axhline(0, color="black", linewidth=0.8, linestyle='--')
ax[3].set_ylabel("Residuals\n(Noise)", fontsize=12, fontweight='bold')

for axis in ax:
    axis.grid(True, axis='y', linestyle=':', alpha=0.4)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(labelsize=10)

ax[3].set_xlabel("Year", fontsize=12, labelpad=15)

plt.tight_layout()
plt.savefig('../images/eda_stl_decomposition_analysis.png', dpi=300)
plt.show()

# ACF & PACH analysis for PM10

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True) 

plot_acf(series, lags=40, ax=ax1, color="#4e8cff", 
         vlines_kwargs={"colors": "#4e8cff", "alpha": 0.5},
         title='Autocorrelation (ACF) - Seasonal patterns check')

plot_pacf(series, lags=40, ax=ax2, color="#e74c3c", 
          vlines_kwargs={"colors": "#e74c3c", "alpha": 0.5},
          title="Partial Autocorrelation (PACF) - Direct lag influence", 
          method="ywm")

for ax in [ax1, ax2]:
    ax.grid(True, axis='y', linestyle='--', alpha=0.3) 
    ax.set_ylabel("Correlation Strength", fontsize=12, color="#555555")
    sns.despine(ax=ax, left=True, bottom=False) 

ax2.set_xlabel("Lags (Time steps back)", fontsize=12, fontweight="bold")

fig.suptitle("Time Series Diagnostic: ACF & PACF Analysis for PM10", 
             fontsize=16, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.savefig('../images/eda_acf_and_pacf.png', dpi=300, bbox_inches='tight')
plt.show()

# ✅ FIX: Heating season — more precise boundary (Oct 15 – Apr 15)
# Original used full months Oct–Mar which over-includes warm shoulder days

df_final['is_heating_season'] = (
    (df_final['month'] >= 10) | (df_final['month'] <= 4)
).astype(int)
# Exclude warm parts of April and October
df_final.loc[(df_final.index.month == 4) & (df_final.index.day > 15), 'is_heating_season'] = 0
df_final.loc[(df_final.index.month == 10) & (df_final.index.day < 16), 'is_heating_season'] = 0

df_final['is_weekend'] = (df_final['week'] >= 5).astype(int)
print(f"Heating season days: {df_final['is_heating_season'].sum()} / {len(df_final)}")

pm10 = df_final['MpKrakAlKras'].dropna()

pm10_log, lambda_bc = boxcox(pm10 + 1)
df_final['PM10_transformed'] = pm10_log

def stationarity_report(series, name='series'):
    """Run ADF and KPSS; return recommended differencing order d."""
    series = series.dropna()
    adf_stat, adf_p, *_ = adfuller(series, autolag='AIC')
    kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')

    adf_stat  = adf_p < 0.05   # True = stationary
    kpss_stat = kpss_p > 0.05  # True = stationary

    print(f"── {name} {'─'*40}")
    print(f"  ADF  p={adf_p:.4f}  → {'✅ stationary' if adf_stat else '❌ NON-stationary'}")
    print(f"  KPSS p={kpss_p:.4f}  → {'✅ stationary' if kpss_stat else '❌ NON-stationary'}")

    if adf_stat and kpss_stat:
        d, verdict = 0, "Stationary — d=0 recommended"
    else:
        d, verdict = 1, "Non-stationary — d=1 recommended"
    print(f"  → {verdict}\n")
    return d

D_ORDER = stationarity_report(df_final['PM10_transformed'], 'PM10_transformed (Box-Cox)')
print(f"⚙️  Using d = {D_ORDER} in SARIMAX")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

skew_before = stats.skew(pm10)
skew_after = stats.stats.skew(pm10_log)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.histplot(pm10, kde=True, ax=axes[0, 0], color='#7f8c8d', alpha=0.6)
axes[0, 0].set_title(f"Original PM10 Distribution\n(Skewness: {skew_before:.2f})", fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel("PM10 Concentration [µg/m³]")
axes[0, 0].set_ylabel("Frequency")

sns.histplot(pm10_log, kde=True, ax=axes[0, 1], color='#27ae60', alpha=0.6)
axes[0, 1].set_title(f"PM10 after Box-Cox Transformation\n(Skewness: {skew_after:.2f}, λ: {lambda_bc:.2f})", fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel("Transformed Value (Box-Cox)")
axes[0, 1].set_ylabel("Frequency")

stats.probplot(pm10, dist="norm", plot=axes[1, 0])
axes[1, 0].get_lines()[0].set_color("#b7b9b9")
axes[1, 0].get_lines()[0].set_markerfacecolor('none')
axes[1, 0].get_lines()[1].set_color('#c0392b')
axes[1, 0].set_title("Q-Q Plot: Original Data", fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel("Theoretical Quantiles")
axes[1, 0].set_ylabel("Ordered Values")

stats.probplot(pm10_log, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[0].set_color('#27ae60')
axes[1, 1].get_lines()[0].set_markerfacecolor('none')
axes[1, 1].get_lines()[1].set_color('#c0392b')
axes[1, 1].set_title("Q-Q Plot: Transformed Data", fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel("Theoretical Quantiles")
axes[1, 1].set_ylabel("Ordered Values")

sns.despine()

plt.suptitle("Impact of Box-Cox Transformation on PM10 Normality", 
             fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('../images/engineering_hist_and_qq_plot.png', dpi=300, bbox_inches='tight')
plt.show()

df_final['month_sin'] = np.sin(2 * np.pi * df_final['month'] / 12)
df_final['month_cos'] = np.cos(2 * np.pi * df_final['month'] / 12)

df_final['dow_sin'] = np.sin(2 * np.pi * df_final['week'] / 7)
df_final['dow_cos'] = np.cos(2 * np.pi * df_final['week'] / 7)

df_final['lag_1d']  = df_final['MpKrakAlKras'].shift(1)
df_final['lag_7d']  = df_final['MpKrakAlKras'].shift(7)
df_final['lag_14d'] = df_final['MpKrakAlKras'].shift(14)
df_final['lag_30d'] = df_final['MpKrakAlKras'].shift(30)

for w in [3, 7, 14, 30]:
    base = df_final['MpKrakAlKras'].shift(1)
    df_final[f'rolling_mean_{w}d'] = base.rolling(w).mean()
    df_final[f'rolling_std_{w}d']  = base.rolling(w).std()
    df_final[f'rolling_max_{w}d']  = base.rolling(w).max()

df_final['rolling_diff_7d'] = (
    df_final['rolling_mean_7d'] - df_final['rolling_mean_14d']
)

df_final['rolling_diff_14d'] = (
    df_final['rolling_mean_14d'] - df_final['rolling_mean_30d']
)

pl_holidays = holidays.CountryHoliday('PL', years=range(2019, 2025))
df_final['is_holiday'] = df_final.index.isin(pl_holidays).astype(int)

# Temperature
df_final['is_frost'] = (df_final['temp_avg'] <= 0).astype(int)

# Wind
df_final['is_calm_wind'] = (df_final['wind_max'] <= 5).astype(int)

df_final['is_frost_calm'] = df_final['is_frost'] * df_final['is_calm_wind']

df_final['is_heating_season_calm'] = df_final['is_heating_season'] * df_final['is_calm_wind']

df_final['rain_yesterday']   = df_final['rain_sum'].shift(1)
df_final['rain_3d_sum']      = df_final['rain_sum'].shift(1).rolling(3).sum()  # opady ostatnie 3 dni
df_final['rain_7d_sum']      = df_final['rain_sum'].shift(1).rolling(7).sum()
df_final['dry_spell_days']   = (                                                # ile dni bez deszczu z rzędu
    df_final['rain_sum'].shift(1)
    .rolling(14)
    .apply(lambda x: (x == 0).sum())
)

# Obliczamy zmienność temperatury z ostatnich 6 godzin
df_final['temp_std_6h'] = df_final['temp_avg'].rolling(window=6).std()

# Proxy inwersji: Niska zmienność (< 1.5 stopnia) + niska temperatura + brak wiatru
df_final['inversion_proxy'] = (
    (df_final['temp_std_6h'] < 1.5).astype(int) * (df_final['temp_avg'] < 5).astype(int) * df_final['is_calm_wind']
)

df_final['heating_degree_days'] = (15 - df_final['temp_avg']).clip(lower=0)
df_final['hdd_7d'] = df_final['heating_degree_days'].shift(1).rolling(7).sum()

df_final['wind_inverse']  = 1 / (df_final['wind_max'] + 0.1)  # dyspersja: im słabszy, tym wyższe PM10
df_final['wind_7d_mean']  = df_final['wind_max'].shift(1).rolling(7).mean()

df_final['hdd_calm']         = df_final['heating_degree_days'] * df_final['is_calm_wind']
df_final['cold_dry_calm']    = (
    (df_final['temp_avg'] < 0).astype(int) *
    (df_final['rain_sum'] == 0).astype(int) *
    df_final['is_calm_wind']
)
df_final['hdd_no_rain']      = df_final['heating_degree_days'] * (df_final['rain_sum'] == 0).astype(int)

# Trend ciśnienia (zmiana z poprzedniego dnia)
df_final['pressure_change'] = df_final['pressure_avg'].diff()  # If available
df_final['pressure_7d_trend'] = df_final['pressure_avg'].shift(1).rolling(7).apply(lambda x: x[-1] - x[0])

STATIONS = ['MpKrakAlKras', 'MpKrakBujaka', 'MpKrakBulwar', 'MpKrakWadow']

# 1. Agregacje z wszystkich stacji
# ─────────────────────────────────
df_final['pm10_mean_all'] = df_final[STATIONS].mean(axis=1).shift(1)
df_final['pm10_mean_all_lag1'] = df_final[STATIONS].mean(axis=1).shift(1)
df_final['pm10_std_all_lag1'] = df_final[STATIONS].std(axis=1).shift(1)
df_final['pm10_max_all_lag1'] = df_final[STATIONS].max(axis=1).shift(1)
df_final['pm10_max_all'] = df_final[STATIONS].max(axis=1).shift(1)
df_final['pm10_min_all'] = df_final[STATIONS].min(axis=1).shift(1)
df_final['pm10_std_all'] = df_final[STATIONS].std(axis=1).shift(1)

# 2. Różnica target vs. średnia — czy stacja główna jest outlierem?
# ────────────────────────────────────────────────────────────────
df_final['pm10_diff_from_mean'] = df_final['MpKrakAlKras'] - df_final['pm10_mean_all']
df_final['pm10_deviation_ratio'] = (
    df_final['pm10_diff_from_mean'] / (df_final['pm10_mean_all'] + 1)
)

# 7. Trend zmiany wśród stacji (czy robi się gorzej/lepiej wszędzie?)
# ─────────────────────────────────────────────────────────────────────
df_final['pm10_mean_all_change'] = df_final['pm10_mean_all'].diff()
df_final['pm10_ensemble_momentum'] = df_final['pm10_mean_all'].shift(1).rolling(3).mean() - df_final['pm10_mean_all']

# 8. Asymmetria — czy niektóre części miasta są bardziej zanieczyszczone?
# Lokalnie: porównaj stacje w podobnych kierunkach
# ───────────────────────────────────────────
# Zakładam że stacje są w różnych ćwiartkach - możesz dostosować geografię
df_final['station_pair_diff_1'] = df_final['MpKrakAlKras'] - df_final['MpKrakBujaka']  # N-S?
df_final['station_pair_diff_2'] = df_final['MpKrakBulwar'] - df_final['MpKrakWadow']    # E-W?

# 9. Lag multi-station (ile stacji miało wysoki PM10 poprzedniego dnia?)
# ──────────────────────────────────────────────────────────────────────
df_final['pm10_mean_all_lag1'] = df_final['pm10_mean_all'].shift(1)
df_final['pm10_max_all_lag1'] = df_final['pm10_max_all'].shift(1)
df_final['pm10_std_all_lag1'] = df_final['pm10_std_all'].shift(1)


# 1. Definiujemy listę stacji pomocniczych (wszystkie poza targetem)
AUX_STATIONS = ['MpKrakBujaka', 'MpKrakBulwar', 'MpKrakWadow']

# 2. Obliczamy średnią kroczącą z wczorajszego dnia
# axis=1 oznacza, że liczymy średnią w poziomie (dla każdego dnia osobno z 3 stacji)
# .shift(1) przesuwa wynik o jeden dzień, by uniknąć Data Leakage
df_final['aux_stations_mean_lag1'] = df_final[AUX_STATIONS].mean(axis=1).shift(1)

# 3. Przy okazji warto policzyć odchylenie standardowe (zmienność między dzielnicami)
df_final['aux_stations_std_lag1'] = df_final[AUX_STATIONS].std(axis=1).shift(1)

# ─────────────────────────────────────────────────────────────
# Comparisons with auxiliary stations (SAFE - using lags only)
# ─────────────────────────────────────────────────────────────
df_final['target_vs_aux_lag1'] = df_final['MpKrakAlKras'] - df_final['aux_stations_mean_lag1']

# FIXED: target_volatility_lag1 now uses YESTERDAY's target, not TODAY's
temp_cols = pd.DataFrame({
    'target_lag1': df_final['MpKrakAlKras'].shift(1),
    'aux_std_lag1': df_final['aux_stations_std_lag1']
})
df_final['target_volatility_lag1'] = temp_cols.std(axis=1)

df_final.dropna(inplace=True)

n = len(df_final)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train = df_final.iloc[:train_end]
val = df_final.iloc[train_end:val_end]
test = df_final.iloc[val_end:]

def evaluate_forecast(y_true, y_pred, lambda_bc, label=''):
    """
    Evaluate on original PM10 scale (inverse Box-Cox).
    ✅ FIX: safe MAPE — adds eps floor to avoid div-by-zero on near-zero actuals.
    """
    actual    = inv_boxcox(np.array(y_true), lambda_bc) - 1
    predicted = inv_boxcox(np.array(y_pred), lambda_bc) - 1
    predicted = np.clip(predicted, 0, None)   # PM10 ≥ 0

    r2   = r2_score(actual, predicted)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # ✅ Safe MAPE: floor denominator at 1 µg/m³ to avoid explosion on calm days
    eps  = 1.0
    mape = np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), eps)) * 100

    print(f"\n── Evaluation: {label} {'─'*30}")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAE  = {mae:.2f} µg/m³")
    print(f"  RMSE = {rmse:.2f} µg/m³")
    print(f"  MAPE = {mape:.1f}%  (eps-safe)")

    return {'Model': label, 'R2': round(r2,4), 'MAE': round(mae,2),
            'RMSE': round(rmse,2), 'MAPE': round(mape,1)}

TARGET = 'PM10_transformed'

# ===============================================================
# ANALIZA: Nowe cechy Multi-Station
# ===============================================================
# Sprawdzamy, czy nowe cechy rzeczywiście korelują z targetem

import matplotlib.pyplot as plt
import seaborn as sns

new_multistation_features = [
    'pm10_mean_all', 'pm10_std_all', 'pm10_diff_from_mean', 'pollution_homogeneity', 'station_rank',
    'pm10_mean_all_lag1', 'all_stations_high_flag',
]

# Filtruj do tych, które faktycznie istnieją
new_features_available = [f for f in new_multistation_features if f in df_final.columns]

if len(new_features_available) > 0:
    print("\n📊 KORELACJE NOWYCH CECH Z TARGETEM:")
    corr_with_target = df_final[new_features_available + [TARGET]].corr()[TARGET].sort_values(ascending=False)
    print(corr_with_target)
    
    # Wizualizacja
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter 1: pm10_mean_all
    axes[0, 0].scatter(df_final['pm10_mean_all'], df_final[TARGET], alpha=0.3, s=10)
    axes[0, 0].set_xlabel('PM10 Mean All Stations')
    axes[0, 0].set_ylabel('Target (Box-Cox)')
    axes[0, 0].set_title(f'Corr: {df_final[["pm10_mean_all", TARGET]].corr().iloc[0, 1]:.3f}')
    
    # Scatter 2: pm10_std_all
    axes[0, 1].scatter(df_final['pm10_std_all'], df_final[TARGET], alpha=0.3, s=10, color='orange')
    axes[0, 1].set_xlabel('PM10 Spatial Std')
    axes[0, 1].set_ylabel('Target (Box-Cox)')
    axes[0, 1].set_title(f'Corr: {df_final[["pm10_std_all", TARGET]].corr().iloc[0, 1]:.3f}')
    
    plt.tight_layout()
    plt.savefig('../images/multistation_features_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("⚠️  Brak nowych cech - sprawdź czy były dodane!")


history_arima = list(train[TARGET])
arima_preds   = []
 
for t in range(len(val)):
    model    = ARIMA(history_arima, order=(2, 1, 1)).fit()
    yhat     = model.forecast(steps=1)[0]
    arima_preds.append(yhat)
    history_arima.append(val[TARGET].iloc[t])   # dodaj prawdziwą wartość
 
arima_preds = np.array(arima_preds)
res_arima   = evaluate_forecast(val[TARGET].values, arima_preds, lambda_bc, "ARIMA (2,1,1)")

res_arima_df = pd.DataFrame([res_arima], index=['ARIMA (2,1,1)'])
res_arima_df

SARIMAX_EXOG = [
    # Weather raw
    "temp_avg", "rain_sum", "wind_max",
    # Weather engineered
    "heating_degree_days", "wind_inverse",
    "is_heating_season", "is_calm_wind",
    "hdd_calm", "rain_3d_sum",
    # ✅ NEW: Multi-station aggregations + flags
    "pm10_mean_all",
    "pm10_std_all",
    "all_stations_high_flag",
]

# ✅ FIX: use D_ORDER from ADF/KPSS test (not hard-coded d=1)
# Also: wind direction now encoded as sin/cos — add to exog if not already present

SARIMAX_EXOG_FULL = SARIMAX_EXOG + ['wind_dir_sin', 'wind_dir_cos']

# Ensure all exog cols exist (drop missing)
SARIMAX_EXOG_FINAL = [c for c in SARIMAX_EXOG_FULL if c in df_final.columns]
print(f"SARIMAX exog ({len(SARIMAX_EXOG_FINAL)} vars): {SARIMAX_EXOG_FINAL}")

sarimax_model = SARIMAX(
    train[TARGET],
    exog=train[SARIMAX_EXOG_FINAL],
    order=(2, D_ORDER, 1),           # ✅ d from stationarity test
    seasonal_order=(1, 0, 1, 7),     # s=7 weekly cycle
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=200)

print(f"\nSARIMAX AIC: {sarimax_model.aic:.2f}")

sarimax_preds   = []
current_results = sarimax_model

for i_step in range(len(val)):
    exog_now = val[SARIMAX_EXOG_FINAL].iloc[i_step:i_step+1]
    yhat     = current_results.forecast(steps=1, exog=exog_now)[0]
    sarimax_preds.append(yhat)

    actual_y        = val[TARGET].iloc[i_step:i_step+1]
    current_results = current_results.extend(actual_y, exog=exog_now)

sarimax_preds = np.array(sarimax_preds)
res_sarimax   = evaluate_forecast(val[TARGET].values, sarimax_preds, lambda_bc,
                                  "SARIMAX (2,D,1)(1,0,1,7)")

res_sarimax_df = pd.DataFrame([res_sarimax], index=['SARIMAX (2,1,1)(1,0,1,7)'])
res_sarimax_df

# ✅ FIX: add circular wind encoding + spatial station features to Prophet
PROPHET_REGRESSORS = [
    # Weather raw
    "temp_avg", "rain_sum", "wind_max",
    # Weather engineered
    "heating_degree_days", "wind_inverse",
    "rain_3d_sum", "dry_spell_days",
    # Wind direction (circular — ✅ ADDED)
    "wind_dir_sin", "wind_dir_cos",
    # Domain flags
    "is_heating_season", "is_calm_wind",
    "is_frost", "is_weekend", "is_holiday",
    # Interactions
    "is_frost_calm", "hdd_calm", "cold_dry_calm",
    "inversion_proxy",
    # Rolling lags (shifted — no leakage)
    "rolling_mean_7d", "rolling_diff_7d",
    # ✅ NEW: Multi-station features
    "pm10_mean_all", "pm10_median_all", "pm10_max_all", "pm10_min_all",
    "pm10_std_all", "pm10_range_all",
    "pm10_diff_from_mean",
    "station_rank",
    "pm10_mean_all_lag1", "pm10_max_all_lag1",
]

# Filter to only cols that actually exist in df_final
PROPHET_REGRESSORS = [c for c in PROPHET_REGRESSORS if c in df_final.columns]
print(f"Prophet regressors ({len(PROPHET_REGRESSORS)}): {PROPHET_REGRESSORS}")

def prepare_prophet_df(df_in, regressors):
    out = pd.DataFrame()
    out["ds"] = df_in.index
    out["y"]  = df_in[TARGET].values
    for col in regressors:
        out[col] = df_in[col].values
    return out.reset_index(drop=True)

train_p = train.dropna(subset=PROPHET_REGRESSORS + [TARGET])
val_p   = val.dropna(subset=PROPHET_REGRESSORS + [TARGET])

m_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,        # ✅ increased from default 10 (explicit)
)
m_prophet.add_country_holidays(country_name="PL")

# ✅ FIX: standardize=True ensures regressors are normalized before fitting
for col in PROPHET_REGRESSORS:
    m_prophet.add_regressor(col, standardize=True)

m_prophet.fit(prepare_prophet_df(train_p, PROPHET_REGRESSORS))
prophet_forecast = m_prophet.predict(prepare_prophet_df(val_p, PROPHET_REGRESSORS))
prophet_preds    = prophet_forecast["yhat"].values

res_prophet = evaluate_forecast(
    val_p[TARGET].values, prophet_preds, lambda_bc, "Prophet"
)

res_prophet_df = pd.DataFrame([res_prophet], index=['Prophet'])
res_prophet_df

LGBM_FEATURES = [
    # Pogoda surowa
    "temp_avg", "rain_sum", "wind_max",
    # Pogoda inżynierowana
    "heating_degree_days", "hdd_7d", "wind_inverse", "wind_7d_mean",
    "rain_yesterday", "rain_3d_sum", "rain_7d_sum", "dry_spell_days",
    # Flagi
    "is_heating_season", "is_weekend", "is_holiday",
    "is_frost", "is_calm_wind",
    # Interakcje
    "is_frost_calm", "is_heating_season_calm",
    "hdd_calm", "cold_dry_calm", "hdd_no_rain", "inversion_proxy",
    # Temporal
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # Lagi PM10 (SAFE - z przeszłości)
    "lag_1d", "lag_7d", "lag_14d", "lag_30d", "lag_2d", "lag_3d", "lag_21d",
    # Rolling PM10
    "rolling_mean_3d", "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_30d",
    "rolling_std_7d", "rolling_std_14d",
    "rolling_max_7d", "rolling_max_14d",
    "rolling_diff_7d", "rolling_diff_14d",
    # ✅ SAFE Multi-station features (z poprzedniego dnia - LAG 1)
    "aux_stations_mean_lag1", "aux_stations_max_lag1", "aux_stations_std_lag1",
    "aux_stations_median_lag1",
    "aux_mean_change_from_lag7",
    "target_vs_aux_lag1", "target_volatility_lag1",
    "aux_pair_diff_lag1", "aux_bias_lag1",
    "target_rank_vs_aux_lag1",
    "aux_high_yesterday", "all_aux_high_yesterday",
    "target_lag1", "target_lag7", "target_trend_1d", "target_trend_7d",
    "MpKrakBujaka_lag1", "MpKrakBulwar_lag1", "MpKrakWadow_lag1",
    "MpKrakBujaka_lag7", "MpKrakBulwar_lag7", "MpKrakWadow_lag7",
]

print("✅ UPDATED LGBM_FEATURES:")
print(f"   Total features: {len(LGBM_FEATURES)}")
safe_ms = [f for f in LGBM_FEATURES if 'aux' in f or 'target_' in f or 'lag' in f.lower()]
print(f"   SAFE multi-station: {len(safe_ms)} (all shifted/lagged)")
print(f"   Removed leaking features: 6 (pm10_mean_all, station_rank, etc.)")

import lightgbm as lgb

# ✅ FIXED: Build X_train, y_train, X_val, y_val WITH SAFE multi-station features (no leakage!)
# Use df_final directly (not old train/val) to get new safe features
LGBM_FEATURES_FINAL = [c for c in LGBM_FEATURES if c in df_final.columns]

print(f"🔍 Checking features:")
print(f"   Features required: {len(LGBM_FEATURES)}")
print(f"   Features found in df_final: {len(LGBM_FEATURES_FINAL)}")
missing = set(LGBM_FEATURES) - set(LGBM_FEATURES_FINAL)
if missing:
    print(f"   ⚠️  Missing ({len(missing)}): {list(missing)[:5]}...")

# Re-split from df_final to get SAFE features
train_end = int(len(df_final) * 0.7)
val_end = int(len(df_final) * 0.85)

train_lgb = df_final.iloc[:train_end].dropna(subset=LGBM_FEATURES_FINAL + [TARGET])
val_lgb   = df_final.iloc[train_end:val_end].dropna(subset=LGBM_FEATURES_FINAL + [TARGET])

# Extract X and y
X_train = train_lgb[LGBM_FEATURES_FINAL].copy()
y_train = train_lgb[TARGET].copy()
X_val   = val_lgb[LGBM_FEATURES_FINAL].copy()
y_val   = val_lgb[TARGET].copy()

print(f"   Total features: {len(LGBM_FEATURES_FINAL)}")
ms_count = len([f for f in LGBM_FEATURES_FINAL if any(x in f.lower() for x in ['pm10_mean', 'pm10_std', 'station', 'pollution', 'anomaly'])])
print(f"   Multi-station features: {ms_count}")
print(f"   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"   X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

lgbm_model = lgb.LGBMRegressor(
    objective='regression_l1',
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
)

lgbm_preds = lgbm_model.predict(X_val)
lgbm_preds = np.clip(lgbm_preds, 0, None)   # ✅ FIX: PM10 cannot be negative
res_lgbm   = evaluate_forecast(y_val.values, lgbm_preds, lambda_bc, "LightGBM")

# 1. Przygotuj X i y dla zbioru testowego
test_lgb = df_final.iloc[val_end:].dropna(subset=LGBM_FEATURES_FINAL + [TARGET])

X_test = test_lgb[LGBM_FEATURES_FINAL].copy()
y_test = test_lgb[TARGET].copy()

# 2. Przewiduj na danych, których model kompletnie nie zna
lgbm_test_preds = lgbm_model.predict(X_test)
lgbm_test_preds = np.clip(lgbm_test_preds, 0, None) # Zabezpieczenie przed wartościami ujemnymi

# 3. Finałowa ewaluacja
print("══════════ FINAL TEST SET RESULTS ══════════")
res_test_lgbm = evaluate_forecast(y_test.values, lgbm_test_preds, lambda_bc, "LightGBM TEST")

# ===============================================================
# 🔍 DETAILED MULTI-STATION FEATURE IMPORTANCE ANALYSIS
# ===============================================================

fi_df = pd.DataFrame({
    'feature': LGBM_FEATURES_FINAL,
    'importance': lgbm_model.feature_importances_
}).sort_values('importance', ascending=False)

# Kategoryzacja cech
multistation_features = [f for f in fi_df['feature'] if 'station' in f.lower() or 'pm10_' in f.lower() or 'pollution' in f.lower() or 'homogeneity' in f.lower()]
weather_features = [f for f in fi_df['feature'] if any(x in f.lower() for x in ['temp', 'rain', 'wind', 'pressure', 'hdd', 'frost', 'calm'])]
temporal_features = [f for f in fi_df['feature'] if any(x in f.lower() for x in ['month', 'dow', 'week', 'holiday', 'heating'])]
lag_features = [f for f in fi_df['feature'] if 'lag' in f.lower() or 'rolling' in f.lower()]

# Podsumowanie według kategorii
print("\n" + "="*70)
print("🎯 FEATURE IMPORTANCE BY CATEGORY")
print("="*70)

categories = {
    "🚨 Multi-Station (NEW)": multistation_features,
    "🌡️  Weather": weather_features,
    "📅 Temporal": temporal_features,
    "⏳ Lags & Rolling": lag_features,
}

for category_name, features_in_cat in categories.items():
    if features_in_cat:
        avg_importance = fi_df[fi_df['feature'].isin(features_in_cat)]['importance'].mean()
        n_features = len(features_in_cat)
        top_3 = fi_df[fi_df['feature'].isin(features_in_cat)].head(3)
        
        print(f"\n{category_name}")
        print(f"  Count: {n_features} features | Avg Importance: {avg_importance:.4f}")
        for idx, row in top_3.iterrows():
            print(f"    • {row['feature']:30s} → {row['importance']:.4f}")

# Wizualizacja TOP 25
print("\n📊 TOP 25 MOST IMPORTANT FEATURES:")
top_25 = fi_df.head(25)
print(top_25.to_string(index=False))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Top 20
top_20 = fi_df.head(20)
colors_map = {
    'multistation': '#E76F51',
    'weather': '#2A9D8F',
    'temporal': '#8338EC',
    'lags': '#F4A261'
}

def get_category_color(feat):
    if 'station' in feat.lower() or 'pm10_' in feat.lower() or 'pollution' in feat.lower() or 'homogeneity' in feat.lower():
        return colors_map['multistation']
    elif any(x in feat.lower() for x in ['temp', 'rain', 'wind', 'pressure', 'hdd', 'frost', 'calm']):
        return colors_map['weather']
    elif any(x in feat.lower() for x in ['month', 'dow', 'week', 'holiday', 'heating']):
        return colors_map['temporal']
    else:
        return colors_map['lags']

colors = [get_category_color(f) for f in top_20['feature']]

ax1.barh(range(len(top_20)), top_20['importance'].values, color=colors)
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20['feature'].values)
ax1.set_xlabel('Importance')
ax1.set_title('LightGBM Top 20 Features (Color-coded by Category)', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Pie chart - % contribution by category
category_importance = {}
for cat_name, cat_features in categories.items():
    cat_imp = fi_df[fi_df['feature'].isin(cat_features)]['importance'].sum()
    cat_label = cat_name.split()[0]  # Extract emoji
    category_importance[cat_label] = cat_imp

ax2.pie(
    category_importance.values(),
    labels=category_importance.keys(),
    autopct='%1.1f%%',
    startangle=90,
    colors=['#E76F51', '#2A9D8F', '#8338EC', '#F4A261']
)
ax2.set_title('Feature Importance Distribution by Category', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../images/multistation_feature_importance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ✅ Analiza: czy nowe cechy wspomogły?
multistation_importance = fi_df[fi_df['feature'].isin(multistation_features)]['importance'].sum()
total_importance = fi_df['importance'].sum()
multistation_pct = (multistation_importance / total_importance) * 100

import matplotlib.pyplot as plt
import lightgbm as lgb

# Wykres ważności cech
plt.figure(figsize=(10, 12))
lgb.plot_importance(lgbm_model, max_num_features=20, importance_type='gain') # 'gain' jest lepszy niż 'split'
plt.title("Top 20 Features - Dlaczego model tak dobrze zgaduje?")
plt.show()

from scipy.special import inv_boxcox
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def plot_final_results(y_true_transformed, y_pred_transformed, dates, lambda_param, title_suffix="Validation Set"):
    # 1. Odwracamy transformację Box-Cox, aby wrócić do µg/m³
    y_true = inv_boxcox(y_true_transformed, lambda_param)
    y_pred = inv_boxcox(y_pred_transformed, lambda_param)
    
    # 2. Obliczamy metryki dla tego konkretnego zbioru
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # --- WYKRES 1: Szereg czasowy (Ostatnie 90 dni) ---
    show_days = min(90, len(y_true)) 
    ax1.plot(dates[-show_days:], y_true[-show_days:], label='Actual PM10', color='#1f77b4', linewidth=2, marker='o', alpha=0.8)
    ax1.plot(dates[-show_days:], y_pred[-show_days:], label='Predicted PM10', color='#ff7f0e', linewidth=2, linestyle='--', marker='x')
    
    ax1.set_title(f'Temporal Performance: Actual vs Predicted ({title_suffix})', fontsize=16, fontweight='bold')
    ax1.set_ylabel('PM10 Concentration [µg/m³]', fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- WYKRES 2: Scatter Plot (Wykres błędu) ---
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.4, 'color':'#2ca02c'}, 
                line_kws={'color':'red', 'linestyle':':'}, ax=ax2)
    
    # Linia idealna 45 stopni
    lims = [0, max(y_true.max(), y_pred.max())]
    ax2.plot(lims, lims, color='black', alpha=0.5, linestyle='--', label='Perfect Prediction')
    
    # Dodanie ramki z metrykami na wykresie
    stats_text = f'R² = {r2:.4f}\nMAE = {mae:.2f} µg/m³\nRMSE = {rmse:.2f} µg/m³'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax2.set_title(f'Prediction Accuracy & Error Distribution ({title_suffix})', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Measured PM10 [µg/m³]', fontsize=12)
    ax2.set_ylabel('Forecasted PM10 [µg/m³]', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

# --- WYWOŁANIE DLA ZBIORU VAL ---
plot_final_results(y_val, lgbm_preds, y_val.index, lambda_bc, title_suffix="Validation Set")

# --- WYWOŁANIE DLA ZBIORU TEST (Ten, który był odcięty na końcu) ---
# Najpierw musisz wygenerować lgbm_test_preds:
# lgbm_test_preds = lgbm_model.predict(X_test)
# plot_final_results(y_test, lgbm_test_preds, y_test.index, lambda_bc, title_suffix="Final Test Set")

