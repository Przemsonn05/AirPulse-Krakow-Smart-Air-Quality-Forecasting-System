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
from scipy import stats
from scipy.stats import boxcox
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.special import inv_boxcox

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

krak_stations = krak_stations.interpolate(method='time')

Q1 = krak_stations['MpKrakAlKras'].quantile(0.25)
Q3 = krak_stations['MpKrakAlKras'].quantile(0.75)
IQR = Q3 - Q1

upper_bound = Q3 + 3.0 * IQR

krak_stations.loc[krak_stations['MpKrakAlKras'] > upper_bound, 'MpKrakAlKras'] = np.nan
krak_stations = krak_stations.interpolate(method='time')

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

krak_yearly = df_final['PM10_Krakow_Avg'].resample('YE').agg(['mean', 'max', 'count']).copy()

exceedances = df_final[df_final['PM10_Krakow_Avg'] > 50]['PM10_Krakow_Avg'].resample('YE').size()

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
        data=df_final[df_final['season'] == season]['PM10_Krakow_Avg'],
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
weekly = df_final.groupby('week')['PM10_Krakow_Avg'].agg(['mean', 'std'])
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
    y='PM10_Krakow_Avg',
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
    values='PM10_Krakow_Avg',
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

sns.regplot(data=df_final, x='wind_max', y='PM10_Krakow_Avg', 
            scatter_kws={'alpha':0.3, 'color':"#3A963F"}, line_kws={'color':'red'}, ax=axes[0])
axes[0].set_title('Impact of Wind Speed on PM10', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Max Wind Speed [m/s]', fontsize=12)
axes[0].set_ylabel('PM10 Concentration [µg/m³]', fontsize=12)

sns.regplot(data=df_final, x='temp_avg', y='PM10_Krakow_Avg', 
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
ax1.plot(df_year.index, df_year['PM10_Krakow_Avg'], color=color1, alpha=0.7, label='PM10')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.fill_between(df_year.index, df_year['PM10_Krakow_Avg'], color=color1, alpha=0.1)

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
cols_to_corr = ['PM10_Krakow_Avg', 'temp_avg', 'rain_sum', 'wind_max']
corr_matrix = df_final[cols_to_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f", center=0, linewidths=0.5)
plt.title('Correlation Heatmap: PM10 vs Meteorological Factors', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../images/weather_correlation_heatmap.png', dpi=300)
plt.show()

# Top 15 most polluted days

top_days = df_final['PM10_Krakow_Avg'].nlargest(15).reset_index()
top_days['label'] = top_days['Date'].dt.strftime("%d %b %Y")

norm = colors.Normalize(vmin=top_days['PM10_Krakow_Avg'].min(), vmax=top_days['PM10_Krakow_Avg'].max())
cmap = cm.get_cmap('Reds')
colors_gradient = [cmap(norm(value)) for value in top_days['PM10_Krakow_Avg']]

fig, ax = plt.subplots(figsize=(12 ,7))

bars = ax.barh(
    top_days['label'][::-1], top_days['PM10_Krakow_Avg'][::-1], color=colors_gradient[::-1], 
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

series = df_final['PM10_Krakow_Avg'].dropna()

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

df_final['is_heating_season'] = df_final['month'].isin([1,2,3,10,11,12]).astype(int)

df_final['is_weekend'] = (df_final['week'] >= 5).astype(int)

pm10 = df_final['PM10_Krakow_Avg'].dropna()

pm10_log, lambda_bc = boxcox(pm10 + 1)
df_final['PM10_transformed'] = pm10_log

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

df_final['lag_1d']  = df_final['PM10_Krakow_Avg'].shift(1)
df_final['lag_7d']  = df_final['PM10_Krakow_Avg'].shift(7)
df_final['lag_14d'] = df_final['PM10_Krakow_Avg'].shift(14)
df_final['lag_30d'] = df_final['PM10_Krakow_Avg'].shift(30)

for w in [3, 7, 14, 30]:
    base = df_final['PM10_Krakow_Avg'].shift(1)
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

df_final.dropna(inplace=True)

n = len(df_final)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train = df_final.iloc[:train_end]
val = df_final.iloc[train_end:val_end]
test = df_final.iloc[val_end:]

def evaluate_forecast(y_true, y_pred, lambda_bc, label=''):
    actual = inv_boxcox(y_true, lambda_bc) - 1
    predicted = inv_boxcox(y_pred, lambda_bc) - 1

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"Evaluation Metrics for {label}:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.4f}%")

    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

TARGET = 'PM10_transformed'

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
    "temp_avg", "rain_sum", "wind_max",
    "heating_degree_days", "wind_inverse",
    "is_heating_season", "is_calm_wind",
    "hdd_calm", "rain_3d_sum",
]

sarimax_model = SARIMAX(
    train[TARGET],
    exog=train[SARIMAX_EXOG],
    order=(2, 1, 1),
    seasonal_order=(1, 0, 1, 7),   # ← s=7 (tygodniowość) zamiast błędnego s=24
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

sarimax_preds   = []
current_results = sarimax_model
 
for i in range(len(val)):
    exog_now = val[SARIMAX_EXOG].iloc[i:i+1]
    yhat     = current_results.forecast(steps=1, exog=exog_now)[0]
    sarimax_preds.append(yhat)
 
    actual_y = val[TARGET].iloc[i:i+1]
    current_results = current_results.extend(actual_y, exog=exog_now)
 
sarimax_preds = np.array(sarimax_preds)
res_sarimax   = evaluate_forecast(val[TARGET].values, sarimax_preds, lambda_bc,
                                  "SARIMAX (2,1,1)(1,0,1,7)")

res_sarimax_df = pd.DataFrame([res_sarimax], index=['SARIMAX (2,1,1)(1,0,1,7)'])
res_sarimax_df

PROPHET_REGRESSORS = [
    # Pogoda — wartości znane (egzogeniczne)
    "temp_avg", "rain_sum", "wind_max",
    "heating_degree_days", "wind_inverse",
    "rain_3d_sum", "dry_spell_days",
    # Flagi domenowe
    "is_heating_season", "is_calm_wind",
    "is_frost", "is_weekend", "is_holiday",
    # Interakcje
    "is_frost_calm", "hdd_calm", "cold_dry_calm",
    "inversion_proxy",
    # Rolling (opóźnione o 1 — poprawne)
    "rolling_mean_7d", "rolling_diff_7d",
]

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
    daily_seasonality=False,    # dane dzienne — bez seasonality godzinowej
    seasonality_mode="multiplicative",   # PM10 ma multiplikatywny charakter (spikes)
    changepoint_prior_scale=0.05,        # regularyzacja trendów
)
m_prophet.add_country_holidays(country_name="PL")
for col in PROPHET_REGRESSORS:
    m_prophet.add_regressor(col)
 
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
    # Lagi PM10
    "lag_1d", "lag_7d", "lag_14d", "lag_30d",
    # Rolling PM10
    "rolling_mean_3d", "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_30d",
    "rolling_std_7d", "rolling_std_14d",
    "rolling_max_7d", "rolling_max_14d",
    "rolling_diff_7d", "rolling_diff_14d",
]

import lightgbm as lgb
train_lgb = train.dropna(subset=LGBM_FEATURES + [TARGET])
val_lgb   = val.dropna(subset=LGBM_FEATURES + [TARGET])
 
X_train, y_train = train_lgb[LGBM_FEATURES], train_lgb[TARGET]
X_val,   y_val   = val_lgb[LGBM_FEATURES],   val_lgb[TARGET]
 
lgbm_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
)

lgbm_preds = lgbm_model.predict(X_val)
res_lgbm   = evaluate_forecast(y_val.values, lgbm_preds, lambda_bc, "LightGBM")