import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import logging
import time
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_PATH = 'temperature_data.csv'
CITY_TIMESERIES_FILENAME = 'city_temperature_timeseries.png'
CITY_SEASON_FILENAME = 'city_season_analysis.png'
WINDOW_SIZE = 30  # окно для скользящего среднего (в днях)


def read_csv(csv_path: str) -> pd.DataFrame:
    logger.info(f"Reading CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    return df


def calc_moving_avg(df: pd.DataFrame, parallel: bool = False) -> pd.DataFrame:
    logger.info(f"[parallel={parallel}] Calculating moving average by city with window size {WINDOW_SIZE}")

    if not parallel:
        for city in df['city'].unique():
            mask = df['city'] == city
            df.loc[mask, 'moving_mean'] = df.loc[mask, 'temperature'].rolling(window=WINDOW_SIZE, center=True).mean()
            df.loc[mask, 'moving_std'] = df.loc[mask, 'temperature'].rolling(window=WINDOW_SIZE, center=True).std()
    else:
        cities = df['city'].unique()
        with Pool(cpu_count()) as pool:
            results = pool.map(calc_city_moving, [(df[df['city'] == city], city) for city in cities])
        
        for city, mean_vals, std_vals in results:
            mask = df['city'] == city
            df.loc[mask, 'moving_mean'] = mean_vals
            df.loc[mask, 'moving_std'] = std_vals

    return df


def calc_city_moving(args):
    city_df, city = args
    mean_vals = city_df['temperature'].rolling(window=WINDOW_SIZE, center=True).mean().values
    std_vals = city_df['temperature'].rolling(window=WINDOW_SIZE, center=True).std().values
    
    return city, mean_vals, std_vals


def calc_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Detecting anomalies")

    df['lower_bound'] = df['moving_mean'] - 2 * df['moving_std']
    df['upper_bound'] = df['moving_mean'] + 2 * df['moving_std']
    df['is_anomaly'] = (df['temperature'] < df['lower_bound']) | (df['temperature'] > df['upper_bound'])

    return df


def calc_trend(df: pd.DataFrame) -> dict:
    logger.info("Calculating trend by city")

    trends = {}
    for city in df['city'].unique():
        city_df = df[df['city'] == city].copy()
        city_df['days'] = (city_df['timestamp'] - city_df['timestamp'].min()).dt.days

        slope, intercept, r_value, p_value, std_err = stats.linregress(city_df['days'], city_df['temperature'])
        df.loc[df['city'] == city, 'trend'] = slope * city_df['days'].values + intercept
        trends[city] = (slope, r_value, p_value)

    return trends


def calc_city_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating statistics by city and season")

    stats = df.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()
    stats.columns = ['city', 'season', 'mean_temp', 'std_temp']

    return stats


def plot_timeseries(df: pd.DataFrame, trends: dict):
    logger.info("Plotting timeseries by city")

    cities = df['city'].unique()
    n_cities = len(cities)
    
    _, axes = plt.subplots(n_cities, 3, figsize=(18, 4*n_cities))
    if n_cities == 1:
        axes = axes.reshape(1, -1)
    
    for idx, city in enumerate(cities):
        city_df = df[df['city'] == city]
        slope, r_value, p_value = trends[city]
        
        ax0 = axes[idx, 0]
        ax0.plot(city_df['timestamp'], city_df['temperature'], alpha=0.5, label='Температура')
        ax0.plot(city_df['timestamp'], city_df['moving_mean'], color='black', linewidth=2, label=f'Скользящее среднее ({WINDOW_SIZE} дней)')
        ax0.fill_between(city_df['timestamp'], city_df['lower_bound'], city_df['upper_bound'], alpha=0.2, label='±2σ')
        ax0.set_ylabel('Температура (°C)', fontsize=10)
        ax0.legend(loc='lower right')
        ax0.set_title(f'{city}: Скользящее среднее', fontsize=11, fontweight='bold')
        ax0.grid(True, alpha=0.2)
        
        ax1 = axes[idx, 1]
        anomalies = city_df[city_df['is_anomaly']]
        ax1.plot(city_df['timestamp'], city_df['temperature'], alpha=0.3, color='gray')
        ax1.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', s=10, label=f'Аномалии ({len(anomalies)})')
        ax1.set_ylabel('Температура (°C)', fontsize=10)
        ax1.legend(loc='lower right')
        ax1.set_title(f'{city}: Аномалии', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.2)
        
        ax2 = axes[idx, 2]
        ax2.plot(city_df['timestamp'], city_df['temperature'], alpha=0.3, label='Температура')
        ax2.plot(city_df['timestamp'], city_df['trend'], color='darkred', linewidth=2, label=f'Тренд: {slope*365:.2f}°C/год')
        ax2.set_ylabel('Температура (°C)', fontsize=10)
        ax2.legend(loc='lower right')
        stats_text = f'R² = {r_value**2:.3f}\np = {p_value:.3f}'
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=9, 
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))
        ax2.set_title(f'{city}: Долгосрочный тренд', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(CITY_TIMESERIES_FILENAME)

    logger.info(f"Timeseries plot saved to {CITY_TIMESERIES_FILENAME}")


def plot_season(df: pd.DataFrame, stats: pd.DataFrame):
    logger.info("Plotting season analysis by city")

    cities = df['city'].unique()
    n_cities = len(cities)
    
    _, axes = plt.subplots(n_cities, 2, figsize=(16, 4*n_cities))
    if n_cities == 1:
        axes = axes.reshape(1, -1)
    
    for idx, city in enumerate(cities):
        city_df = df[df['city'] == city]
        city_stats = stats[stats['city'] == city]
        
        ax1 = axes[idx, 0]
        seasons = city_stats['season'].values
        means = city_stats['mean_temp'].values
        stds = city_stats['std_temp'].values
        
        ax1.bar(seasons, means, yerr=stds*2, capsize=5, alpha=0.7, color='steelblue')
        ax1.set_ylabel('Температура (°C)', fontsize=10)
        ax1.set_title(f'{city}: Средняя температура ±2σ', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.2, axis='y')
        
        ax2 = axes[idx, 1]
        anomalies = city_df[city_df['is_anomaly']]
        normal = city_df[~city_df['is_anomaly']]
        
        ax2.scatter(normal['timestamp'], normal['temperature'], s=1, alpha=0.3, color='gray', label='Норма')
        ax2.scatter(anomalies['timestamp'], anomalies['temperature'], s=10, color='red', label=f'Аномалии ({len(anomalies)})')
        ax2.set_xlabel('Дата', fontsize=10)
        ax2.set_ylabel('Температура (°C)', fontsize=10)
        ax2.set_title(f'{city}: Аномалии', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(CITY_SEASON_FILENAME)

    logger.info(f"City analysis plot saved to {CITY_SEASON_FILENAME}")


if __name__ == "__main__":
    logger.info("Starting temperature analysis")

    df = read_csv(DATA_PATH)
    
    # sequential
    df_seq = df.copy()
    t0 = time.time()
    df_seq = calc_moving_avg(df_seq, parallel=False)
    time_seq = time.time() - t0
    
    # parallel
    df_par = df.copy()
    t0 = time.time()
    df_par = calc_moving_avg(df_par, parallel=True)
    time_par = time.time() - t0
    
    logger.info(f"Sequential: {time_seq:.4f}s")
    logger.info(f"Parallel: {time_par:.4f}s")
    logger.info(f"Speedup: {time_seq/time_par:.2f}x")

    '''
    Выводы по сравнению sequential и parallel вариантов:

    INFO - Sequential: 0.0249s
    INFO - Parallel: 1.4558s
    INFO - Speedup: 0.02x

    Параллельная версия медленнее из-за overhead на создание процессов
    Для небольших данных, как в нашем случае, последовательный вариант эффективнее
    Параллелизм выгоден только на больших объемах данных
    '''
    
    df = df_par
    df = calc_anomalies(df)
    trends = calc_trend(df)
    plot_timeseries(df, trends)
    
    stats = calc_city_season_stats(df)
    plot_season(df, stats)
    
    for city in stats['city'].unique():
        city_df = df[df['city'] == city]
        n_anom_ts = city_df['is_anomaly'].sum()

        logger.info(f"{city}:")
        logger.info(f"\tAnomalies: {n_anom_ts} ({n_anom_ts/len(city_df)*100:.2f}%)")

    logger.info("Temperature analysis completed")
