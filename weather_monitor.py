import requests
import pandas as pd
import logging
import time
import asyncio
import aiohttp

logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_KEY = ''
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'
CITIES_TO_CHECK = ['Berlin', 'Cairo', 'Dubai', 'Beijing', 'Moscow', 'Rio de Janeiro']


def get_current_temp(city: str, api_key: str) -> dict:
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        return {
            'city': city,
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'description': data['weather'][0]['description']
        }
    except Exception as e:
        logger.error(f"Error fetching weather for {city}: {e}")
        return None


def get_season_stats(df: pd.DataFrame, city: str, season: str) -> dict:
    city_season = df[(df['city'] == city) & (df['season'] == season)]
    
    mean_temp = city_season['temperature'].mean()
    std_temp = city_season['temperature'].std()
    
    return {
        'mean': mean_temp,
        'std': std_temp,
        'lower': mean_temp - 2 * std_temp,
        'upper': mean_temp + 2 * std_temp
    }


def check_anomaly(current_temp: float, stats: dict) -> dict:
    is_anomaly = current_temp < stats['lower'] or current_temp > stats['upper']
    
    if current_temp < stats['lower']:
        status = 'colder than normal'
    elif current_temp > stats['upper']:
        status = 'warmer than normal'
    else:
        status = 'within normal range'
    
    return {
        'is_anomaly': is_anomaly,
        'status': status,
        'deviation': abs(current_temp - stats['mean'])
    }


def get_current_season() -> str:
    from datetime import datetime
    month = datetime.now().month
    
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'


async def get_current_temp_async(session: aiohttp.ClientSession, city: str, api_key: str) -> dict:
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        async with session.get(BASE_URL, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            return {
                'city': city,
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'description': data['weather'][0]['description']
            }
    except Exception as e:
        logger.error(f"Error fetching weather for {city}: {e}")
        return None


async def fetch_all_temps_async(cities: list, api_key: str) -> list:
    async with aiohttp.ClientSession() as session:
        tasks = [get_current_temp_async(session, city, api_key) for city in cities]
        return await asyncio.gather(*tasks)


def fetch_all_temps_sync(cities: list, api_key: str) -> list:
    return [get_current_temp(city, api_key) for city in cities]


if __name__ == "__main__":
    df = pd.read_csv('temperature_data.csv')
    season = get_current_season()
    
    logger.info("Sync version")
    t0 = time.time()
    results_sync = fetch_all_temps_sync(CITIES_TO_CHECK, API_KEY)
    time_sync = time.time() - t0
    logger.info(f"Sync time: {time_sync:.3f}s")
    
    logger.info("Async version")
    t0 = time.time()
    results_async = asyncio.run(fetch_all_temps_async(CITIES_TO_CHECK, API_KEY))
    time_async = time.time() - t0
    logger.info(f"Async time: {time_async:.3f}s")
    
    logger.info(f"\nSpeedup: {time_sync/time_async:.2f}x")
    
    logger.info("Anomaly check")
    for current in results_async:
        if current:
            city = current['city']
            logger.info(f"{city}: {current['temperature']}°C ({current['description']})")
            stats = get_season_stats(df, city, season)

            result = check_anomaly(current['temperature'], stats)
            logger.info(f"Range: {stats['lower']:.1f}°C - {stats['upper']:.1f}°C")
            logger.info(f"Status: {result['status']}")
            if result['is_anomaly']:
                logger.warning("Anomaly!")

                '''
                На момент проверки погода в Рио де Жанейро была аномальной, в остальных городах - нормальной

                INFO - Rio de Janeiro: 36.01°C (clear sky)
                INFO - Range: 10.2°C - 30.0°C
                INFO - Status: warmer than normal
                WARNING - Anomaly!
                '''
    
    '''
    Выводы по сравнению sync и async вариантов:
    
    INFO - Sync time: 1.099s
    INFO - Async time: 0.198s
    INFO - Speedup: 5.55x
    
    Асинхронный подход быстрее в 5.55 раза
    Для запросов к API к нескольким городам async эффективнее
    '''
