import asyncio
import datetime
import hashlib
import logging
import os
import pickle
import time
from functools import partial, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jdatetime
import numpy as np
import pandas as pd
import seaborn as sns
from fake_useragent import UserAgent
from gldpy import GLD
from scipy.optimize import minimize
from scipy.stats import kurtosis, norm, skew, skewnorm
from tsetmc_api.symbol import Symbol  # Ensure this is a valid import

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# ============================
# Logger Setup
# ============================

def setup_logger(name: str = "OptionAnalysisLogger", log_file: str = "option_analysis.log") -> logging.Logger:
    """
    Sets up a logger with both console and file handlers.

    Args:
        name (str): Name of the logger.
        log_file (str): File path for logging.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

# ============================
# Data Definitions
# ============================

# Mapping of CSV columns to internal names
GENERAL_COLUMN_NAMES: Dict[str, str] = {
    "contract_size": "contract_size",
    "ua_tse_code": "ua_tse_code",
    "ua_ticker": "ua_ticker",
    "ua_close_price": "last_spot_price",
    "ua_yesterday_price": "ua_yesterday_price",
    "begin_date": "begin_date",
    "end_date": "end_date",
    "strike_price": "strike_price",
    "days_to_maturity": "days",
    "tse_code": "tse_code",
    "last_price": "last_price",
    "open_positions": "open_positions",
    "close_price": "close_price",
    "yesterday_price": "yesterday_price",
    "notional_value": "notional_value",
    "trades_value": "trades_value",
    "trades_volume": "trades_volume",
    "trades_num": "trades_num",
    "name": "option_name",
    "ticker": "ticker",
    "bid_price": "bid_price",
    "bid_volume": "bid_volume",
    "ask_price": "ask_price",
    "ask_volume": "ask_volume",
    "yesterday_open_positions": "yesterday_open_positions",
    "option_type": "option_type"
}


# Market scenarios mapping
MARKET_SCENARIOS = {
    'Normal Market': {'skewness': 0.0, 'kurtosis': 3.0},
    'Bullish Market': {'skewness': 0.5, 'kurtosis': 3.0},
    'Bearish Market': {'skewness': -0.5, 'kurtosis': 3.0},
    'High Volatility': {'skewness': 0.0, 'kurtosis': 6.0},
    'Low Volatility': {'skewness': 0.0, 'kurtosis': 2.0},
    'Bullish High Volatility': {'skewness': 0.5, 'kurtosis': 6.0},
    'Bearish High Volatility': {'skewness': -0.5, 'kurtosis': 6.0},
    'Bullish Low Volatility': {'skewness': 0.5, 'kurtosis': 2.0},
    'Bearish Low Volatility': {'skewness': -0.5, 'kurtosis': 2.0},
}

# User-Agent Setup
fake_user_agent = UserAgent()

# ============================
# DiskCache Decorator
# ============================

class DiskCache:
    """
    Decorator for caching function outputs to disk to avoid redundant computations or data fetching.
    """
    def __init__(self, cache_dir: str = 'function_cache', expiry_days: int = 1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.expiry_days = expiry_days

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            # Create a unique hash for the function call based on name and arguments
            key = f"{func.__module__}.{func.__qualname__}:" \
                  f"{pickle.dumps((args, kwargs))}"
            hash_key = hashlib.md5(key.encode()).hexdigest()
            cache_file = self.cache_dir / f"{hash_key}.pkl"
            
            # Check if cache exists and is fresh
            if cache_file.exists():
                cache_mtime = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.datetime.now() - cache_mtime).days < self.expiry_days:
                    try:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        logger.debug(f"Loaded cached result for {func.__name__} from {cache_file}")
                        return result
                    except Exception as e:
                        logger.error(f"Failed to load cache for {func.__name__}: {e}. Recomputing.")
            
            # Call the actual function
            result = func(*args, **kwargs)
            
            # Save the result to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {func.__name__} at {cache_file}")
            except Exception as e:
                logger.error(f"Failed to cache result for {func.__name__}: {e}")
            
            return result
        return wrapped

# ============================
# Data Fetching Module
# ============================

class OptionDataFetcher:
    @staticmethod
    def clean_entire_market_data_from_csv(raw_data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning process for CSV data.")
        
        # Rename columns to match internal naming conventions
        cleaned_df = raw_data.rename(columns=GENERAL_COLUMN_NAMES)
        logger.debug(f"Columns after renaming: {cleaned_df.columns.tolist()}")
        
        # Ensure required columns are present
        required_columns = ['option_name', 'option_type', 'strike_price', 'days',
                            'last_spot_price', 'ask_price', 'bid_price', 'ua_tse_code',
                            'contract_size', 'open_positions']  # Added 'open_positions'
        missing_required = [col for col in required_columns if col not in cleaned_df.columns]
        if missing_required:
            logger.error(f"Missing required columns after cleaning: {missing_required}")
            # Handle missing 'contract_size' by setting a default value
            if 'contract_size' in missing_required:
                logger.warning("Missing 'contract_size' column. Setting default value to 1000.")
                cleaned_df['contract_size'] = 1000
                required_columns.remove('contract_size')
                missing_required.remove('contract_size')

            # Re-check missing required columns
            missing_required = [col for col in required_columns if col not in cleaned_df.columns]
            if missing_required:
                raise KeyError(f"Missing required columns: {missing_required}")
        else:
            logger.debug("All required columns are present after renaming.")
        
        # Select only the required columns
        cleaned_df = cleaned_df[required_columns].copy()
        
        # Additional Data Validation
        logger.info("Performing additional data validation.")
        for col in ['strike_price', 'days', 'last_spot_price', 'ask_price', 'bid_price', 'contract_size']:
            if cleaned_df[col].isna().any():
                logger.error(f"Missing values found in required column: {col}")
                raise KeyError(f"Missing values in required column: {col}")
        
        # Normalize option_type casing
        cleaned_df['option_type'] = cleaned_df['option_type'].str.upper()
        
        # Filter out options with non-positive days to maturity
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df['days'] > 0]
        filtered_count = len(cleaned_df)
        logger.info(f"Filtered out {initial_count - filtered_count} options with non-positive days to maturity.")
        
        # Filter out options with non-positive premiums
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df[(cleaned_df['ask_price'] > 0) & (cleaned_df['bid_price'] > 0)]
        filtered_count = len(cleaned_df)
        logger.info(f"Filtered out {initial_count - filtered_count} options with non-positive premiums.")
        
        # Fill missing 'open_positions' with 0 if any
        if cleaned_df['open_positions'].isna().any():
            logger.warning("Missing values found in 'open_positions'. Filling with 0.")
            cleaned_df['open_positions'].fillna(0, inplace=True)
        
        # Debug log to verify columns
        logger.info(f"Columns in cleaned_data: {cleaned_df.columns.tolist()}")
        
        logger.info("Data cleaning for CSV completed.")
        return cleaned_df


    
#%
#  HISTORICAL DATA FROM TSE
############################

class HistoricalDataFetcher:
    """
    Fetches historical data for symbols with caching to optimize performance.
    """
    CACHE_DIR = Path('historical_data_cache')  # Directory to store cached data
    CACHE_DIR.mkdir(exist_ok=True)  # Create the cache directory if it doesn't exist
    CACHE_EXPIRY_DAYS = 1  # Number of days before cache is considered stale

    @staticmethod
    async def fetch_daily_history(symbol_id: str) -> Optional[pd.DataFrame]:
        """
        Fetches the daily history of a symbol asynchronously with logging and error handling.
        Converts dates from Jalali to Gregorian if necessary.
        Implements caching to avoid redundant data fetching.

        Args:
            symbol_id (str): The symbol ID to fetch data for.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the stock's daily history.
                                     Returns None if fetching fails.
        """
        try:
            logger.info(f"Fetching daily history for symbol_id: {symbol_id}")
            
            # Define the cache file path for the symbol
            cache_file = HistoricalDataFetcher.CACHE_DIR / f"{symbol_id}.pkl"

            # Check if cached data exists and is fresh
            if cache_file.exists():
                cache_mtime = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.datetime.now() - cache_mtime).days < HistoricalDataFetcher.CACHE_EXPIRY_DAYS:
                    logger.info(f"Loading cached data for symbol_id: {symbol_id} from {cache_file}")
                    try:
                        with open(cache_file, 'rb') as f:
                            stock_df = pickle.load(f)
                        return stock_df
                    except Exception as e:
                        logger.error(f"Failed to load cached data for {symbol_id}: {e}. Fetching anew.")
                else:
                    logger.info(f"Cached data for {symbol_id} is stale. Re-fetching.")
            else:
                logger.info(f"No cached data found for symbol_id: {symbol_id}. Fetching data.")

            # Fetch data from the API
            symbol = Symbol(symbol_id=symbol_id)
            daily_history = await symbol.get_daily_history_async()
            if not daily_history:
                logger.warning(f"No daily history data retrieved for symbol_id: {symbol_id}")
                return None

            # Convert data to DataFrame and handle Jalali date conversion
            stock_data = []
            for row in daily_history:
                try:
                    # Convert Jalali date to Gregorian
                    if isinstance(row.date, jdatetime.date):
                        gregorian_date = row.date.togregorian()
                    elif isinstance(row.date, datetime.datetime):
                        gregorian_date = row.date
                    else:
                        gregorian_date = pd.NaT
                        logger.warning(f"Unknown date format for row: {row}")

                    stock_data.append({
                        'date': gregorian_date,
                        'close': row.close,
                        'volume': row.volume,
                        'value': row.value
                    })
                except Exception as e:
                    logger.warning(f"Skipping invalid date value: {row.date} - {e}")

            stock_df = pd.DataFrame(stock_data)
            # Drop rows with invalid dates
            stock_df = stock_df.dropna(subset=['date'])
            logger.info(f"Successfully fetched and converted daily history for symbol_id: {symbol_id}")

            # Save the fetched data to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(stock_df, f)
                logger.info(f"Cached data for symbol_id: {symbol_id} saved to {cache_file}")
            except Exception as e:
                logger.error(f"Failed to cache data for {symbol_id}: {e}")

            return stock_df
        except Exception as e:
            logger.error(f"Error fetching daily history for symbol_id: {symbol_id} - {e}")
            return None

    @staticmethod
    async def fetch_all_symbols(symbol_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetches daily history for all symbols asynchronously with caching.

        Args:
            symbol_ids (List[str]): List of symbol IDs.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbol_id to its historical DataFrame.
        """
        logger.info("Starting asynchronous fetching of historical price data with caching.")
        tasks = [HistoricalDataFetcher.fetch_daily_history(symbol_id) for symbol_id in symbol_ids]
        results = await asyncio.gather(*tasks)

        historical_data = {}
        for symbol_id, df in zip(symbol_ids, results):
            if df is not None and not df.empty:
                historical_data[symbol_id] = df
                logger.debug(f"Historical data fetched for symbol_id: {symbol_id}")
            else:
                logger.warning(f"No historical data available for symbol_id: {symbol_id}")

        logger.info(f"Historical data fetched for {len(historical_data)} out of {len(symbol_ids)} symbols.")
        return historical_data

#%
#  API Option DATA LOADER
############################

class APIDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = pd.DataFrame()
        self.cleaned_data = pd.DataFrame()
        self.historical_data = {}

    def fetch_and_clean_data(self):
        try:
            logger.info(f"Reading data from CSV file: {self.file_path}")
            raw_data = pd.read_csv(self.file_path, encoding='utf-8', dtype=str)
            logger.info(f"CSV data contains {len(raw_data)} records.")

            # Drop any rows where 'contract_size' is 'contract_size' (duplicated headers)
            raw_data = raw_data[raw_data['contract_size'] != 'contract_size']
            logger.info(f"Data contains {len(raw_data)} records after removing duplicated headers.")

            # Convert necessary columns to appropriate data types
            numeric_columns = [
                'contract_size', 'ua_tse_code', 'ua_close_price', 'ua_yesterday_price',
                'strike_price', 'days_to_maturity', 'last_price', 'open_positions',
                'close_price', 'yesterday_price', 'notional_value', 'trades_value',
                'trades_volume', 'trades_num', 'bid_price', 'bid_volume', 'ask_price',
                'ask_volume', 'yesterday_open_positions'
            ]
            for col in numeric_columns:
                if col in raw_data.columns:
                    raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')

            # Clean option data
            self.cleaned_data = OptionDataFetcher.clean_entire_market_data_from_csv(raw_data)
            logger.info(f"Cleaned option data contains {len(self.cleaned_data)} records.")

            # Define required columns including 'open_positions'
            required_columns = ['option_name', 'option_type', 'strike_price', 'days',
                                'last_spot_price', 'ask_price', 'bid_price', 'ua_tse_code',
                                'contract_size', 'open_positions']  # Added 'open_positions'

            # Select only the required columns
            self.cleaned_data = self.cleaned_data[required_columns].copy()
            logger.info(f"Cleaned data contains {len(self.cleaned_data)} records after selecting required columns.")

            # Perform additional validation if necessary
            if self.cleaned_data['open_positions'].isna().any():
                logger.warning("Missing values found in 'open_positions'. Filling with 0.")
                self.cleaned_data['open_positions'].fillna(0, inplace=True)

            # Fetch historical data for unique ua_tse_code
            unique_symbols = self.cleaned_data['ua_tse_code'].unique().tolist()
            logger.info(f"Unique symbols for historical data fetching: {unique_symbols}")

            # Use asyncio to fetch historical data with caching
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.historical_data = loop.run_until_complete(
                HistoricalDataFetcher.fetch_all_symbols(unique_symbols)
            )
            loop.close()
            logger.info("Historical data fetching completed with caching.")
        except Exception as e:
            logger.error(f"Failed to fetch and clean data: {e}")
            raise


    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Returns the cleaned option data.

        Returns:
            pd.DataFrame: Cleaned option data.
        """
        return self.cleaned_data

    def get_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the fetched historical data.

        Returns:
            Dict[str, pd.DataFrame]: Historical data mapped by symbol_id.
        """
        return self.historical_data

#%
#  DATA LOADER
############################



class DataLoader:
    """
    General-purpose data loader for CSV and historical data fetching with caching.
    """
    def __init__(self, csv_file_path: str, cache_dir: str = 'data_cache', cache_expiry_days: int = 1):
        self.csv_file_path = csv_file_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_expiry_days = cache_expiry_days
        self.raw_data = pd.DataFrame()
        self.cleaned_data = pd.DataFrame()
        self.historical_data = {}
    
    @DiskCache(cache_dir='data_cache', expiry_days=1)
    def load_csv_data(self) -> pd.DataFrame:
        """
        Loads raw CSV data, cleans it, and returns the cleaned DataFrame.

        Returns:
            pd.DataFrame: Cleaned option data.
        """
        try:
            logger.info(f"Loading CSV data from {self.csv_file_path}")
            raw_data = pd.read_csv(self.csv_file_path, encoding='utf-8')
            logger.info(f"CSV data contains {len(raw_data)} records.")
            cleaned_df = OptionDataFetcher.clean_entire_market_data_from_csv(raw_data)
            logger.info(f"Cleaned data contains {len(cleaned_df)} records.")
            return cleaned_df
        except Exception as e:
            logger.error(f"Failed to load and clean CSV data: {e}")
            raise

    async def fetch_historical_data_async(self, symbol_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Asynchronously fetches historical data for given symbol IDs.

        Args:
            symbol_ids (List[str]): List of symbol IDs.

        Returns:
            Dict[str, pd.DataFrame]: Historical data mapped by symbol_id.
        """
        try:
            logger.info(f"Fetching historical data for symbols: {symbol_ids}")
            self.historical_data = await HistoricalDataFetcher.fetch_all_symbols(symbol_ids)
            logger.info("Historical data fetching completed.")
            return self.historical_data
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise

    def load_all_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Loads and cleans CSV data and fetches historical data.

        Returns:
            Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]: Cleaned data and historical data.
        """
        try:
            self.cleaned_data = self.load_csv_data()
            unique_symbols = self.cleaned_data['ua_tse_code'].unique().tolist()
            logger.info(f"Unique symbols for historical data: {unique_symbols}")
            
            # Fetch historical data asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.historical_data = loop.run_until_complete(
                HistoricalDataFetcher.fetch_all_symbols(unique_symbols)
            )
            loop.close()
            logger.info("All data loading completed.")
            return self.cleaned_data, self.historical_data
        except Exception as e:
            logger.error(f"Failed to load all data: {e}")
            raise

