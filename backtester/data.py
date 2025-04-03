import pandas as pd
from datetime import datetime
import ccxt
import sqlite3
import os
import time
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

# Global cache for database connections
_db_connection = None
_db_lock = threading.Lock()

def get_db_path():
    """Get the path to the database file in the data folder"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "historical_data.db")

def get_db_connection():
    """Get a database connection with proper thread safety."""
    global _db_connection
    with _db_lock:
        if _db_connection is None:
            _db_connection = sqlite3.connect(get_db_path(), check_same_thread=False)
        return _db_connection

def create_database():
    """Create database and tables if they don't exist"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create table with proper schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                symbol TEXT,
                timeframe TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)
        conn.commit()

def load_data_from_db(symbol, timeframe, start_date, end_date, suppress_warnings=False):
    """Load data from database with validation"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT * FROM historical_data 
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, start_date, end_date))
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate data continuity
        # Convert timeframe for pandas date_range to avoid FutureWarning
        freq = timeframe
        if timeframe.endswith('m'):
            # Replace 'm' with 'min' for minutes
            freq = timeframe.replace('m', 'min')
        elif timeframe.endswith('h'):
            # Keep 'h' as is for hours
            pass
        elif timeframe.endswith('d'):
            # Keep 'd' as is for days
            pass
        
        expected_times = pd.date_range(start=start_date, end=end_date, freq=freq)
        missing_times = expected_times.difference(df['timestamp'])
        
        if not missing_times.empty and not suppress_warnings:
            print(f"Warning: Missing data points for {timeframe} timeframe:")
            print(f"Missing times: {missing_times}")
        
        return df

def timeframe_to_seconds(timeframe):
    """Convert timeframe string to seconds"""
    amount = int(timeframe[:-1])
    unit = timeframe[-1]
    
    if unit == 'm':
        return amount * 60
    elif unit == 'h':
        return amount * 3600
    elif unit == 'd':
        return amount * 86400
    elif unit == 'w':
        return amount * 604800
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")

def fetch_data(symbol, timeframe, start_date, end_date, skip_db_check=False):
    """
    Fetch historical data from Binance
    
    Args:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (e.g., "15m", "1h", "4h")
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        skip_db_check (bool): Skip database check if already done
    
    Returns:
        pd.DataFrame: Historical data
    """
    # Initialize exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })
    
    # Convert dates to timestamps
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
    
    # Convert timeframe to milliseconds
    tf_seconds = timeframe_to_seconds(timeframe)
    tf_ms = tf_seconds * 1000
    
    # Check existing data in database if not skipped
    if not skip_db_check:
        existing_data = load_data_from_db(symbol, timeframe, start_date, end_date)
        if not existing_data.empty:
            print(f"Using existing {timeframe} data from database")
            return existing_data
    
    # Calculate number of candles needed
    total_candles = (end_ts - start_ts) // tf_ms + 1
    
    print(f"Fetching {total_candles} candles for {timeframe} from {start_date} to {end_date}")
    
    # Fetch data in chunks to optimize memory usage
    chunk_size = 1000  # Process 1000 candles at a time
    all_candles = []
    current_ts = start_ts
    
    with tqdm(total=total_candles, desc=f"Fetching {timeframe} candles") as pbar:
        while current_ts < end_ts:
            try:
                # Calculate remaining candles to fetch
                remaining_candles = min(chunk_size, (end_ts - current_ts) // tf_ms + 1)
                
                # Fetch candles
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=remaining_candles
                )
                
                if not candles:
                    break
                
                # Convert to DataFrame and optimize memory
                df = pd.DataFrame(
                    candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Filter data within the specified date range
                df = df[(df['timestamp'] >= pd.to_datetime(start_date)) & 
                       (df['timestamp'] <= pd.to_datetime(end_date))]
                
                if not df.empty:
                    # Optimize memory usage
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    all_candles.append(df)
                
                # Update progress
                pbar.update(len(candles))
                
                # Update timestamp
                current_ts = candles[-1][0] + tf_ms
                
                # Add delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
    
    if not all_candles:
        return pd.DataFrame()
    
    # Combine all DataFrames efficiently
    df = pd.concat(all_candles, ignore_index=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df

def save_to_database(data, symbol, timeframe):
    """Save data to database with validation"""
    if data.empty:
        return
    
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        # Validate data before saving
        data = data.copy()
        data['symbol'] = symbol
        data['timeframe'] = timeframe
        
        # Convert timestamp to string format for SQLite
        data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Remove any existing data for this period
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM historical_data 
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp BETWEEN ? AND ?
        """, (symbol, timeframe, data['timestamp'].min(), data['timestamp'].max()))
        
        # Save new data
        data.to_sql('historical_data', conn, if_exists='append', index=False)
        conn.commit()

def fetch_single_timeframe(args):
    """Helper function for parallel data fetching"""
    symbol, timeframe, start_date, end_date, suppress_warnings = args
    try:
        # Try to load from database first
        data = load_data_from_db(symbol=symbol, timeframe=timeframe, 
                               start_date=start_date, end_date=end_date,
                               suppress_warnings=suppress_warnings)
        if not data.empty:
            return timeframe, data
    except Exception as e:
        print(f"Failed to load {timeframe} data from database: {e}")
    
    # Fetch from Binance if database load fails
    print(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}")
    data = fetch_data(symbol=symbol, timeframe=timeframe,
                     start_date=start_date, end_date=end_date)
    
    if not data.empty:
        save_to_database(data, symbol, timeframe)
    
    return timeframe, data

def fetch_multi_timeframe_data(symbol, start_date, end_date, timeframes=None, suppress_warnings=False):
    """
    Fetch historical data for multiple timeframes in parallel
    
    Args:
        symbol (str): Trading pair symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        timeframes (list, optional): List of timeframes to fetch (e.g., ["1m", "15m", "1h"]). 
                                      Defaults to ["1m", "15m", "1h", "4h"] if None.
        suppress_warnings (bool): Whether to suppress warnings about missing data points
    
    Returns:
        dict: Dictionary with timeframes as keys and DataFrames as values
    """
    # Use default timeframes if none provided
    if timeframes is None:
        timeframes = ["1m", "15m", "1h", "4h"]
    
    # Prepare arguments for parallel fetching
    args = [(symbol, tf, start_date, end_date, suppress_warnings) for tf in timeframes]
    
    # Fetch data in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for timeframe, data in executor.map(fetch_single_timeframe, args):
            if not data.empty:
                results[timeframe] = data
    
    return results

def main():
    """Main function for testing"""
    symbol = "BTC/USDT"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    timeframes = ["15m", "1h", "4h"]
    
    data = fetch_multi_timeframe_data(symbol, start_date, end_date)
    print(f"Retrieved data for {len(data)} timeframes")

if __name__ == "__main__":
    main() 