import pandas as pd
from tqdm import tqdm
from .data import load_data_from_db
from .decision_maker import get_ai_decision
from .analysis import calculate_metrics, print_results
import numpy as np

class CryptoBacktest:
    def __init__(self, start_date, end_date, market_data, initial_capital=10000, fee=0.0005):
        """
        Initialize backtest with market data
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            market_data (dict): Dictionary of market data DataFrames for different timeframes,
                                where DataFrames have a 'timestamp' column.
            initial_capital (float): Starting capital for the backtest.
            fee (float): Trading fee percentage (e.g., 0.0005 for 0.05%).
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.market_data = market_data
        
        # Ensure all market data DataFrames are indexed by timestamp and sorted
        for tf, df in self.market_data.items():
            if 'timestamp' not in df.columns:
                 raise ValueError(f"DataFrame for timeframe {tf} must have a 'timestamp' column.")
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                 try:
                     df['timestamp'] = pd.to_datetime(df['timestamp'])
                 except Exception as e:
                      raise ValueError(f"Could not convert 'timestamp' column to datetime for {tf}: {e}")
            
            # Set timestamp as index if not already
            if not isinstance(df.index, pd.DatetimeIndex):
                 df.set_index('timestamp', inplace=True)
            
            # Sort by index
            if not df.index.is_monotonic_increasing:
                 df.sort_index(inplace=True)
            self.market_data[tf] = df # Store processed df back

        # Check if we have the required 1m timeframe data
        if "1m" not in self.market_data or self.market_data["1m"].empty:
            raise ValueError("1m timeframe data is required for execution and cannot be empty.")

        # Get the main 1m data (already indexed and sorted)
        self.data = self.market_data["1m"].copy()
        
        # Initialize capital and trading parameters from arguments
        self.initial_capital = initial_capital
        self.capital = self.initial_capital
        self.fee = fee  # Use fee from argument
        
        # Initialize position tracking
        self.position = None
        self.position_size = 0
        self.leverage = 1
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trades = []
        self.capital_history = []
        
        # Initialize performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0
    
    def run_backtest(self, provider="openai", model="gpt-4o", system_prompt=None, ai_input_config=None):
        """Run the backtest"""
        print(f"\n===== Starting Backtest: {self.start_date} ~ {self.end_date} =====")
        print(f"Decision method: {provider}_{model}")
        
        # Initialize results
        self.trades = []
        self.capital_history = []
        self.capital = self.initial_capital
        self.position = None
        self.capital_history.append(self.capital)
        
        # Get all timestamps from 1m data index
        timestamps = self.data.index.tolist()
        
        # Run backtest
        for current_time in tqdm(timestamps, desc="Processing"):
            # --- Data Alignment: Get the most recent data point AT OR BEFORE current_time --- 
            current_market_snapshot = {}
            all_data_available = True
            for timeframe, df in self.market_data.items():
                 # Find the index of the latest data point at or before current_time
                 idx_loc = df.index.searchsorted(current_time, side='right') - 1
                 if idx_loc < 0:
                      # No data available at or before this time for this timeframe
                      # print(f"Warning: No data available for {timeframe} at or before {current_time}. Skipping step.")
                      all_data_available = False
                      break
                 current_market_snapshot[timeframe] = df.iloc[idx_loc].to_dict()
            
            if not all_data_available:
                 self.capital_history.append(self.capital) # Record capital even if skipping
                 continue
            # --- End Data Alignment --- 

            # Get the current price from the 1m snapshot for execution logic
            try:
                current_price = current_market_snapshot['1m']['close']
            except KeyError:
                 print(f"Error: Could not get '1m' close price at {current_time}. Skipping step.")
                 self.capital_history.append(self.capital)
                 continue
            
            # Check if we need to close a position due to stop loss or take profit
            if self.position:
                # Check for stop loss or take profit conditions
                if self.position == 'LONG':
                    # Long position: stop loss is below entry, take profit is above entry
                    if current_price <= self.stop_loss:
                        self.close_position(current_price, current_time, "Stop Loss")
                    elif current_price >= self.take_profit:
                        self.close_position(current_price, current_time, "Take Profit")
                else:  # SHORT position
                    # Short position: stop loss is above entry, take profit is below entry
                    if current_price >= self.stop_loss:
                        self.close_position(current_price, current_time, "Stop Loss")
                    elif current_price <= self.take_profit:
                        self.close_position(current_price, current_time, "Take Profit")
            
            # Only get AI decision if no position is open
            if not self.position:
                # --- AI Decision Call: Pass the *full* market data --- 
                decision = get_ai_decision(
                    market_data=self.market_data, # Pass the dict of DataFrames
                    current_time=current_time,
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                    ai_input_config=ai_input_config # Pass the config here
                )
                # --- End AI Decision Call ---

                # Execute decision only if ALL required parameters are present
                required_keys = ['direction', 'position_size', 'leverage', 'stop_loss', 'take_profit']
                if all(key in decision for key in required_keys):
                    # Now we are sure all keys exist, use direct access
                    direction = decision['direction']
                    position_size_ratio = decision['position_size']
                    leverage = decision['leverage']
                    stop_loss_pct = decision['stop_loss']
                    take_profit_pct = decision['take_profit']

                    # Optional: Add further validation for values if needed (e.g., numeric types, ranges)

                    if direction == 'LONG':
                        sl_price = current_price * (1 - stop_loss_pct)
                        tp_price = current_price * (1 + take_profit_pct)
                        self.open_position('LONG', current_price, position_size_ratio,
                                         leverage, sl_price, tp_price, current_time)
                    elif direction == 'SHORT':
                        sl_price = current_price * (1 + stop_loss_pct)
                        tp_price = current_price * (1 - take_profit_pct)
                        self.open_position('SHORT', current_price, position_size_ratio,
                                         leverage, sl_price, tp_price, current_time)
                    # Note: If direction is not LONG or SHORT, it won't match the conditions above.
                # else:
                    # Optional: Log that the trade was skipped due to missing keys
                    # print(f"Skipping trade at {current_time}: Incomplete AI decision - missing keys.")

            # Update capital history
            self.capital_history.append(self.capital)
        
        # Close any open position at the end
        final_timestamp = timestamps[-1] if timestamps else None
        if self.position and final_timestamp:
            try:
                final_price = self.data.loc[final_timestamp]['close']
                self.close_position(final_price, final_timestamp, "End of Backtest")
            except KeyError:
                 print(f"Warning: Could not find final price data at {final_timestamp} to close position.")
        
        # Calculate metrics
        metrics = calculate_metrics({
            'trades': self.trades,
            'capital_history': self.capital_history,
            'initial_capital': self.initial_capital,
            'capital': self.capital
        })
        
        # Print results
        print_results({
            'trades': self.trades,
            'capital_history': self.capital_history,
            'initial_capital': self.initial_capital,
            'capital': self.capital
        }, metrics)
        
        return metrics, self.trades, self.capital_history
    
    def open_position(self, direction, price, size_ratio, leverage, stop_loss, take_profit, timestamp):
        """Open a new position with proper capital management"""
        if self.position:
            return
        
        # Calculate capital to invest based on the size ratio from AI
        capital_to_invest = self.capital * size_ratio

        # Check if capital to invest is positive
        if capital_to_invest <= 0:
             print(f"Warning: Calculated capital to invest is zero or negative ({capital_to_invest:.2f}) at {timestamp}. Skipping trade.")
             return
        
        # Calculate the final position value by applying leverage
        position_value = capital_to_invest * leverage

        # Calculate fees based on the leveraged position value
        entry_fee = position_value * self.fee
        
        # Check if capital is sufficient for the investment amount *plus* the entry fee
        # Note: We compare against capital_to_invest, not self.capital, to ensure margin requirements are met.
        # A more sophisticated check might consider maintenance margin.
        if capital_to_invest <= entry_fee:
             print(f"Warning: Insufficient capital ({capital_to_invest:.2f}) for entry fee ({entry_fee:.2f}) at {timestamp}. Skipping trade.")
             return
             
        # Update position
        self.position = direction
        self.position_size = position_value # Store the leveraged position value
        self.leverage = leverage
        self.entry_price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Record trade
        self.trades.append({
            'action': f'OPEN_{direction.upper()}',
            'entry_time': timestamp,
            'entry_price': price,
            'direction': direction,
            'size': position_value, # Record the leveraged position value
            'leverage': leverage,
            'stop_loss': stop_loss, # Store price level
            'take_profit': take_profit, # Store price level
            'fee': entry_fee
        })
        
        # Update capital (only deduct fees)
        # The actual capital used for margin is implicitly handled by the broker in real trading.
        # Here, we only account for the fee deduction.
        self.capital -= entry_fee
    
    def close_position(self, price, timestamp, reason):
        """Close position and update capital"""
        if not self.position:
            return
        
        # Calculate PnL based on the full leveraged position size
        if self.position == 'LONG':
            pnl = (price - self.entry_price) / self.entry_price * self.position_size # self.position_size already includes leverage effect
        else:  # short
            pnl = (self.entry_price - price) / self.entry_price * self.position_size # self.position_size already includes leverage effect
        
        # Calculate fees based on the full leveraged position size
        exit_fee = self.position_size * self.fee
        
        # Update capital
        self.capital += pnl - exit_fee
        
        # Update trade record (find the last open trade)
        # Find the last OPEN action to update
        last_open_trade_index = -1
        for i in range(len(self.trades) - 1, -1, -1):
            if self.trades[i]['action'].startswith('OPEN'):
                 last_open_trade_index = i
                 break
                 
        if last_open_trade_index != -1:
            self.trades[last_open_trade_index].update({
                'action': f'CLOSE_{self.position}',
                'exit_time': timestamp,
                'exit_price': price,
                'pnl': pnl,
                'exit_fee': exit_fee,
                'exit_reason': reason
            })
        else:
             print(f"Error: Could not find matching open trade to close at {timestamp}")

        
        # Update performance metrics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.total_profit += pnl
        
        # Reset position
        self.position = None
        self.position_size = 0
        self.leverage = 1
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0