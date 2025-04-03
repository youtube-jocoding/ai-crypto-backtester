import os
import json
from datetime import datetime
from dotenv import load_dotenv
from backtester import CryptoBacktest, plot_results
from backtester.data import fetch_multi_timeframe_data, create_database

# Load environment variables
load_dotenv()

def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """Main function for testing"""
    # Load configuration
    config = load_config()
    backtest_cfg = config['backtest_settings']
    ai_cfg = config['ai_settings']

    symbol = backtest_cfg['symbol']
    start_date = backtest_cfg['start_date']
    end_date = backtest_cfg['end_date']
    initial_capital = backtest_cfg['initial_capital']
    fee = backtest_cfg['fee']

    provider = ai_cfg['provider']
    model = ai_cfg['model']
    
    # Load system prompt from file if specified, otherwise use the prompt in config
    if 'system_prompt_file' in ai_cfg:
        system_prompt_file = ai_cfg['system_prompt_file']
        with open(system_prompt_file, 'r') as f:
            system_prompt = f.read()
    else:
        system_prompt = ai_cfg.get('system_prompt', '')
        
    ai_input_timeframes = ai_cfg['input_data']['timeframes'] # For fetching data

    # Initialize database
    create_database()
    
    # Determine required timeframes: 1m (for execution) + AI input timeframes
    required_timeframes = set(["1m"]) | set(ai_input_timeframes.keys())

    # Fetch all required data first
    print("Fetching market data...")
    market_data = fetch_multi_timeframe_data(
        symbol=symbol, 
        start_date=start_date, 
        end_date=end_date, 
        timeframes=list(required_timeframes), # Pass the required timeframes
        suppress_warnings=True
    )
    
    # Initialize backtest with market data, capital, and fee
    backtest = CryptoBacktest(
        start_date=start_date, 
        end_date=end_date, 
        market_data=market_data,
        initial_capital=initial_capital,
        fee=fee
    )
    
    # Run backtest
    backtest.run_backtest(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        ai_input_config=ai_cfg['input_data'] # Pass AI input config to backtest run
    )
    
    # Plot results and save HTML report
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    # Generate filename based on symbol and date
    if model.startswith("o"):
        filename = f"results/backtest_{model}_{ai_cfg['reasoning_effort']}_{symbol.replace('/', '_')}_{start_date}_to_{end_date}.html"
    else:
        filename = f"results/backtest_{model}_{symbol.replace('/', '_')}_{start_date}_to_{end_date}.html"
    # Plot results and save to HTML file
    plot_results(backtest, save_path=filename, model=model)
    print(f"Report saved to: {filename}")

if __name__ == "__main__":
    main()
