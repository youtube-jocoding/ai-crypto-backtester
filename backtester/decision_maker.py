import os
import json
import random
import pandas as pd
from datetime import datetime
from openai import OpenAI

def get_ai_decision(market_data, current_time, provider="random", model=None, system_prompt=None, ai_input_config=None):
    """
    Get AI or random trading decision
    
    Args:
        market_data (dict): Dictionary containing DataFrames for different timeframes
                            (keys are timeframes like '1m', '15m', '1h', etc., values are DataFrames indexed by timestamp)
        current_time (datetime): Current time of the backtest
        provider (str): Decision provider ("random" or "openai")
        model (str): OpenAI model name
        system_prompt (str): System prompt for OpenAI
        ai_input_config (dict): Configuration for AI input data (timeframes, limits)
    """
    if provider == "random":
        return _get_random_decision()
    elif provider == "openai":
        # Pass the full market_data dictionary and AI input config
        return _call_openai_api(market_data, current_time, model, system_prompt, ai_input_config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def _get_random_decision():
    """Generate random trading decision"""
    # Random direction
    direction_choices = ["LONG", "SHORT", "NO_POSITION"]
    direction = random.choice(direction_choices)
    
    # Random parameters
    position_size = round(random.uniform(0.1, 1.0), 2)  # 10%~100%
    leverage = random.randint(1, 5)  # 1x~5x leverage
    stop_loss = round(random.uniform(0.01, 0.05), 2)  # 1%~5% stop loss
    take_profit = round(random.uniform(0.01, 0.1), 2)  # 1%~10% take profit
    
    return {
        "direction": direction,
        "position_size": position_size,
        "leverage": leverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasoning": "Random trading decision"
    }

def _json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def _prepare_market_data(full_market_data, current_time, ai_input_config):
    """
    Prepare multi-timeframe market data for API calls based on ai_input_config.

    Args:
        full_market_data (dict): Dictionary containing DataFrames for different timeframes
        current_time (datetime): Current time of the backtest
        ai_input_config (dict): Configuration specifying timeframes and candle limits.
                                Example: {'timeframes': {'15m': {'limit': 96}, '1h': {'limit': 48}}}

    Returns:
        str: JSON string representing the formatted historical data.
    """
    # Default empty config if not provided
    if ai_input_config is None or 'timeframes' not in ai_input_config:
        print("Warning: AI input configuration is missing or invalid. No data prepared.")
        timeframes_config = {}
    else:
        timeframes_config = ai_input_config['timeframes']

    formatted_data = {
        "current_time": current_time.isoformat(),
        "price_data": {}
    }

    for timeframe, config in timeframes_config.items():
        if timeframe in full_market_data:
            df = full_market_data[timeframe]
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                 print(f"Warning: Index for {timeframe} is not DatetimeIndex. Skipping.")
                 continue

            # Filter data strictly *before* the current_time
            past_data = df[df.index < current_time]

            # Get the last N candles based on the limit in config
            limit = config.get('limit')
            if limit is None or not isinstance(limit, int) or limit <= 0:
                print(f"Warning: Invalid or missing 'limit' for {timeframe} in config. Skipping.")
                continue
            
            n_candles = min(limit, len(past_data))
            if n_candles == 0:
                continue # Skip if no past data available or limit is 0
                
            historical_candles = past_data.tail(n_candles)

            # Convert the relevant data to a list of dictionaries
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            cols_to_use = [col for col in ohlcv_cols if col in historical_candles.columns]

            if not historical_candles.empty:
                 historical_candles_reset = historical_candles[cols_to_use].reset_index()
                 historical_candles_reset = historical_candles_reset.rename(columns={'index': 'timestamp'})
                 formatted_data["price_data"][timeframe] = historical_candles_reset.to_dict(orient='records')

    return json.dumps(formatted_data, default=_json_serial)

def _clean_json_response(response_text):
    """Clean and parse JSON response from AI model"""
    try:
        # Remove JSON code blocks if present
        if "```" in response_text:
            # Extract JSON part
            if response_text.startswith("```"):
                parts = response_text.split("\n", 1)
                if len(parts) > 1:
                    response_text = parts[1]
                if "```" in response_text:
                    response_text = response_text.rsplit("```", 1)[0]
            else:
                pattern = r"```(?:json)?(.*?)```"
                import re
                matches = re.findall(pattern, response_text, re.DOTALL)
                if matches:
                    response_text = matches[0]
        
        return json.loads(response_text.strip())
    except Exception as e:
        print(f"Response parsing error: {e}")
        print(f"Original response: {response_text[:200]}...")
        return {"direction": "NO_POSITION"}

def _call_openai_api(market_data, current_time, model="gpt-4", system_prompt=None, ai_input_config=None):
    """Call OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Set it in .env file.")
    
    client = OpenAI(api_key=api_key)
    
    # Prepare historical data string using the ai_input_config
    market_data_str = _prepare_market_data(market_data, current_time, ai_input_config)

    # Minimal check if any data was prepared
    try:
        prepared_data_check = json.loads(market_data_str)
        if not prepared_data_check.get("price_data") or not prepared_data_check["price_data"]:
            # print(f"Warning: No historical price data prepared for AI at {current_time} based on config. Returning NO_POSITION.")
            return {"direction": "NO_POSITION", "reasoning": "No historical data available based on config"}
    except json.JSONDecodeError:
        print(f"Error decoding prepared market data JSON at {current_time}. Returning NO_POSITION.")
        return {"direction": "NO_POSITION", "reasoning": "Error preparing market data"}

    try:
        # Check if model name starts with "o" to determine role and additional parameters
        is_o_model = model.startswith("o")
        
        # Get additional parameters from ai_input_config if available
        reasoning_effort = None
        if ai_input_config and 'reasoning_effort' in ai_input_config:
            reasoning_effort = ai_input_config.get('reasoning_effort')
        
        # Prepare parameters for API call
        api_params = {
            "model": model,
            "messages": [
                {"role": "developer" if is_o_model else "system", "content": system_prompt},
                {"role": "user", "content": market_data_str}
            ],
        }
        
        # Add temperature only for non-"o" models
        if not is_o_model:
            api_params["temperature"] = 0.0  # 백테스트 결과의 재현성과 일관성을 확보
        
        # Add reasoning_effort for "o" models if specified
        if is_o_model and reasoning_effort:
            api_params["reasoning_effort"] = reasoning_effort
            print(f"Using 'developer' role and reasoning_effort='{reasoning_effort}' for model: {model}")
        
        response = client.chat.completions.create(**api_params)
        
        decision_text = response.choices[0].message.content.strip()
        return _clean_json_response(decision_text)
    
    except Exception as e:
        print(f"OpenAI API call error: {e}")
        return {"direction": "NO_POSITION"}