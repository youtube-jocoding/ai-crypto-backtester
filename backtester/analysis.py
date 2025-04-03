import pandas as pd
import numpy as np

def calculate_metrics(backtest):
    """Calculate comprehensive performance metrics"""
    # Support both dictionary and object based input
    if hasattr(backtest, 'trades'):
        trades = backtest.trades
        capital_history = backtest.capital_history
        initial_capital = backtest.initial_capital
    else:
        trades = backtest.get('trades', [])
        capital_history = backtest.get('capital_history', [])
        initial_capital = backtest.get('initial_capital', 0)
    
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'total_profit': 0,
            'total_return': 0,
            'avg_profit_per_trade': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_trade_duration': 0
        }
    
    # Get closed trades
    closed_trades = [t for t in trades if t.get('action', '').startswith('CLOSE')]
    
    if not closed_trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'total_profit': 0,
            'total_return': 0,
            'avg_profit_per_trade': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_trade_duration': 0
        }
    
    # Calculate basic metrics
    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # Calculate profit metrics
    total_profit = sum(t.get('pnl', 0) for t in closed_trades)
    avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
    
    # Calculate total return
    final_capital = capital_history[-1] if capital_history else initial_capital
    total_return = (final_capital / initial_capital - 1) * 100 if initial_capital else 0
    
    # Calculate win/loss metrics
    avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
    largest_win = max((t.get('pnl', 0) for t in winning_trades), default=0)
    largest_loss = min((t.get('pnl', 0) for t in losing_trades), default=0)
    
    # Calculate profit factor
    total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades)) if losing_trades else 0
    profit_factor = sum(t.get('pnl', 0) for t in winning_trades) / total_losses if total_losses > 0 else float('inf')
    
    # Calculate drawdown
    if capital_history:
        capital_series = pd.Series(capital_history)
        rolling_max = capital_series.expanding().max()
        drawdowns = (capital_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) * 100 if not drawdowns.empty else 0
    else:
        max_drawdown = 0
    
    # Calculate Sharpe and Sortino ratios
    if capital_history and len(capital_history) > 1:
        returns = pd.Series(capital_history).pct_change().dropna()
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
            downside_returns = returns[returns < 0]
            sortino_ratio = np.sqrt(252) * (returns.mean() / downside_returns.std()) if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
    else:
        sharpe_ratio = 0
        sortino_ratio = 0
    
    # Calculate average trade duration
    if closed_trades:
        durations = []
        for trade in closed_trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                duration = (exit_time - entry_time).total_seconds() / 3600  # Convert to hours
                durations.append(duration)
        avg_trade_duration = sum(durations) / len(durations) if durations else 0
    else:
        avg_trade_duration = 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'total_profit': total_profit,
        'total_return': total_return,
        'avg_profit_per_trade': avg_profit_per_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'avg_trade_duration': avg_trade_duration
    }

def print_results(backtest, metrics=None):
    """Print backtest results"""
    if not metrics:
        metrics = calculate_metrics(backtest)
    
    # Support both dictionary and object based input
    if hasattr(backtest, 'trades'):
        initial_capital = backtest.initial_capital
        final_capital = backtest.capital
    else:
        initial_capital = backtest.get('initial_capital', 0)
        final_capital = backtest.get('capital', 0)
    
    print(f"\n===== Backtest Results =====")
    print(f"Initial capital: ${initial_capital:.2f}")
    print(f"Final capital: ${final_capital:.2f}")
    print(f"Total profit: ${final_capital - initial_capital:.2f}")
    print(f"Return: {((final_capital / initial_capital - 1) * 100):.2f}%")
    
    print(f"\n----- Trade Statistics -----")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    
    print(f"\n----- Average and Risk Metrics -----")
    print(f"Average profit per trade: ${metrics['avg_profit_per_trade']:.2f}")
    print(f"Average win: ${metrics['avg_win']:.2f}")
    print(f"Average loss: ${metrics['avg_loss']:.2f}")
    print(f"Largest win: ${metrics['largest_win']:.2f}")
    print(f"Largest loss: ${metrics['largest_loss']:.2f}")
    print(f"Maximum drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Average trade duration: {metrics['avg_trade_duration']:.2f} hours")