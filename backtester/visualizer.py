import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .analysis import calculate_metrics
import numpy as np
from datetime import datetime, timedelta

def create_trade_statistics(trades_df):
    """Create detailed trade statistics"""
    # Check if we have 'pnl' or 'profit' column (handle both cases)
    profit_col = 'profit' if 'profit' in trades_df.columns else 'pnl'
    
    stats = {
        'total_trades': len(trades_df),
        'winning_trades': len(trades_df[trades_df[profit_col] > 0]),
        'losing_trades': len(trades_df[trades_df[profit_col] <= 0]),
        'win_rate': len(trades_df[trades_df[profit_col] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'avg_win': trades_df[trades_df[profit_col] > 0][profit_col].mean() if len(trades_df[trades_df[profit_col] > 0]) > 0 else 0,
        'avg_loss': trades_df[trades_df[profit_col] < 0][profit_col].mean() if len(trades_df[trades_df[profit_col] < 0]) > 0 else 0,
        'largest_win': trades_df[profit_col].max() if len(trades_df) > 0 else 0,
        'largest_loss': trades_df[profit_col].min() if len(trades_df) > 0 else 0,
        'total_pnl': trades_df[profit_col].sum() if len(trades_df) > 0 else 0,
        'avg_trade_duration': (trades_df['exit_time'] - trades_df['entry_time']).mean() if len(trades_df) > 0 else timedelta(0)
    }
    return stats

def plot_results(backtest, save_path=None, title=None, model=None):
    """
    Generate interactive HTML report for backtest results with improved visualization
    
    Args:
        backtest: Backtest object containing results
        save_path (str): Optional. Path to save the HTML report
        title (str): Optional. Title for the report
        model (str): Optional. Model information (e.g., 'gpt-4o', 'claude-3.7-sonnet')
    """
    if not backtest.capital_history:
        print("No trades to visualize")
        return
    
    # Create result DataFrame for capital history
    capital_values = backtest.capital_history
    timestamps = backtest.data.index[:len(capital_values)].tolist()
    
    # Ensure both arrays have the same length
    min_length = min(len(timestamps), len(capital_values))
    timestamps = timestamps[:min_length]
    capital_values = capital_values[:min_length]
    
    capital_df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'capital': capital_values
    }).sort_values('timestamp')
    
    # Calculate Market Return (Buy & Hold strategy)
    first_price = backtest.data['close'].iloc[0]
    last_price = backtest.data['close'].iloc[-1]
    market_return_pct = ((last_price / first_price) - 1) * 100
    
    # Calculate market equity curve
    market_capital = []
    for i, price in enumerate(backtest.data['close'][:len(timestamps)]):
        market_value = backtest.initial_capital * (price / first_price)
        market_capital.append(market_value)
    
    capital_df['market_capital'] = market_capital[:min_length]
    
    # Calculate metrics
    metrics = calculate_metrics(backtest)
    
    # Prepare trade data
    trades_df = pd.DataFrame(backtest.trades)
    
    if not trades_df.empty:
        # Convert entry_time and exit_time to datetime if they exist
        if 'entry_time' in trades_df.columns:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        if 'exit_time' in trades_df.columns:
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Map actions to 'long' or 'short' for consistency
        trades_df['action'] = trades_df['action'].apply(
            lambda x: 'long' if 'LONG' in x else 'short' if 'SHORT' in x else x
        )
        
        # Add balance after trade column for hover info
        trades_df['balance_after_trade'] = backtest.initial_capital
        profit_col = 'profit' if 'profit' in trades_df.columns else 'pnl'
        for i, trade in trades_df.iterrows():
            if i > 0 and 'CLOSE' in trade['action']:
                trades_df.loc[i, 'balance_after_trade'] = trades_df['balance_after_trade'].iloc[i-1] + trade[profit_col]
        
        # 중요 수정: 입력/종료 거래 구분 방식 개선
        # trades_df 내부 구조를 분석하고 올바르게 처리
        # 데이터 구조에 따라 맞춤형 처리 필요
        has_separate_exit_rows = any('CLOSE' in action for action in trades_df['action'])
        
        if has_separate_exit_rows:
            # 별도의 종료 행이 있는 경우 (원래 방식)
            entry_trades = trades_df[~trades_df['action'].str.contains('CLOSE')]
            exit_trades = trades_df[trades_df['action'].str.contains('CLOSE')]
        else:
            # 한 행에 모든 정보가 있는 경우 (새로운 방식)
            entry_trades = trades_df.copy()
            exit_trades = trades_df.copy()  # 동일한 행을 사용하지만 종료 데이터로 취급
        
        # Standardize exit reason labels
        # 선택적으로 적용: 종료 사유가 없으면 'Unknown'으로 처리
        if 'exit_reason' not in trades_df.columns:
            trades_df['exit_reason'] = 'Unknown'
            exit_trades['exit_reason'] = 'Unknown'
        else:
            trades_df['exit_reason'] = trades_df['exit_reason'].apply(
                lambda x: 'Take Profit' if x == 'Take Profit' or x == 'TP' else 
                        'Stop Loss' if x == 'Stop Loss' or x == 'SL' else 
                        'Manual Exit' if x == 'no_hit' or x is None or pd.isna(x) else x
            )
        
        # Calculate trade outcome
        profit_col = 'profit' if 'profit' in trades_df.columns else 'pnl'
        trades_df['outcome'] = trades_df[profit_col].apply(
            lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'Breakeven'
        )
        
        # 수정: exit_trades에도 outcome 추가
        if not exit_trades.empty and profit_col in exit_trades.columns:
            exit_trades['outcome'] = exit_trades[profit_col].apply(
                lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'Breakeven'
            )
        
        # Rename columns to match sample (only if not already renamed)
        if 'pnl' in trades_df.columns and 'profit' not in trades_df.columns:
            trades_df = trades_df.rename(columns={
                'pnl': 'profit',
                'size': 'amount'
            })
            if 'pnl' in exit_trades.columns:
                exit_trades = exit_trades.rename(columns={
                    'pnl': 'profit',
                    'size': 'amount'
                })
        elif 'size' in trades_df.columns and 'amount' not in trades_df.columns:
            trades_df = trades_df.rename(columns={'size': 'amount'})
            if 'size' in exit_trades.columns:
                exit_trades = exit_trades.rename(columns={'size': 'amount'})
    
    # Calculate trade statistics
    trade_stats = create_trade_statistics(trades_df) if not trades_df.empty else {
        'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 
        'win_rate': 0, 'avg_win': 0, 'avg_loss': 0,
        'largest_win': 0, 'largest_loss': 0, 'total_pnl': 0,
        'avg_trade_duration': timedelta(0)
    }
    
    # Prepare additional metrics for HTML report
    profit_col = 'profit' if 'profit' in trades_df.columns else 'pnl' if not trades_df.empty else 'profit'
    report_metrics = {
        'initial_balance': backtest.initial_capital,
        'final_balance': capital_values[-1] if len(capital_values) > 0 else backtest.initial_capital,
        'total_return_pct': ((capital_values[-1] / backtest.initial_capital) - 1) * 100 if len(capital_values) > 0 else 0,
        'market_return_pct': market_return_pct,
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'win_rate': trade_stats['win_rate'],
        'profit_factor': metrics.get('profit_factor', 0),
        'max_drawdown_pct': metrics.get('max_drawdown', 0),
        'num_trades': trade_stats['total_trades'],
        'avg_trade': trade_stats['total_pnl'] / trade_stats['total_trades'] if trade_stats['total_trades'] > 0 else 0,
        'avg_holding_time': trade_stats['avg_trade_duration'].total_seconds() / 3600 if trade_stats['avg_trade_duration'] != timedelta(0) else 0
    }
    
    # Set report title
    if title is None:
        title = f'Backtest Results ({backtest.start_date} ~ {backtest.end_date}) with {model}'
    
    # Create more advanced visualization
    fig = make_subplots(
        rows=3, 
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"type": "domain"}, {"colspan": 1}]
        ],
        row_heights=[0.5, 0.2, 0.3],
        vertical_spacing=0.08,
        subplot_titles=(
            "Price Chart with Trade Entries & Exits", 
            "Equity Curve & Drawdown",
            "Trade Distribution",
            "Trade Performance"
        )
    )
    
    # 1. Price Chart with Trades
    fig.add_trace(
        go.Candlestick(
            x=backtest.data.index,
            open=backtest.data['open'],
            high=backtest.data['high'],
            low=backtest.data['low'],
            close=backtest.data['close'],
            name="Price",
            increasing=dict(line=dict(color='#26A69A', width=1), fillcolor='#26A69A'),
            decreasing=dict(line=dict(color='#EF5350', width=1), fillcolor='#EF5350'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # 시각화 설정 - 더 깔끔한 표시를 위한 변수
    entry_exit_pairs = []  # 진입/출구 쌍 저장
    trade_counts = {"long": 0, "short": 0}  # 각 방향별 거래 수
    
    # 범례 중복 방지를 위한 사용된 범례 추적
    used_legends = {
        "Long Entry": False,
        "Short Entry": False,
        "Exit (Take Profit)": False,
        "Exit (Stop Loss)": False,
        "Exit (Manual Exit)": False,
        "Profit Zone": False,
        "Loss Zone": False
    }
    
    # Add trade visualization to price chart
    if not trades_df.empty:
        # Process trades
        entry_trades = trades_df[~trades_df['action'].str.contains('CLOSE')]
        exit_trades = trades_df[trades_df['action'].str.contains('CLOSE')]
        
        # Create a dictionary to map entry trades to exit trades for more reliable pairing
        entry_to_exit_map = {}
        current_position = None
        
        # Map entries to exits more reliably
        for idx, row in trades_df.iterrows():
            if 'CLOSE' not in row['action']:  # Entry trade
                current_position = idx
            else:  # Exit trade
                if current_position is not None:
                    entry_to_exit_map[current_position] = idx
                    current_position = None
        
        # 진입-출구 매핑 처리 및 시각화 준비
        for idx, trade in entry_trades.iterrows():
            action = trade['action']
            trade_counts[action] += 1
            
            # 각 방향별 첫 번째 거래만 범례에 표시
            showlegend_entry = not used_legends[f"{action.capitalize()} Entry"]
            if showlegend_entry:
                used_legends[f"{action.capitalize()} Entry"] = True
            
            # Find corresponding exit trade using our mapping
            exit_idx = entry_to_exit_map.get(idx)
            exit_data = None
            if exit_idx is not None:
                exit_data = exit_trades.loc[exit_idx]
            elif not has_separate_exit_rows and 'exit_price' in trade and not pd.isna(trade['exit_price']):
                # 단일 행 구조일 경우 동일한 행이 출력 데이터
                exit_data = trade
            else:
                # Fallback to previous method
                matching_exits = exit_trades[exit_trades.index > idx]
                if not matching_exits.empty:
                    exit_data = matching_exits.iloc[0]
            
            # 출구 정보 있을 경우만 처리
            if exit_data is not None and 'exit_price' in exit_data and not pd.isna(exit_data['exit_price']):
                exit_price = exit_data['exit_price']
                # Use index for fallback exit_time
                exit_time = exit_data['exit_time'] if 'exit_time' in exit_data and not pd.isna(exit_data['exit_time']) else backtest.data.index[-1]
                
                # 거래 정보 저장
                entry_exit_pairs.append({
                    'entry_time': trade['entry_time'],
                    'entry_price': trade['entry_price'],
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'action': action,
                    'profit': exit_data['profit'],
                    'stop_loss': trade.get('stop_loss'),
                    'take_profit': trade.get('take_profit'),
                    'exit_reason': exit_data.get('exit_reason', 'Unknown')
                })
        
        # 저장된 모든 거래 쌍에 대해 일관된 시각화 적용
        for i, pair in enumerate(entry_exit_pairs):
            action = pair['action']
            arrow_direction = "up" if action == "long" else "down"
            arrow_color = "#00C853" if action == "long" else "#FF5252"
            exit_reason = pair['exit_reason']
            is_win = pair['profit'] > 0
            
            # 진입점 표시
            fig.add_trace(
                go.Scatter(
                    x=[pair['entry_time']],
                    y=[pair['entry_price']],
                    mode='markers',
                    marker=dict(
                        symbol=f"triangle-{arrow_direction}",
                        size=12,
                        color=arrow_color,
                        line=dict(color='white', width=1)
                    ),
                    name=f"{action.capitalize()} Entry",
                    hovertemplate=(
                        f"<b>{action.capitalize()} Entry</b><br>" +
                        "Time: %{x}<br>" +
                        "Price: %{y:.2f} USDT<br>"
                    ),
                    showlegend=i == 0 and not used_legends[f"{action.capitalize()} Entry"],
                    legendgroup=action
                ),
                row=1, col=1
            )
            
            # 거래 구간 색상으로 표시 (이익 = 연한 녹색, 손실 = 연한 빨간색)
            area_color = "rgba(0, 200, 83, 0.15)" if is_win else "rgba(255, 82, 82, 0.15)"
            area_name = "Profit Zone" if is_win else "Loss Zone"
            
            # 각 영역 타입별 첫 거래만 범례 표시
            show_area_legend = False
            if is_win and not used_legends.get("Profit Zone", False):
                show_area_legend = True
                used_legends["Profit Zone"] = True
            elif not is_win and not used_legends.get("Loss Zone", False):
                show_area_legend = True
                used_legends["Loss Zone"] = True
            
            # 거래 구간 영역 추가 - 직사각형 영역으로 변경하여 더 명확하게 표시
            # 투명도를 높여 0.3으로 설정
            area_color = "rgba(0, 200, 83, 0.3)" if is_win else "rgba(255, 82, 82, 0.3)"
            
            # 차트 전체 높이를 사용해 수직 영역으로 표시
            y_min = min(backtest.data['low']) * 0.998
            y_max = max(backtest.data['high']) * 1.002
            
            # 직사각형 영역 추가
            fig.add_shape(
                type="rect",
                x0=pair['entry_time'],
                y0=y_min,
                x1=pair['exit_time'],
                y1=y_max,
                fillcolor=area_color,
                opacity=0.7,
                layer="below",
                line_width=0,
                row=1, col=1
            )
            
            # 범례 항목 별도 추가
            if show_area_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='markers',
                        marker=dict(size=10, color=area_color, symbol="square"),
                        name=area_name,
                        legendgroup=f"area_{is_win}",
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # 수직선으로 출구 지점 표시
            exit_color = "#00C853" if is_win else "#FF5252"
            exit_label = f"Exit ({exit_reason})"
            
            # 범례에 한 번만 표시
            show_exit_legend = False
            if exit_reason == 'Take Profit' and not used_legends["Exit (Take Profit)"]:
                show_exit_legend = True
                used_legends["Exit (Take Profit)"] = True
            elif exit_reason == 'Stop Loss' and not used_legends["Exit (Stop Loss)"]:
                show_exit_legend = True
                used_legends["Exit (Stop Loss)"] = True
            elif exit_reason == 'Manual Exit' and not used_legends["Exit (Manual Exit)"]:
                show_exit_legend = True
                used_legends["Exit (Manual Exit)"] = True
            
            # 수직선 추가
            fig.add_shape(
                type="line",
                x0=pair['exit_time'],
                y0=min(backtest.data['low']) * 0.998,  # 차트 하단 근처
                x1=pair['exit_time'],
                y1=max(backtest.data['high']) * 1.002,  # 차트 상단 근처
                line=dict(
                    color=exit_color,
                    width=1.5,
                    dash="dot",
                ),
                name=exit_label,
                row=1, col=1
            )
            
            # 출구 가격점 표시 (원래 마커 대신 텍스트로 대체)
            profit_text = f"+{pair['profit']:.2f}" if pair['profit'] > 0 else f"{pair['profit']:.2f}"
            
            fig.add_annotation(
                x=pair['exit_time'],
                y=pair['exit_price'],
                text=profit_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=exit_color,
                ax=20,  # 화살표 x축 이동
                ay=0,   # 화살표 y축 이동
                font=dict(
                    size=9,
                    color=exit_color
                ),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=exit_color,
                borderwidth=1,
                borderpad=2,
                row=1, col=1
            )
    
    # 2. Equity Curve & Drawdown
    # Calculate drawdown
    capital_df['peak'] = capital_df['capital'].cummax()
    capital_df['drawdown'] = (capital_df['capital'] - capital_df['peak']) / capital_df['peak'] * 100
    
    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=capital_df['timestamp'],
            y=capital_df['capital'],
            mode='lines',
            name='Strategy Equity',
            line=dict(width=2.5, color='#2196F3'),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)',
            hovertemplate="Strategy: %{y:.2f} USDT<br>Date: %{x}"
        ),
        row=2, col=1
    )
    
    # Add Market Equity curve (Buy & Hold)
    fig.add_trace(
        go.Scatter(
            x=capital_df['timestamp'],
            y=capital_df['market_capital'],
            mode='lines',
            name='Market (Buy & Hold)',
            line=dict(width=2, color='#9C27B0', dash='dash'),
            hovertemplate="Market: %{y:.2f} USDT<br>Date: %{x}"
        ),
        row=2, col=1
    )
    
    # Add drawdown as area chart below
    fig.add_trace(
        go.Scatter(
            x=capital_df['timestamp'],
            y=capital_df['drawdown'],
            mode='lines',
            name='Drawdown %',
            line=dict(width=1, color='#FF5252'),
            fill='tozeroy',
            fillcolor='rgba(255, 82, 82, 0.1)',
            hovertemplate="Drawdown: %{y:.2f}%<br>Date: %{x}"
        ),
        row=2, col=1
    )
    
    # 3. Trade Distribution Visualization
    if not trades_df.empty:
        # Create better trade distribution visualization
        entry_df = trades_df[~trades_df['action'].str.contains('CLOSE')]
        
        # Trade type distribution
        fig.add_trace(
            go.Pie(
                labels=["Long", "Short"],
                values=[
                    len(entry_df[entry_df['action'] == 'long']),
                    len(entry_df[entry_df['action'] == 'short'])
                ],
                name="Trade Types",
                marker=dict(colors=['#00C853', '#FF5252']),
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}",
                hole=0.5,
                textposition='inside',
                insidetextorientation='radial'
            ),
            row=3, col=1
        )
        
        # 4. Trade Performance Analysis
        exit_trades = trades_df[trades_df['action'].str.contains('CLOSE')].copy() if has_separate_exit_rows else trades_df.copy()
        
        # 데이터 구조 진단
        # Print debug statements removed
        
        # Extract trade direction from action
        if not exit_trades.empty:
            if has_separate_exit_rows:
                exit_trades['direction'] = exit_trades['action'].apply(
                    lambda x: 'Long' if 'LONG' in x else 'Short'
                )
            else:
                exit_trades['direction'] = exit_trades['action'].apply(
                    lambda x: 'Long' if x == 'long' else 'Short'
                )
            
            # 간단한 직접 계산방식으로 변경
            # 범주 및 값 직접 계산
            summary_data = {
                'Long': {'Take Profit': {'Win': 0, 'Loss': 0}, 
                         'Stop Loss': {'Win': 0, 'Loss': 0}, 
                         'Manual Exit': {'Win': 0, 'Loss': 0}},
                'Short': {'Take Profit': {'Win': 0, 'Loss': 0}, 
                          'Stop Loss': {'Win': 0, 'Loss': 0}, 
                          'Manual Exit': {'Win': 0, 'Loss': 0}}
            }
            
            # 데이터 분석
            for _, trade in exit_trades.iterrows():
                if 'direction' not in trade or 'outcome' not in trade or 'exit_reason' not in trade:
                    continue
                    
                direction = trade['direction']
                outcome = trade['outcome']
                reason = trade['exit_reason']
                
                # 기본값 사용
                if pd.isna(reason) or reason == '':
                    reason = 'Manual Exit'
                
                # 필요한 키가 있는지 확인
                if direction not in summary_data:
                    summary_data[direction] = {}
                if reason not in summary_data[direction]:
                    summary_data[direction][reason] = {'Win': 0, 'Loss': 0}
                
                # 승/패만 계산 (Breakeven 무시)
                if outcome == 'Win':
                    summary_data[direction][reason]['Win'] += 1
                elif outcome == 'Loss':
                    summary_data[direction][reason]['Loss'] += 1
            
            # 데이터를 플롯용 리스트로 변환
            categories = []
            win_values = []
            loss_values = []
            
            for direction in ['Long', 'Short']:
                for reason in ['Take Profit', 'Stop Loss', 'Manual Exit']:
                    if direction in summary_data and reason in summary_data[direction]:
                        categories.append(f"{direction}:{reason}")
                        win_values.append(summary_data[direction][reason]['Win'])
                        loss_values.append(summary_data[direction][reason]['Loss'])
            
            # Print debug statements removed
        
        # 카테고리가 없는 경우 기본값 사용
        if not exit_trades.empty and len(categories) > 0:
            # Print debug statements removed
            pass
        else:
            # Print debug statements removed
            categories = ['Long:Take Profit', 'Long:Stop Loss', 'Long:Manual Exit', 
                         'Short:Take Profit', 'Short:Stop Loss', 'Short:Manual Exit']
            win_values = [0, 0, 0, 0, 0, 0]
            loss_values = [0, 0, 0, 0, 0, 0]
            
            # 거래 데이터가 있지만 카테고리가 추출되지 않은 경우 
            # 전체 거래 요약 데이터를 고려해 최소 하나의 값 표시
            if not exit_trades.empty:
                total_wins = trade_stats['winning_trades']
                total_losses = trade_stats['losing_trades']
                if total_wins > 0:
                    win_values[0] = total_wins  # Long Take Profit에 표시
                if total_losses > 0:
                    loss_values[1] = total_losses  # Long Stop Loss에 표시
        
        # Create a grouped bar chart for wins
        fig.add_trace(
            go.Bar(
                x=categories,
                y=win_values,
                name="Wins",
                marker_color='#00C853',
                text=win_values,
                textposition='auto',
                hovertemplate="<b>Wins</b><br>%{x}: %{y}"
            ),
            row=3, col=2
        )
        
        # Create a grouped bar chart for losses
        fig.add_trace(
            go.Bar(
                x=categories,
                y=loss_values,
                name="Losses",
                marker_color='#FF5252',
                text=loss_values,
                textposition='auto',
                hovertemplate="<b>Losses</b><br>%{x}: %{y}"
            ),
            row=3, col=2
        )
        
        # Update y-axis range to ensure visibility even with zero values
        max_value = max(max(win_values, default=0), max(loss_values, default=0))
        fig.update_yaxes(
            title_text="Number of Trades",
            range=[0, max_value + 1 or 5],
            gridcolor="#EEEEEE",
            row=3, col=2
        )
    
    # Update layout for the entire figure
    fig.update_layout(
        template="plotly_white",
        height=1200,
        width=1300,
        margin=dict(t=120, b=40, l=40, r=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CCCCCC",
            borderwidth=1
        ),
        title=dict(
            text=f"<b>Crypto AI Backtester</b>",
            x=0.5,
            y=0.99,
            xanchor="center",
            yanchor="top",
            font=dict(size=22, family="Arial, sans-serif")
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Date & Time", 
        row=1, col=1,
        rangeslider_visible=False,
        gridcolor="#EEEEEE"
    )
    fig.update_xaxes(
        title_text="Date & Time", 
        row=2, col=1,
        gridcolor="#EEEEEE"
    )
    
    fig.update_yaxes(
        title_text="Price (USDT)", 
        row=1, col=1,
        gridcolor="#EEEEEE",
        zeroline=True,
        zerolinecolor="#DDDDDD",
        zerolinewidth=1.5
    )
    
    fig.update_yaxes(
        title_text="Equity / Drawdown", 
        row=2, col=1,
        gridcolor="#EEEEEE",
        zeroline=True,
        zerolinecolor="#DDDDDD",
        zerolinewidth=1.5,
        ticksuffix="%",
        secondary_y=True
    )
    
    # Add Y-axis title for equity curve
    fig.update_layout(
        yaxis2=dict(
            title="Equity Value (USDT)",
            side="left",
            gridcolor="#EEEEEE",
            zeroline=True,
            zerolinecolor="#DDDDDD",
            zerolinewidth=1.5
        )
    )
    
    # If save_path is provided, create HTML report with metrics cards
    if save_path:
        # Create HTML content with metrics cards
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Crypto AI Backtester Report: {model}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
                :root {{
                    --primary-color: #2196F3;
                    --success-color: #00C853;
                    --danger-color: #FF5252;
                    --warning-color: #FFC107;
                    --text-color: #212121;
                    --text-light: #757575;
                    --card-bg: #FFFFFF;
                    --bg-light: #F5F7FA;
                    --border-color: #E0E0E0;
                }}
                * {{
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }}
                body {{
                    font-family: 'Inter', sans-serif;
                    background-color: var(--bg-light);
                    color: var(--text-color);
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 2rem 1rem;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 2rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid var(--border-color);
                }}
                .header h1 {{
                    font-size: 2.2rem;
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                    color: var(--primary-color);
                }}
                .header h2 {{
                    font-size: 1.5rem;
                    font-weight: 500;
                    color: var(--text-light);
                }}
                .header h3 {{
                    font-size: 1.2rem;
                    font-weight: 400;
                    color: var(--text-light);
                    margin-top: 0.5rem;
                }}
                .report-card {{
                    background-color: var(--card-bg);
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                    padding: 2rem;
                    margin-bottom: 2rem;
                }}
                .summary-section {{
                    display: flex;
                    flex-wrap: wrap;
                    margin-bottom: 1.5rem;
                    gap: 1rem;
                }}
                .summary-section + .summary-section {{
                    margin-top: 0.5rem;
                    padding-top: 1.5rem;
                    border-top: 1px solid var(--border-color);
                }}
                .summary-item {{
                    flex: 1;
                    min-width: 200px;
                    background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.4) 100%);
                    border-radius: 8px;
                    padding: 1.5rem;
                    text-align: center;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                    border: 1px solid var(--border-color);
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }}
                .summary-item .label {{
                    font-size: 0.9rem;
                    text-transform: uppercase;
                    font-weight: 600;
                    letter-spacing: 0.05em;
                    color: var(--text-light);
                    margin-bottom: 0.5rem;
                }}
                .summary-item .value {{
                    font-size: 1.8rem;
                    font-weight: 700;
                }}
                .summary-item.positive .value {{
                    color: var(--success-color);
                }}
                .summary-item.negative .value {{
                    color: var(--danger-color);
                }}
                .summary-item.neutral .value {{
                    color: var(--primary-color);
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }}
                .metric-card {{
                    background-color: white;
                    padding: 1.5rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                    border: 1px solid var(--border-color);
                }}
                .metric-card .metric-title {{
                    font-size: 0.85rem;
                    color: var(--text-light);
                    font-weight: 500;
                    margin-bottom: 0.5rem;
                    display: flex;
                    align-items: center;
                }}
                .metric-card .metric-value {{
                    font-size: 1.6rem;
                    font-weight: 700;
                    margin-bottom: 0.25rem;
                }}
                .positive {{
                    color: var(--success-color);
                }}
                .negative {{
                    color: var(--danger-color);
                }}
                .chart-container {{
                    width: 100%;
                    margin: 2rem 0;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 3rem;
                    padding-top: 1rem;
                    border-top: 1px solid var(--border-color);
                    color: var(--text-light);
                    font-size: 0.9rem;
                }}
                @media (max-width: 768px) {{
                    .metrics-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .summary-item {{
                        min-width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Crypto AI Backtester Report: {model}</h1>
                    <h2>({backtest.start_date} ~ {backtest.end_date})</h2>
                </div>
                
                <div class="report-card">
                    <div class="summary-section">
                        <div class="summary-item {{'positive' if report_metrics['total_return_pct'] >= 0 else 'negative'}}">
                            <div class="label">Strategy Return</div>
                            <div class="value">{report_metrics['total_return_pct']:.2f}%</div>
                        </div>
                        <div class="summary-item {{'positive' if report_metrics['market_return_pct'] >= 0 else 'negative'}}">
                            <div class="label">Market Return</div>
                            <div class="value">{report_metrics['market_return_pct']:.2f}%</div>
                        </div>
                    </div>
                    
                    <div class="summary-section">
                        <div class="summary-item {{'positive' if report_metrics['sharpe_ratio'] >= 1 else 'neutral'}}">
                            <div class="label">Sharpe Ratio</div>
                            <div class="value">{report_metrics['sharpe_ratio']:.2f}</div>
                        </div>
                        <div class="summary-item {{'positive' if report_metrics['win_rate'] >= 50 else 'neutral'}}">
                            <div class="label">Win Rate</div>
                            <div class="value">{report_metrics['win_rate']:.1f}%</div>
                        </div>
                        <div class="summary-item negative">
                            <div class="label">Max Drawdown</div>
                            <div class="value">{report_metrics['max_drawdown_pct']:.2f}%</div>
                        </div>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-title">Initial Capital</div>
                            <div class="metric-value">{report_metrics['initial_balance']:.2f} USDT</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Final Capital</div>
                            <div class="metric-value {{'positive' if report_metrics['final_balance'] > report_metrics['initial_balance'] else 'negative'}}">{report_metrics['final_balance']:.2f} USDT</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Total Trades</div>
                            <div class="metric-value">{report_metrics['num_trades']}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Profit Factor</div>
                            <div class="metric-value {{'positive' if report_metrics['profit_factor'] > 1 else 'negative'}}">{report_metrics['profit_factor']:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Average Trade</div>
                            <div class="metric-value {{'positive' if report_metrics['avg_trade'] > 0 else 'negative'}}">{report_metrics['avg_trade']:.2f} USDT</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Average Hold Time</div>
                            <div class="metric-value">{report_metrics['avg_holding_time']:.1f} hours</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Winning Trades</div>
                            <div class="metric-value positive">{trade_stats['winning_trades']}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Losing Trades</div>
                            <div class="metric-value negative">{trade_stats['losing_trades']}</div>
                        </div>
                    </div>
                </div>
                
                <div class="chart-container" id="chart"></div>
                
                <div class="footer">
                    <p>Generated by Crypto AI Backtester on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
            
            <script>
                var chartData = {fig.to_json()};
                Plotly.newPlot('chart', chartData.data, chartData.layout);
            </script>
        </body>
        </html>
        """
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Write HTML to file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Enhanced interactive report generated: {save_path}")
    else:
        # Display interactive plot if save_path is not provided
        fig.show()
    
    return