from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

app = Flask(__name__)

def analyze_stock(ticker, period="2y"): # <- Ubah jadi 2y atau max
    # Download data
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        return {"error": f"Data saham {ticker} tidak ditemukan."}
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Kalkulasi MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    cross_days = 0
    cross_type = "Belum Ada Sinyal"
    macd = df['MACD'].values
    signal = df['Signal_Line'].values

    for i in range(len(df)-1, 0, -1):
        if macd[i] > signal[i] and macd[i-1] <= signal[i-1]:
            cross_days = len(df) - 1 - i
            cross_type = "Golden Cross"
            break
        elif macd[i] < signal[i] and macd[i-1] >= signal[i-1]:
            cross_days = len(df) - 1 - i
            cross_type = "Death Cross"
            break

    # Kalkulasi RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    # Kalkulasi S&R
    n = 10 if len(df) < 60 else 15
    df['Local_Max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]['Close']
    df['Local_Min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]['Close']

    resistances = df['Local_Max'].dropna().unique()
    supports = df['Local_Min'].dropna().unique()

    recent_close = float(df['Close'].iloc[-1])
    recent_rsi = float(df['RSI'].iloc[-1])

    valid_supports = sorted([s for s in supports if s < recent_close], reverse=True)
    support_1 = float(valid_supports[0]) if len(valid_supports) > 0 else float(df['Close'].min())
    support_2 = float(valid_supports[1]) if len(valid_supports) > 1 else support_1 * 0.95

    valid_resistances = sorted([r for r in resistances if r > recent_close])
    resis_1 = float(valid_resistances[0]) if len(valid_resistances) > 0 else float(df['Close'].max())
    resis_2 = float(valid_resistances[1]) if len(valid_resistances) > 1 else resis_1 * 1.05

    # --- LOGIKA KESIMPULAN BARU (Sesuai Request) ---
    if cross_type == "Golden Cross" and cross_days <= 2:
        status_text = "SANGAT BAGUS. Baru saja Golden Cross. Boleh mulai masuk (Buy on Breakout)."
        status_color = "text-green-400"
    elif cross_type == "Golden Cross" and cross_days > 2 and cross_days <= 5:
        status_text = "HATI-HATI. Tren masih naik tapi rawan profit taking. Cek apakah harga mendekati Resistance 1."
        status_color = "text-yellow-400"
    elif cross_type == "Golden Cross" and cross_days > 5:
        status_text = "TERLAMBAT. Sudah lama naik. Lebih baik tunggu koreksi ke Support 1 atau tunggu sinyal MACD baru."
        status_color = "text-orange-400"
    else: # Death Cross
        status_text = "BAHAYA. Sedang fase koreksi. Tunggu harga memantul di Support 1 atau Support 2 sebelum beli."
        status_color = "text-red-400"

    if recent_close < support_1 * 1.02 and recent_rsi < 40:
        strategi_text = "Harga sedang dekat Support 1 dengan RSI rendah. Boleh spekulasi beli dengan target jual di Resis 1."
    elif recent_close > resis_1 * 0.98:
        strategi_text = "Harga hampir menabrak Resistance 1. Rawan koreksi turun. Tahan dulu niat beli."
    else:
        strategi_text = "Harga berada di area tengah. Wait & See."

    # Format untuk TradingView (Menghilangkan .JK dan mengganti menjadi format IDX)
    tv_symbol = ticker.replace('.JK', '')
    if tv_symbol == ticker:
        tv_market = tv_symbol
    else:
        tv_market = f"IDX:{tv_symbol}"

    # Untuk chart ApexCharts (100 hari)
    df_chart = df.tail(100)
    dates = df_chart.index.strftime('%Y-%m-%d').tolist()
    closes = [float(x) for x in df_chart['Close'].tolist()]

    return {
        "ticker": ticker.upper(),
        "tv_symbol": tv_market,
        "close": recent_close,
        "rsi": recent_rsi,
        "cross_type": cross_type, # Tambahan untuk Frontend
        "cross_days": cross_days, # Tambahan untuk Frontend
        "r1": resis_1, "r2": resis_2,
        "s1": support_1, "s2": support_2,
        "status_text": status_text,
        "status_color": status_color,
        "strategi_text": strategi_text,
        "dates": dates,
        "closes": closes
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    data = None
    error = None
    if request.method == 'POST':
        ticker = request.form.get('ticker', 'BUMI').strip().upper()
        if not ticker.endswith('.JK') and not ticker.endswith('.US'):
            ticker += '.JK'
        try:
            data = analyze_stock(ticker)
            if "error" in data:
                error = data["error"]
                data = None
        except Exception as e:
            error = str(e)
    return render_template('index.html', data=data, error=error)

if __name__ == '__main__':
    app.run(debug=True)