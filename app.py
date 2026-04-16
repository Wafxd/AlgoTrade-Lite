from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import math

app = Flask(__name__)

def analyze_stock(ticker, period="6mo"):
    # Kita buat object Ticker dulu untuk fallback nanti
    ticker_obj = yf.Ticker(ticker)
    
    # Download data historis harian untuk Analisa
    df = yf.download(ticker, period=period, progress=False)
    
    if df.empty:
        return {"error": f"Data saham {ticker} tidak ditemukan."}
    
    # 1. BUKA BUNGKUS KOLOM DULU (Flatten MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # 2. BERSIHKAN DATA KOSONG (NaN)
    if 'Close' in df.columns:
        df = df.dropna(subset=['Close'])
    else:
        return {"error": f"Format data saham {ticker} dari Yahoo Finance tidak dikenali."}
    
    # Pengaman tambahan: Jika setelah dibersihkan datanya habis
    if df.empty:
        return {"error": f"Data harga saham {ticker} tidak valid atau sedang tidak tersedia."}

    # --- FITUR BARU: Ambil Harga LIVE (Memaksa tembus cache lelet Yahoo) ---
    try:
        # Trik: Paksa ambil data pergerakan 15 menitan hari ini
        df_live = yf.download(ticker, period="1d", interval="15m", progress=False)
        
        if not df_live.empty:
            if isinstance(df_live.columns, pd.MultiIndex):
                df_live.columns = df_live.columns.droplevel(1)
            if 'Close' in df_live.columns:
                df_live = df_live.dropna(subset=['Close'])
            
            # Ambil harga paling ujung dari data intraday
            current_price = float(df_live['Close'].iloc[-1])
            hist_last = float(df['Close'].iloc[-1]) # Data harian terakhir
            
            # Jika harga live beda dengan data harian terakhir, 
            # berarti Yahoo harian telat 1 hari. Kita jadikan data harian terakhir sbg "kemarin"
            if current_price != hist_last:
                prev_close_ui = hist_last
            else:
                prev_close_ui = float(df['Close'].iloc[-2]) if len(df) > 1 else hist_last
        else:
            # Fallback kalau gagal ambil data intraday
            current_price = float(ticker_obj.fast_info.lastPrice)
            prev_close_ui = float(ticker_obj.fast_info.previousClose)
            
    except Exception:
        # Fallback terakhir kalau Yahoo benar-benar down
        current_price = float(df['Close'].iloc[-1])
        prev_close_ui = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
    # ---------------------------------------------------------------

    # 1. Kalkulasi MACD (TETAP PAKAI DATA HISTORIS df['Close'])
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

    # 2. Kalkulasi RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. Bollinger Bands Squeeze (Deteksi Sideways)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA20'] + (df['STD20'] * 2)
    df['Lower_BB'] = df['SMA20'] - (df['STD20'] * 2)
    
    df['BB_Width'] = (df['Upper_BB'] - df['Lower_BB']) / df['SMA20']
    recent_bb_width = float(df['BB_Width'].iloc[-1])
    min_bb_width_60 = float(df['BB_Width'].tail(60).min())
    
    is_squeeze = recent_bb_width <= (min_bb_width_60 * 1.15)

    # 4. Kalkulasi S&R
    n = 5 if len(df) < 60 else 7
    df['Local_Max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]['Close']
    df['Local_Min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]['Close']

    resistances = sorted(df['Local_Max'].dropna().unique())
    supports = sorted(df['Local_Min'].dropna().unique(), reverse=True)

    # recent_close ini khusus buat ngitung analisa, biarkan dari data historis df
    recent_close = float(df['Close'].iloc[-1])
    recent_rsi = float(df['RSI'].iloc[-1])

    hist_min = float(df['Close'].min())
    idx_floor = 50.0 if hist_min >= 50.0 else 1.0 

    valid_supports = [s for s in supports if s < recent_close]
    filtered_supports = []
    for s in valid_supports:
        if not filtered_supports:
            filtered_supports.append(s)
        else:
            if (filtered_supports[-1] - s) / filtered_supports[-1] >= 0.025: 
                filtered_supports.append(s)

    support_1 = max(float(filtered_supports[0]), idx_floor) if len(filtered_supports) > 0 else hist_min
    if len(filtered_supports) > 1:
        support_2 = max(float(filtered_supports[1]), idx_floor)
    else:
        support_2 = hist_min if hist_min < support_1 else max(support_1 * 0.9, idx_floor)

    avg_range = (df['High'].tail(14) - df['Low'].tail(14)).mean()
    potensi_pantul = support_1 - avg_range
    potensi_pantul = math.floor(potensi_pantul)

    if potensi_pantul <= idx_floor or (support_1 - potensi_pantul) < (support_1 * 0.01):
        potensi_pantul_text = None
    else:
        potensi_pantul_text = f"Potensi jarum/pantulan ke Rp {potensi_pantul:,.0f}"

    valid_resistances = sorted([r for r in resistances if r > recent_close])
    resis_1 = float(valid_resistances[0]) if len(valid_resistances) > 0 else float(df['Close'].max())
    resis_2 = float(valid_resistances[1]) if len(valid_resistances) > 1 else resis_1 * 1.05

    # 5. LOGIKA KESIMPULAN (Berpatokan pada recent_close analisa)
    if is_squeeze:
        status_text = "AWAS VOLATILITAS! Saham sedang jenuh sideways (BB Squeeze). Energi sedang dikumpulkan. Bersiap untuk Breakout besar dalam waktu dekat!"
        status_color = "text-purple-400"
        strategi_text = "Pantau ketat volume. Jika harga tembus Resistance 1 dengan volume tinggi, ikut Beli. Jika jebol Support 1, segera hindari."
    else:
        if cross_type == "Golden Cross" and cross_days <= 2:
            status_text = "SANGAT BAGUS. Baru saja Golden Cross. Boleh mulai masuk (Buy on Breakout)."
            status_color = "text-green-400"
        elif cross_type == "Golden Cross" and cross_days > 2 and cross_days <= 5:
            status_text = "HATI-HATI. Tren masih naik tapi rawan profit taking. Cek apakah harga mendekati Resistance 1."
            status_color = "text-yellow-400"
        elif cross_type == "Golden Cross" and cross_days > 5:
            status_text = "TERLAMBAT. Sudah lama naik. Lebih baik tunggu koreksi ke Support 1 atau tunggu sinyal MACD baru."
            status_color = "text-orange-400"
        else:
            status_text = "BAHAYA. Sedang fase koreksi. Tunggu harga memantul di Support 1 atau Support 2 sebelum beli."
            status_color = "text-red-400"

        if recent_close < support_1 * 1.02 and recent_rsi < 40:
            strategi_text = "Harga sedang dekat Support 1 dengan RSI rendah. Boleh spekulasi beli dengan target jual di Resis 1."
        elif recent_close > resis_1 * 0.98:
            strategi_text = "Harga hampir menabrak Resistance 1. Rawan koreksi turun. Tahan dulu niat beli."
        else:
            strategi_text = "Harga berada di area tengah. Wait & See."

    tv_symbol = ticker.replace('.JK', '')
    if tv_symbol == ticker:
        tv_market = tv_symbol
    else:
        tv_market = f"IDX:{tv_symbol}"

    df_chart = df.tail(100)
    dates = df_chart.index.strftime('%Y-%m-%d').tolist()
    closes = [float(x) for x in df_chart['Close'].tolist()]

    return {
        "ticker": ticker.upper(),
        "tv_symbol": tv_market,
        "close": recent_close, # Ini EOD untuk jaga-jaga
        "current_price": current_price, # INI YANG TAMPIL DI WEB
        "prev_close_ui": prev_close_ui, # INI BUAT NGITUNG PERSEN DI WEB
        "rsi": recent_rsi,
        "cross_type": cross_type,
        "cross_days": cross_days,
        "is_squeeze": is_squeeze, 
        "r1": resis_1, "r2": resis_2,
        "s1": support_1, "s2": support_2,
        "potensi_pantul": potensi_pantul_text,
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