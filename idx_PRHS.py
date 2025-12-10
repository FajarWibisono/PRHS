import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime
warnings.filterwarnings("ignore")

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Calculator PRHS - Perkiraan Rentang Harga Saham",
    page_icon="📈",
    layout="wide"
)

# --- Konstanta ---
DEFAULT_PER = 15.0  # Default PER Industri
DEFAULT_PBV = 2.0   # Default PBV Industri
DEFAULT_BETA = 1.0  # Default Beta
DEFAULT_RISK_FREE_RATE = 0.06  # Default Risk-Free Rate (6%)
DEFAULT_MARKET_RISK_PREMIUM = 0.05  # Default Market Risk Premium (5%)
DEFAULT_COST_OF_DEBT = 0.08  # Default Cost of Debt (8%)
DEFAULT_TAX_RATE = 0.22  # Default Tax Rate (22%)
DEFAULT_TERMINAL_GROWTH_RATE = 0.03  # Default Terminal Growth Rate (3%)

# --- Fungsi Bantuan ---
def format_currency(value):
    """Format nilai menjadi format mata uang Rupiah"""
    if pd.isna(value) or value == 0:
        return "N/A"
    return f"Rp {value:,.0f}"

def format_percentage(value):
    """Format nilai menjadi format persentase"""
    if pd.isna(value) or value == 0:
        return "N/A"
    return f"{value:.2f}%"

def validate_input(value, min_value=0, default_value=0):
    """Validasi input dan kembalikan nilai default jika tidak valid"""
    if pd.isna(value) or value < min_value:
        return default_value
    return value

# --- Fungsi Pengambilan Data ---
def get_stock_data(ticker_symbol):
    """
    Ambil data saham dari yfinance dengan fokus pada akurasi data EPS dan PER historis
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        # Data dasar
        current_price = validate_input(info.get('currentPrice', 0))
        eps_trailing = validate_input(info.get('trailingEps', 0))
        eps_forward = validate_input(info.get('forwardEps', 0))
        roe = validate_input(info.get('returnOnEquity', 0), 0, 0) * 100  # dalam persen
        opm = validate_input(info.get('operatingMargins', 0), 0, 0) * 100  # dalam persen
        npm = validate_input(info.get('profitMargins', 0), 0, 0) * 100  # dalam persen
        # Data untuk perhitungan nilai intrinsik
        dividend_yield = validate_input(info.get('dividendYield', 0), 0, 0) * 100  # dalam persen
        dividend_rate = validate_input(info.get('dividendRate', 0), 0, 0)
        beta = validate_input(info.get('beta', DEFAULT_BETA), 0, DEFAULT_BETA)
        book_value = validate_input(info.get('bookValue', 0), 0, 0)
        shares_outstanding = validate_input(info.get('sharesOutstanding', 0), 0, 1)
        debt_to_equity = validate_input(info.get('debtToEquity', 0.5), 0, 0.5)
        revenue_growth = validate_input(info.get('revenueGrowth', 0), 0, 0)

        # Ambil data harga historis (5 tahun terakhir)
        hist = ticker.history(period="5y", interval="1d")
        current_year = str(pd.Timestamp.now().year)

        # Inisialisasi data harga tahunan
        price_high_low = {}

        # Dapatkan data harga tinggi dan rendah untuk setiap tahun
        for year in range(2020, 2025):  # Contoh: 2020-2024
            year_str = str(year)
            year_data = hist[hist.index.year == year]
            if not year_data.empty:
                price_high_low[year_str] = {
                    'high': validate_input(year_data['High'].max(), 0, 0),
                    'low': validate_input(year_data['Low'].min(), 0, 0)
                }
            else:
                price_high_low[year_str] = {'high': 0, 'low': 0}

        # Untuk tahun berjalan
        current_year_data = hist[hist.index.year == int(current_year)]
        if not current_year_data.empty:
            price_high_low[current_year] = {
                'high': validate_input(current_year_data['High'].max(), 0, current_price),
                'low': validate_input(current_year_data['Low'].min(), 0, current_price)
            }
        else:
            price_high_low[current_year] = {'high': current_price, 'low': current_price}

        # Ambil data EPS historis dengan pendekatan yang lebih akurat
        eps_history = {}
        # Coba ambil data dari laporan keuangan tahunan
        try:
            financials = ticker.financials
            if not financials.empty and 'Diluted EPS' in financials.index:
                eps_row = financials.loc['Diluted EPS']
                for year in eps_row.index:
                    year_str = str(year.year)
                    eps_history[year_str] = validate_input(eps_row[year], 0, 0)
        except Exception as e:
            st.warning(f"⚠️ Gagal mengambil data EPS dari laporan tahunan: {str(e)}")

        # Jika tidak ada atau tidak lengkap, coba dari laporan keuangan kuartalan
        if len(eps_history) < 5:
            try:
                quarterly_financials = ticker.quarterly_financials
                if 'Diluted EPS' in quarterly_financials.index:
                    eps_quarterly = quarterly_financials.loc['Diluted EPS']
                    eps_annual = {}
                    # Gabungkan data kuartalan menjadi tahunan
                    for date in eps_quarterly.index:
                        year = str(date.year)
                        if year not in eps_annual:
                            eps_annual[year] = 0
                        eps_annual[year] += validate_input(eps_quarterly[date], 0, 0)
                    # Tambahkan data EPS tahunan ke history
                    for year, eps in eps_annual.items():
                        if year not in eps_history or eps_history[year] == 0:
                            eps_history[year] = eps
            except Exception as e:
                st.warning(f"⚠️ Gagal mengambil data EPS dari laporan kuartalan: {str(e)}")

        # Jika masih kurang, coba dari data yfinance.info
        if len(eps_history) < 5:
            try:
                # Ambil data EPS dari info
                if eps_trailing > 0:
                    current_year = str(pd.Timestamp.now().year)
                    eps_history[current_year] = eps_trailing
            except Exception as e:
                st.warning(f"⚠️ Gagal mengambil data EPS dari info: {str(e)}")

        # Pastikan kita punya data untuk tahun-tahun terakhir
        for year in range(2020, 2025):  # Contoh: 2020-2024
            year_str = str(year)
            if year_str not in eps_history:
                eps_history[year_str] = 0

        # Hitung PER historis dengan lebih akurat
        per_history = {}
        for year in price_high_low:
            if year in eps_history and eps_history[year] > 0:
                per_history[year] = {
                    'high': price_high_low[year]['high'] / eps_history[year],
                    'low': price_high_low[year]['low'] / eps_history[year]
                }
            else:
                per_history[year] = {'high': 0, 'low': 0}

        # Ambil EPS untuk tahun ini - gunakan forward EPS jika tersedia
        current_year = str(pd.Timestamp.now().year)
        eps_current = eps_forward if eps_forward > 0 else eps_trailing

        # Pastikan EPS tahun ada di history
        if current_year not in eps_history or eps_history[current_year] == 0:
            eps_history[current_year] = eps_current

        # Ambil data teknikal
        rsi, macd, signal, ma_20, ma_50, ma_200 = 0, 0, 0, 0, 0, 0
        try:
            hist_technical = ticker.history(period="1y", interval="1d")
            # Hitung RSI
            delta = hist_technical['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi = validate_input(rsi.iloc[-1], 0, 0)

            # Hitung MACD
            exp1 = hist_technical['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist_technical['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd = validate_input(macd.iloc[-1], 0, 0)
            signal = validate_input(signal.iloc[-1], 0, 0)

            # Hitung Moving Average
            ma_20 = validate_input(hist_technical['Close'].rolling(window=20).mean().iloc[-1], 0, 0)
            ma_50 = validate_input(hist_technical['Close'].rolling(window=50).mean().iloc[-1], 0, 0)
            ma_200 = validate_input(hist_technical['Close'].rolling(window=200).mean().iloc[-1], 0, 0)
        except Exception as e:
            st.warning(f"⚠️ Gagal mengambil data teknikal: {str(e)}")

        # Ambil data untuk DCF
        free_cash_flow = None
        try:
            free_cash_flow = ticker.cashflow.loc['Free Cash Flow'] if 'Free Cash Flow' in ticker.cashflow.index else None
        except Exception as e:
            st.warning(f"⚠️ Gagal mengambil data Free Cash Flow: {str(e)}")

        return {
            'ticker': ticker_symbol,
            'current_price': current_price,
            'eps_trailing': eps_trailing,
            'eps_forward': eps_forward,
            'eps_current': eps_current,
            'eps_history': eps_history,
            'price_high_low': price_high_low,
            'per_history': per_history,
            'roe': roe,
            'opm': opm,
            'npm': npm,
            'info': info,
            'rsi': rsi,
            'macd': macd,
            'signal': signal,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'ma_200': ma_200,
            'free_cash_flow': free_cash_flow,
            'revenue_growth': revenue_growth,
            'beta': beta,
            'dividend_yield': dividend_yield,
            'dividend_rate': dividend_rate,
            'book_value': book_value,
            'shares_outstanding': shares_outstanding,
            'debt_to_equity': debt_to_equity
        }
    except Exception as e:
        st.error(f"❌ Error mengambil data saham: {str(e)}")
        return None

# --- Fungsi Perhitungan Estimasi Harga ---
def calculate_price_estimates(stock_data, per_industry, pbv_industry):
    """
    Hitung estimasi harga berdasarkan PER dan PBV dengan perhitungan yang lebih akurat
    """
    eps_current = stock_data['eps_current']

    # Dapatkan data PER historis
    per_history = stock_data['per_history']

    # Dapatkan nilai-nilai PER yang diperlukan
    per_low_values = [v['low'] for v in per_history.values() if v['low'] > 0]
    per_high_values = [v['high'] for v in per_history.values() if v['high'] > 0]

    # Skenario RENDAH:
    # Harga rendah dari PER LOW terendah (dari 5 tahun terakhir)
    # Harga tinggi dari PER LOW tertinggi (dari 5 tahun terakhir)
    per_low_min = min(per_low_values) if per_low_values else 0
    per_low_max = max(per_low_values) if per_low_values else 0
    rendah_low = per_low_min * eps_current if per_low_min > 0 else 0
    rendah_high = per_low_max * eps_current if per_low_max > 0 else 0

    # Skenario OPTIMIS:
    # Harga rendah dari PER HIGH terendah (dari 5 tahun terakhir)
    # Harga tinggi dari PER HIGH tertinggi (dari 5 tahun terakhir)
    per_high_min = min(per_high_values) if per_high_values else 0
    per_high_max = max(per_high_values) if per_high_values else 0
    optimis_low = per_high_min * eps_current if per_high_min > 0 else 0
    optimis_high = per_high_max * eps_current if per_high_max > 0 else 0

    # Perhitungan MoS (Margin of Safety)
    # MoS dihitung berdasarkan selisih antara harga estimasi rendah tinggi dan harga saat ini
    mos = ((rendah_high - stock_data['current_price']) / rendah_high * 100) if rendah_high > 0 else 0

    # Perhitungan dari PER Industri
    price_from_per_industry = per_industry * eps_current if per_industry > 0 else 0

    # Perhitungan dari PBV Industri
    price_from_pbv_industry = pbv_industry * stock_data['book_value'] if pbv_industry > 0 and stock_data['book_value'] > 0 else 0

    # Perhitungan DCF
    dcf_value = 0
    if stock_data['free_cash_flow'] is not None and not stock_data['free_cash_flow'].empty:
        try:
            # Ambil 5 tahun terakhir FCF
            fcf_values = stock_data['free_cash_flow'].iloc[:5]
            if len(fcf_values) > 0:
                # Proyeksi FCF 5 tahun ke depan dengan pertumbuhan pendapatan
                projected_fcf = []
                last_fcf = fcf_values.iloc[0]
                for year in range(5):
                    last_fcf = last_fcf * (1 + stock_data['revenue_growth'])
                    projected_fcf.append(last_fcf)

                # Hitung terminal value
                terminal_value = projected_fcf[-1] * (1 + DEFAULT_TERMINAL_GROWTH_RATE) / (DEFAULT_RISK_FREE_RATE + stock_data['beta'] * DEFAULT_MARKET_RISK_PREMIUM - DEFAULT_TERMINAL_GROWTH_RATE)

                # Hitung DCF
                dcf_value = 0
                for year, fcf in enumerate(projected_fcf):
                    dcf_value += fcf / ((1 + DEFAULT_RISK_FREE_RATE + stock_data['beta'] * DEFAULT_MARKET_RISK_PREMIUM) ** (year + 1))
                dcf_value += terminal_value / ((1 + DEFAULT_RISK_FREE_RATE + stock_data['beta'] * DEFAULT_MARKET_RISK_PREMIUM) ** 5)

                # Bagi dengan jumlah saham beredar
                if stock_data['shares_outstanding'] > 0:
                    dcf_value = dcf_value / stock_data['shares_outstanding']
        except Exception as e:
            st.warning(f"⚠️ Gagal menghitung DCF: {str(e)}")
            dcf_value = 0

    # Perhitungan Gordon Growth Model (lebih robust)
    ggm_value = 0
    if stock_data['dividend_rate'] > 0:
        try:
            # Gordon Growth Model: P = D1 / (r - g)
            # D1 = D0 * (1 + g)
            d1 = stock_data['dividend_rate'] * (1 + stock_data['revenue_growth'])
            # Cost of Equity menggunakan CAPM
            cost_of_equity = DEFAULT_RISK_FREE_RATE + stock_data['beta'] * DEFAULT_MARKET_RISK_PREMIUM
            ggm_value = d1 / (cost_of_equity - stock_data['revenue_growth'])
        except Exception as e:
            st.warning(f"⚠️ Gagal menghitung Gordon Growth Model: {str(e)}")
            ggm_value = 0

    # Perhitungan Asset-Based Valuation (lebih robust)
    asset_value = stock_data['book_value'] * 1.2  # Asumsi premium 20% atas nilai buku

    # Perhitungan Residual Income Model
    rim_value = 0
    try:
        # Residual Income Model: P = B0 + Σ (RI_t / (1+r)^t)
        # RI_t = (ROE_t - r) * B_{t-1}
        book_value = stock_data['book_value']
        cost_of_equity = DEFAULT_RISK_FREE_RATE + stock_data['beta'] * DEFAULT_MARKET_RISK_PREMIUM
        roe = stock_data['roe'] / 100  # Konversi dari persen ke desimal
        if book_value > 0 and roe > 0:
            rim_value = book_value
            for year in range(5):
                residual_income = (roe - cost_of_equity) * book_value
                rim_value += residual_income / ((1 + cost_of_equity) ** (year + 1))
                book_value *= (1 + stock_data['revenue_growth'])
    except Exception as e:
        st.warning(f"⚠️ Gagal menghitung Residual Income Model: {str(e)}")
        rim_value = 0

    return {
        'rendah_low': rendah_low,
        'rendah_high': rendah_high,
        'optimis_low': optimis_low,
        'optimis_high': optimis_high,
        'mos': mos,
        'price_from_per_industry': price_from_per_industry,
        'price_from_pbv_industry': price_from_pbv_industry,
        'dcf_value': dcf_value,
        'ggm_value': ggm_value,
        'asset_value': asset_value,
        'rim_value': rim_value,
        'per_low_min': per_low_min,
        'per_low_max': per_low_max,
        'per_high_min': per_high_min,
        'per_high_max': per_high_max
    }

# --- Fungsi Tampilan ---
def display_results_table(stock_data, estimates, per_industry, pbv_industry):
    """
    Tampilkan hasil dalam bentuk tabel dengan tata letak yang baru
    """
    # Header dengan nama saham
    st.markdown(f"## {stock_data['ticker']}")

    # Harga saat ini dan EPS saat ini dengan ukuran lebih besar
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Harga Saat Ini")
        st.markdown(f"### {format_currency(stock_data['current_price'])}")
    with col2:
        st.markdown(f"### EPS Saat Ini")
        st.markdown(f"### {stock_data['eps_current']:,.2f}")

    # Tabel pertama: Perkiraan rentang harga skenario RENDAH dan OPTIMIS
    st.markdown("### Perkiraan Rentang Harga")
    price_data = {
        '': ['Perkiraan Skenario RENDAH', 'Perkiraan Skenario OPTIMIS'],
        'Harga Rendah': [
            format_currency(estimates['rendah_low']),
            format_currency(estimates['optimis_low'])
        ],
        'Harga Tinggi': [
            format_currency(estimates['rendah_high']),
            format_currency(estimates['optimis_high'])
        ]
    }
    price_df = pd.DataFrame(price_data)
    st.dataframe(price_df, width='stretch', hide_index=True)

    # Tabel kedua: Perkiraan harga dari PER dan PBV sektor industri (dipindahkan ke bawah tabel perkiraan rentang harga)
    st.markdown("### Perkiraan Harga dari Sektor Industri")
    industry_data = {
        '': ['Perkiraan Harga dari PER Sektor Industri', 'Perkiraan Harga dari PBV Sektor Industri'],
        'Nilai': [
            format_currency(estimates['price_from_per_industry']),
            format_currency(estimates['price_from_pbv_industry'])
        ]
    }
    industry_df = pd.DataFrame(industry_data)
    st.dataframe(industry_df, width='stretch', hide_index=True)

    # Tabel data historis EPS dan harga (dipindahkan ke bawah tabel perkiraan rentang harga)
    st.subheader("📚 Data Historis")
    col_eps, col_price = st.columns(2)
    with col_eps:
        st.write("**EPS 5 Tahun Terakhir**")
        # Urutkan tahun dari terbaru ke terlama
        eps_df = pd.DataFrame.from_dict(stock_data['eps_history'], orient='index', columns=['EPS'])
        eps_df.index.name = 'Tahun'
        eps_df = eps_df.sort_index(ascending=False)
        st.dataframe(eps_df.style.format({'EPS': '{:,.2f}'}), width='stretch', hide_index=False)
    with col_price:
        st.write("**Harga High & Low 5 Tahun Terakhir**")
        # Urutkan tahun dari terbaru ke terlama
        price_df = pd.DataFrame.from_dict(stock_data['price_high_low'], orient='index')
        price_df.index.name = 'Tahun'
        price_df = price_df.sort_index(ascending=False)
        st.dataframe(price_df.style.format({'high': 'Rp {:,.0f}', 'low': 'Rp {:,.0f}'}), width='stretch', hide_index=False)

    # Tampilkan PER historis (dipindahkan ke bawah tabel perkiraan rentang harga)
    st.subheader("📊 PER Historis")
    # Urutkan tahun dari terbaru ke terlama
    per_df = pd.DataFrame.from_dict(stock_data['per_history'], orient='index')
    per_df.index.name = 'Tahun'
    per_df = per_df.sort_index(ascending=False)
    st.dataframe(per_df.style.format({'high': '{:.2f}', 'low': '{:.2f}'}), width='stretch', hide_index=False)


    # Tabel profitabilitas
    st.markdown("### Profitabilitas")
    profit_data = {
        '': ['ROE', 'OPM', 'NPM'],
        'Nilai': [
            format_percentage(stock_data['roe']),
            format_percentage(stock_data['opm']),
            format_percentage(stock_data['npm'])
        ]
    }
    profit_df = pd.DataFrame(profit_data)
    st.dataframe(profit_df, width='stretch', hide_index=True)

    # Penjelasan ROE Adjusted (dipindahkan ke bawah tabel profitabilitas)
    st.markdown("""
    **Catatan tentang ROE Adjusted:**
    Hitunglah ROE adjusted untuk memberikan gambaran fundamental yang lebih baik.
    Rumus ROE Adjusted = (Laba Bersih - Item Tidak Berulang) / Ekuitas Rata-rata
    ROE Adjusted menghilangkan pengaruh item tidak berulang (seperti penjualan aset sekali kali atau
    restrukturisasi) untuk memberikan gambaran yang lebih akurat tentang profitabilitas inti perusahaan.
    """)

    # Tabel nilai intrinsik (dipindahkan ke bawah tabel profitabilitas)
    st.markdown("### Nilai Intrinsik")
    intrinsic_data = {
        '': ['DCF (Discounted Cash Flow)', 'Gordon Growth Model', 'Asset-Based Valuation', 'Residual Income Model'],
        'Nilai': [
            format_currency(estimates['dcf_value']),
            format_currency(estimates['ggm_value']),
            format_currency(estimates['asset_value']),
            format_currency(estimates['rim_value'])
        ]
    }
    intrinsic_df = pd.DataFrame(intrinsic_data)
    st.dataframe(intrinsic_df, width='stretch', hide_index=True)

    # Tabel indikator teknikal (dipindahkan ke bawah tabel nilai intrinsik)
    st.markdown("### Indikator Teknikal")
    technical_data = {
        '': ['RSI', 'MACD', 'MA 20', 'MA 50', 'MA 200'],
        'Nilai': [
            f"{stock_data['rsi']:.2f}" if stock_data['rsi'] > 0 else "N/A",
            f"{stock_data['macd']:.2f}" if stock_data['macd'] != 0 else "N/A",
            format_currency(stock_data['ma_20']),
            format_currency(stock_data['ma_50']),
            format_currency(stock_data['ma_200'])
        ]
    }
    technical_df = pd.DataFrame(technical_data)
    st.dataframe(technical_df, width='stretch', hide_index=True)

# --- Fungsi Export ke PDF ---
def export_to_pdf(stock_data, estimates, per_industry, pbv_industry):
    """
    Export hasil analisis ke PDF, mencerminkan tampilan web
    """
    # Buat buffer untuk PDF
    buffer = BytesIO()
    # Buat dokumen PDF
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    story = []

    # Style
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center
    )

    # Tambahkan judul
    story.append(Paragraph(f"Analisis Saham {stock_data['ticker']}", title_style))
    story.append(Spacer(1, 12))

    # Tambahkan tanggal
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1  # Center
    )
    story.append(Paragraph(f"Dibuat pada: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", date_style))
    story.append(Spacer(1, 12))

    # Tambahkan harga dan EPS
    data = [
        ['Harga Saat Ini', format_currency(stock_data['current_price'])],
        ['EPS Saat Ini', f"{stock_data['eps_current']:,.2f}"]
    ]
    table = Table(data, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Tambahkan tabel perkiraan harga
    story.append(Paragraph("Perkiraan Rentang Harga", styles['Heading2']))
    story.append(Spacer(1, 6))
    price_data = [
        ['Skenario', 'Harga Rendah', 'Harga Tinggi'],
        ['RENDAH', format_currency(estimates['rendah_low']), format_currency(estimates['rendah_high'])],
        ['OPTIMIS', format_currency(estimates['optimis_low']), format_currency(estimates['optimis_high'])]
    ]
    price_table = Table(price_data, colWidths=[2*inch, 2*inch, 2*inch])
    price_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(price_table)
    story.append(Spacer(1, 12))

    # Tambahkan tabel perkiraan harga dari industri (dipindahkan ke sini)
    story.append(Paragraph("Perkiraan Harga dari Sektor Industri", styles['Heading2']))
    story.append(Spacer(1, 6))
    industry_data = [
        ['Metode', 'Nilai'],
        ['PER Sektor Industri', format_currency(estimates['price_from_per_industry'])],
        ['PBV Sektor Industri', format_currency(estimates['price_from_pbv_industry'])]
    ]
    industry_table = Table(industry_data, colWidths=[2.5*inch, 2.5*inch])
    industry_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(industry_table)
    story.append(Spacer(1, 12))

    # Tambahkan tabel data historis EPS
    story.append(Paragraph("Data Historis EPS", styles['Heading2']))
    story.append(Spacer(1, 6))
    eps_hist_data = [['Tahun', 'EPS']]
    # Urutkan tahun dari terbaru ke terlama
    sorted_years = sorted(stock_data['eps_history'].keys(), reverse=True)
    for year in sorted_years:
        eps_hist_data.append([year, f"{stock_data['eps_history'][year]:,.2f}"])
    eps_hist_table = Table(eps_hist_data, colWidths=[1.5*inch, 1.5*inch])
    eps_hist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(eps_hist_table)
    story.append(Spacer(1, 12))

    # Tambahkan tabel data historis harga
    story.append(Paragraph("Data Historis Harga", styles['Heading2']))
    story.append(Spacer(1, 6))
    price_hist_data = [['Tahun', 'Harga Tertinggi', 'Harga Terendah']]
    # Urutkan tahun dari terbaru ke terlama
    sorted_years = sorted(stock_data['price_high_low'].keys(), reverse=True)
    for year in sorted_years:
        price_hist_data.append([year, format_currency(stock_data['price_high_low'][year]['high']), format_currency(stock_data['price_high_low'][year]['low'])])
    price_hist_table = Table(price_hist_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    price_hist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(price_hist_table)
    story.append(Spacer(1, 12))

    # Tambahkan tabel data historis PER
    story.append(Paragraph("Data Historis PER", styles['Heading2']))
    story.append(Spacer(1, 6))
    per_hist_data = [['Tahun', 'PER Tertinggi', 'PER Terendah']]
    # Urutkan tahun dari terbaru ke terlama
    sorted_years = sorted(stock_data['per_history'].keys(), reverse=True)
    for year in sorted_years:
        per_hist_data.append([year, f"{stock_data['per_history'][year]['high']:.2f}", f"{stock_data['per_history'][year]['low']:.2f}"])
    per_hist_table = Table(per_hist_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    per_hist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(per_hist_table)
    story.append(Spacer(1, 12))

    # Tambahkan tabel profitabilitas
    story.append(Paragraph("Profitabilitas", styles['Heading2']))
    story.append(Spacer(1, 6))
    profit_data = [
        ['Metrik', 'Nilai'],
        ['ROE', format_percentage(stock_data['roe'])],
        ['OPM', format_percentage(stock_data['opm'])],
        ['NPM', format_percentage(stock_data['npm'])]
    ]
    profit_table = Table(profit_data, colWidths=[2.5*inch, 2.5*inch])
    profit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(profit_table)
    story.append(Spacer(1, 12))

    # Tambahkan catatan ROE Adjusted
    story.append(Paragraph("Catatan tentang ROE Adjusted:", styles['Heading3']))
    story.append(Spacer(1, 6))
    roe_text = """
    Hitunglah ROE adjusted untuk memberikan gambaran fundamental yang lebih baik.
    Rumus ROE Adjusted = (Laba Bersih - Item Tidak Berulang) / Ekuitas Rata-rata
    ROE Adjusted menghilangkan pengaruh item tidak berulang (seperti penjualan aset sekali kali atau
    restrukturisasi) untuk memberikan gambaran yang lebih akurat tentang profitabilitas inti perusahaan.
    """
    story.append(Paragraph(roe_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Tambahkan tabel nilai intrinsik
    story.append(Paragraph("Nilai Intrinsik", styles['Heading2']))
    story.append(Spacer(1, 6))
    intrinsic_data = [
        ['Metode', 'Nilai'],
        ['DCF (Discounted Cash Flow)', format_currency(estimates['dcf_value'])],
        ['Gordon Growth Model', format_currency(estimates['ggm_value'])],
        ['Asset-Based Valuation', format_currency(estimates['asset_value'])],
        ['Residual Income Model', format_currency(estimates['rim_value'])]
    ]
    intrinsic_table = Table(intrinsic_data, colWidths=[3*inch, 3*inch])
    intrinsic_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(intrinsic_table)
    story.append(Spacer(1, 12))

    # Tambahkan tabel indikator teknikal
    story.append(Paragraph("Indikator Teknikal", styles['Heading2']))
    story.append(Spacer(1, 6))
    technical_data = [
        ['Indikator', 'Nilai'],
        ['RSI', f"{stock_data['rsi']:.2f}" if stock_data['rsi'] > 0 else "N/A"],
        ['MACD', f"{stock_data['macd']:.2f}" if stock_data['macd'] != 0 else "N/A"],
        ['MA 20', format_currency(stock_data['ma_20'])],
        ['MA 50', format_currency(stock_data['ma_50'])],
        ['MA 200', format_currency(stock_data['ma_200'])]
    ]
    technical_table = Table(technical_data, colWidths=[2.5*inch, 2.5*inch])
    technical_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(technical_table)
    story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)

    # Get PDF value
    pdf_value = buffer.getvalue()
    buffer.close()

    # Encode to base64
    b64 = base64.b64encode(pdf_value).decode()

    # Create download link
    href = f'<a href="data:application/pdf;base64,{b64}" download="analisis_{stock_data["ticker"]}.pdf">Download PDF</a>'
    return href

# --- Fungsi Grafik ---
def plot_risk_reward_chart(rendah_high, rendah_low, optimis_high, optimis_low, current_price):
    """
    Buat bar chart horizontal untuk Risk & Reward
    """
    # Data untuk skenario rendah
    rendah_data = {
        'Kategori': ['Harga Tinggi (Rendah)', 'Harga Saat Ini', 'Harga Rendah (Rendah)'],
        'Harga': [rendah_high, current_price, rendah_low],
        'Warna': ['#2ecc71', '#3498db', '#e74c3c']
    }

    # Data untuk skenario optimis
    optimis_data = {
        'Kategori': ['Harga Tinggi (Optimis)', 'Harga Saat Ini', 'Harga Rendah (Optimis)'],
        'Harga': [optimis_high, current_price, optimis_low],
        'Warna': ['#2ecc71', '#3498db', '#e74c3c']
    }

    fig_rendah = px.bar(rendah_data, y='Kategori', x='Harga', color='Warna',
                        title='Risk & Reward - Skenario Rendah',
                        labels={'Harga': 'Harga (Rp)'}, height=300)
    fig_rendah.update_layout(showlegend=False, xaxis_tickformat=',.0f')
    fig_rendah.update_traces(texttemplate='%{x:.0f}', textposition='outside')

    fig_optimis = px.bar(optimis_data, y='Kategori', x='Harga', color='Warna',
                         title='Risk & Reward - Skenario Optimis',
                         labels={'Harga': 'Harga (Rp)'}, height=300)
    fig_optimis.update_layout(showlegend=False, xaxis_tickformat=',.0f')
    fig_optimis.update_traces(texttemplate='%{x:.0f}', textposition='outside')

    return fig_rendah, fig_optimis

def plot_technical_indicators(ticker_symbol):
    """
    Tampilkan grafik indikator teknikal
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y", interval="1d")

        # Buat subplot
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Harga dan Moving Average', 'RSI', 'MACD'),
            row_heights=[0.5, 0.25, 0.25]
        )

        # Grafik harga dan moving average
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['Close'], name='Harga', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['Close'].rolling(window=20).mean(), name='MA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['Close'].rolling(window=50).mean(), name='MA 50', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['Close'].rolling(window=200).mean(), name='MA 200', line=dict(color='red')),
            row=1, col=1
        )

        # Grafik RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        fig.add_trace(
            go.Scatter(x=hist.index, y=rsi, name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        # Tambahkan garis batas RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Grafik MACD
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        fig.add_trace(
            go.Scatter(x=hist.index, y=macd, name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=signal, name='Signal', line=dict(color='red')),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'Indikator Teknikal {ticker_symbol}',
            xaxis3_title='Tanggal',
            height=800
        )
        return fig
    except Exception as e:
        st.warning(f"⚠️ Gagal membuat grafik indikator teknikal: {str(e)}")
        return None

# --- Main App ---
st.title("📈 Calculator PRHS - Perkiraan Rentang Harga Saham")
st.markdown("Navigasi dinamika pasar saham menjadi lebih mudah dengan bantuan Humanis PRHS.")
st.markdown("Calculator ini membantu memberikan **PRAKIRAAN HARGA**. Masukkan ticker KODE SAHAM (XXXX) dan **PER Sektor Industri serta PBV Sektor Industri.**")
st.markdown(
    "Masukkan di pencarian KODE SAHAM lalu geser kekanan sampai menemukan **P/E Ratio X, dan Price to BV X.** "
    "Untuk referensi, kunjungi: [IDX Financial Data and Ratio](https://idx.co.id/id/data-pasar/laporan-statistik/digital-statistic/monthly/financial-report-and-ratio-of-listed-companies/financial-data-and-ratio?filter=eyJ5ZWFyIjoiMjAyNSIsIm1vbnRoIjoiMTEiLCJxdWFydGVyIjowLCJ0eXBlIjoibW9udGhseSJ9)"
)

# Input Form
with st.form("prhs_form"):
    st.subheader("📊 Data Saham")
    ticker_input = st.text_input("Nama Saham (contoh: ABMM)", help="Masukkan kode saham tanpa .JK, sistem akan otomatis menambahkan .JK")

    # Field default untuk PER dan PBV Industri
    col1, col2 = st.columns(2)
    with col1:
        per_industry = st.number_input("PER Sektor Industri", min_value=0.0, value=DEFAULT_PER, step=0.1, help="Masukkan rata-rata PER untuk sektor industri saham tersebut")
    with col2:
        pbv_industry = st.number_input("PBV Sektor Industri", min_value=0.0, value=DEFAULT_PBV, step=0.1, help="Masukkan rata-rata PBV untuk sektor industri saham tersebut")

    submit_button = st.form_submit_button("Hitung Perkiraan Harga")

if submit_button:
    if not ticker_input.strip():
        st.error("⚠️ Silakan masukkan nama saham.")
    else:
        ticker_symbol = f"{ticker_input.strip().upper()}.JK"
        with st.spinner(f"Mengambil data saham {ticker_symbol}..."):
            stock_data = get_stock_data(ticker_symbol)
            if stock_data is None:
                st.error("❌ Gagal mengambil data saham dari API. Pastikan kode saham benar dan terdaftar di IDX.")
            else:
                # Hitung estimasi harga
                estimates = calculate_price_estimates(stock_data, per_industry, pbv_industry)

                # Tampilkan hasil
                display_results_table(stock_data, estimates, per_industry, pbv_industry)

                # Tampilkan grafik indikator teknikal
                st.subheader("📊 Grafik Indikator Teknikal")
                fig_technical = plot_technical_indicators(ticker_symbol)
                if fig_technical:
                    st.plotly_chart(fig_technical, width='stretch')

                # Tampilkan bar chart Risk & Reward
                st.subheader("📉 Risk & Reward")
                fig_rendah, fig_optimis = plot_risk_reward_chart(
                    estimates['rendah_high'],
                    estimates['rendah_low'],
                    estimates['optimis_high'],
                    estimates['optimis_low'],
                    stock_data['current_price']
                )
                st.plotly_chart(fig_rendah, width='stretch')
                st.plotly_chart(fig_optimis, width='stretch')

                # Tambahkan tombol export PDF (dipindahkan ke bagian paling bawah)
                st.markdown("---")
                st.subheader("📄 Export Hasil Analisis")
                pdf_link = export_to_pdf(stock_data, estimates, per_industry, pbv_industry)
                st.markdown(pdf_link, unsafe_allow_html=True)
                st.success("✅ Perhitungan selesai!")

