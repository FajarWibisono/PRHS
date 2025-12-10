import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
DEFAULT_PER = 15.0
DEFAULT_PBV = 2.0
DEFAULT_BETA = 1.0
DEFAULT_RISK_FREE_RATE = 0.06
DEFAULT_MARKET_RISK_PREMIUM = 0.05
DEFAULT_TERMINAL_GROWTH_RATE = 0.03

# --- Fungsi Bantuan ---
def format_currency(value):
    if pd.isna(value) or value is None or value == 0:
        return "N/A"
    return f"Rp {value:,.0f}"

def format_percentage(value):
    if pd.isna(value) or value is None or value == 0:
        return "N/A"
    return f"{value:.2f}%"

def validate_input(value, min_value=0, default_value=0):
    if pd.isna(value) or value is None or value < min_value:
        return default_value
    return value

# --- Fungsi Pengambilan Data ---
def get_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        current_price = validate_input(info.get('currentPrice', 0))
        eps_trailing = validate_input(info.get('trailingEps', 0))
        eps_forward = validate_input(info.get('forwardEps', 0))
        roe = validate_input(info.get('returnOnEquity', 0), 0, 0) * 100
        opm = validate_input(info.get('operatingMargins', 0), 0, 0) * 100
        npm = validate_input(info.get('profitMargins', 0), 0, 0) * 100

        dividend_yield = validate_input(info.get('dividendYield', 0), 0, 0) * 100
        dividend_rate = validate_input(info.get('dividendRate', 0), 0, 0)
        beta = validate_input(info.get('beta', DEFAULT_BETA), 0, DEFAULT_BETA)
        book_value = validate_input(info.get('bookValue', 0), 0, 0)
        shares_outstanding = validate_input(info.get('sharesOutstanding', 0), 0, 1)
        debt_to_equity = validate_input(info.get('debtToEquity', 0.5), 0, 0.5)
        revenue_growth = validate_input(info.get('revenueGrowth', 0), 0, 0)

        hist = ticker.history(period="5y", interval="1d")
        current_year = str(pd.Timestamp.now().year)

        price_high_low = {}
        for year in range(2020, 2026):
            year_str = str(year)
            year_data = hist[hist.index.year == year]
            if not year_data.empty:
                price_high_low[year_str] = {
                    'high': validate_input(year_data['High'].max(), 0, 0),
                    'low': validate_input(year_data['Low'].min(), 0, 0)
                }
            else:
                price_high_low[year_str] = {'high': 0, 'low': 0}

        # Ambil EPS historis (lebih robust)
        eps_history = {}

        # Coba dari laporan tahunan
        try:
            financials = ticker.financials
            if not financials.empty:
                eps_candidates = ['Diluted EPS', 'Basic EPS', 'EPS']
                for candidate in eps_candidates:
                    if candidate in financials.index:
                        eps_row = financials.loc[candidate]
                        for date in eps_row.index:
                            year_str = str(date.year)
                            val = eps_row[date]
                            if pd.notna(val) and val != 0:
                                eps_history[year_str] = float(val)
                        break
        except Exception as e:
            st.warning(f"⚠️ Gagal ambil EPS dari laporan tahunan: {e}")

        # Coba dari laporan kuartalan
        if len(eps_history) < 5:
            try:
                qf = ticker.quarterly_financials
                eps_candidates = ['Diluted EPS', 'Basic EPS', 'EPS']
                for candidate in eps_candidates:
                    if candidate in qf.index:
                        eps_q = qf.loc[candidate]
                        annual_eps = {}
                        for date in eps_q.index:
                            yr = str(date.year)
                            val = eps_q[date]
                            if pd.notna(val):
                                annual_eps[yr] = annual_eps.get(yr, 0) + val
                        for yr, val in annual_eps.items():
                            if val > 0 and (yr not in eps_history or eps_history[yr] == 0):
                                eps_history[yr] = float(val)
                        break
            except Exception as e:
                st.warning(f"⚠️ Gagal ambil EPS dari laporan kuartalan: {e}")

        # Gunakan trailing EPS sebagai fallback
        if len(eps_history) < 5 and eps_trailing > 0:
            eps_history[current_year] = eps_trailing

        # Pastikan semua tahun 2020–2025 ada (jika tidak ada → None)
        for year in range(2020, 2026):
            year_str = str(year)
            if year_str not in eps_history:
                eps_history[year_str] = None

        # Hitung PER historis (hanya jika EPS dan harga valid)
        per_history = {}
        for year in price_high_low:
            high = price_high_low[year]['high']
            low = price_high_low[year]['low']
            eps = eps_history.get(year)

            per_high = high / eps if pd.notna(eps) and eps and high > 0 else None
            per_low = low / eps if pd.notna(eps) and eps and low > 0 else None

            per_history[year] = {'high': per_high, 'low': per_low}

        # EPS saat ini
        eps_current = eps_forward if eps_forward > 0 else eps_trailing
        if current_year not in eps_history or eps_history[current_year] is None:
            eps_history[current_year] = eps_current

        # Indikator teknikal
        rsi = macd = signal = ma_20 = ma_50 = ma_200 = 0
        try:
            hist_tech = ticker.history(period="1y", interval="1d")
            close = hist_tech['Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 0

            exp12 = close.ewm(span=12).mean()
            exp26 = close.ewm(span=26).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9).mean()
            macd = float(macd_line.iloc[-1])
            signal = float(signal_line.iloc[-1])

            ma_20 = float(close.rolling(20).mean().iloc[-1])
            ma_50 = float(close.rolling(50).mean().iloc[-1])
            ma_200 = float(close.rolling(200).mean().iloc[-1])
        except Exception as e:
            st.warning(f"⚠️ Gagal ambil data teknikal: {e}")

        # Free Cash Flow
        free_cash_flow = None
        try:
            if 'Free Cash Flow' in ticker.cashflow.index:
                free_cash_flow = ticker.cashflow.loc['Free Cash Flow']
        except Exception as e:
            st.warning(f"⚠️ Gagal ambil Free Cash Flow: {e}")

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

# --- Fungsi Perhitungan ---
def calculate_price_estimates(stock_data, per_industry, pbv_industry):
    eps = stock_data['eps_current']

    per_hist = stock_data['per_history']
    per_low_vals = [v['low'] for v in per_hist.values() if pd.notna(v['low']) and v['low'] > 0]
    per_high_vals = [v['high'] for v in per_hist.values() if pd.notna(v['high']) and v['high'] > 0]

    per_low_min = min(per_low_vals) if per_low_vals else 0
    per_low_max = max(per_low_vals) if per_low_vals else 0
    per_high_min = min(per_high_vals) if per_high_vals else 0
    per_high_max = max(per_high_vals) if per_high_vals else 0

    rendah_low = per_low_min * eps if per_low_min > 0 else 0
    rendah_high = per_low_max * eps if per_low_max > 0 else 0
    optimis_low = per_high_min * eps if per_high_min > 0 else 0
    optimis_high = per_high_max * eps if per_high_max > 0 else 0

    mos = ((rendah_high - stock_data['current_price']) / rendah_high * 100) if rendah_high > 0 else 0
    price_per_ind = per_industry * eps if per_industry > 0 else 0
    price_pbv_ind = pbv_industry * stock_data['book_value'] if pbv_industry > 0 and stock_data['book_value'] > 0 else 0

    dcf_value = ggm_value = asset_value = rim_value = 0

    # DCF
    if stock_data['free_cash_flow'] is not None and not stock_data['free_cash_flow'].empty:
        try:
            fcf = stock_data['free_cash_flow'].iloc[:5]
            if len(fcf) > 0:
                last = fcf.iloc[0]
                proj = []
                for _ in range(5):
                    last = last * (1 + stock_data['revenue_growth'])
                    proj.append(last)
                r = DEFAULT_RISK_FREE_RATE + stock_data['beta'] * DEFAULT_MARKET_RISK_PREMIUM
                tv = proj[-1] * (1 + DEFAULT_TERMINAL_GROWTH_RATE) / (r - DEFAULT_TERMINAL_GROWTH_RATE)
                dcf = sum(proj[i] / ((1 + r) ** (i + 1)) for i in range(5)) + tv / ((1 + r) ** 5)
                dcf_value = dcf / stock_data['shares_outstanding'] if stock_data['shares_outstanding'] > 0 else 0
        except:
            pass

    # Gordon Growth
    if stock_data['dividend_rate'] > 0:
        try:
            d1 = stock_data['dividend_rate'] * (1 + stock_data['revenue_growth'])
            r = DEFAULT_RISK_FREE_RATE + stock_data['beta'] * DEFAULT_MARKET_RISK_PREMIUM
            ggm_value = d1 / (r - stock_data['revenue_growth'])
        except:
            pass

    asset_value = stock_data['book_value'] * 1.2

    # Residual Income
    try:
        bv = stock_data['book_value']
        r = DEFAULT_RISK_FREE_RATE + stock_data['beta'] * DEFAULT_MARKET_RISK_PREMIUM
        roe = stock_data['roe'] / 100
        if bv > 0 and roe > 0:
            rim_value = bv
            for _ in range(5):
                ri = (roe - r) * bv
                rim_value += ri / (1 + r)
                bv *= (1 + stock_data['revenue_growth'])
    except:
        pass

    return {
        'rendah_low': rendah_low,
        'rendah_high': rendah_high,
        'optimis_low': optimis_low,
        'optimis_high': optimis_high,
        'mos': mos,
        'price_from_per_industry': price_per_ind,
        'price_from_pbv_industry': price_pbv_ind,
        'dcf_value': dcf_value,
        'ggm_value': ggm_value,
        'asset_value': asset_value,
        'rim_value': rim_value,
        'per_low_min': per_low_min,
        'per_low_max': per_low_max,
        'per_high_min': per_high_min,
        'per_high_max': per_high_max
    }

# --- Tampilan Hasil ---
def display_results_table(stock_data, estimates, per_industry, pbv_industry):
    st.markdown(f"## {stock_data['ticker']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Harga Saat Ini\n### {format_currency(stock_data['current_price'])}")
    with col2:
        st.markdown(f"### EPS Saat Ini\n### {stock_data['eps_current']:,.2f}")

    st.markdown("### Perkiraan Rentang Harga")
    price_df = pd.DataFrame({
        '': ['Perkiraan Skenario RENDAH', 'Perkiraan Skenario OPTIMIS'],
        'Harga Rendah': [format_currency(estimates['rendah_low']), format_currency(estimates['optimis_low'])],
        'Harga Tinggi': [format_currency(estimates['rendah_high']), format_currency(estimates['optimis_high'])]
    })
    st.dataframe(price_df, hide_index=True, use_container_width=True)

    st.markdown("### Perkiraan Harga dari Sektor Industri")
    ind_df = pd.DataFrame({
        '': ['PER Sektor Industri', 'PBV Sektor Industri'],
        'Nilai': [format_currency(estimates['price_from_per_industry']), format_currency(estimates['price_from_pbv_industry'])]
    })
    st.dataframe(ind_df, hide_index=True, use_container_width=True)

    st.subheader("📚 Data Historis")
    col_eps, col_price = st.columns(2)
    with col_eps:
        st.write("**EPS 5 Tahun Terakhir**")
        eps_df = pd.DataFrame.from_dict(stock_data['eps_history'], orient='index', columns=['EPS'])
        eps_df.index.name = 'Tahun'
        eps_df = eps_df.sort_index(ascending=False)
        eps_df['EPS'] = eps_df['EPS'].apply(lambda x: 'N/A' if pd.isna(x) or x == 0 else f"{x:,.2f}")
        st.dataframe(eps_df, use_container_width=True)
    with col_price:
        st.write("**Harga High & Low**")
        price_df = pd.DataFrame.from_dict(stock_data['price_high_low'], orient='index')
        price_df.index.name = 'Tahun'
        price_df = price_df.sort_index(ascending=False)
        price_df['high'] = price_df['high'].apply(lambda x: 'N/A' if x == 0 else f"Rp {x:,.0f}")
        price_df['low'] = price_df['low'].apply(lambda x: 'N/A' if x == 0 else f"Rp {x:,.0f}")
        st.dataframe(price_df, use_container_width=True)

    st.subheader("📊 PER Historis")
    per_df = pd.DataFrame.from_dict(stock_data['per_history'], orient='index')
    per_df.index.name = 'Tahun'
    per_df = per_df.sort_index(ascending=False)
    per_df['high'] = per_df['high'].apply(lambda x: 'N/A' if pd.isna(x) or x == 0 else f"{x:.2f}")
    per_df['low'] = per_df['low'].apply(lambda x: 'N/A' if pd.isna(x) or x == 0 else f"{x:.2f}")
    st.dataframe(per_df, use_container_width=True)

    st.markdown("### Profitabilitas")
    st.dataframe(pd.DataFrame({
        '': ['ROE', 'OPM', 'NPM'],
        'Nilai': [format_percentage(stock_data['roe']), format_percentage(stock_data['opm']), format_percentage(stock_data['npm'])]
    }), hide_index=True, use_container_width=True)

    st.markdown("""
    **Catatan tentang ROE Adjusted:**  
    ROE Adjusted = (Laba Bersih - Item Tidak Berulang) / Ekuitas Rata-rata.  
    Memberikan gambaran profitabilitas inti yang lebih akurat.
    """)

    st.markdown("### Nilai Intrinsik")
    st.dataframe(pd.DataFrame({
        '': ['DCF', 'Gordon Growth', 'Asset-Based', 'Residual Income'],
        'Nilai': [
            format_currency(estimates['dcf_value']),
            format_currency(estimates['ggm_value']),
            format_currency(estimates['asset_value']),
            format_currency(estimates['rim_value'])
        ]
    }), hide_index=True, use_container_width=True)

    st.markdown("### Indikator Teknikal")
    st.dataframe(pd.DataFrame({
        '': ['RSI', 'MACD', 'MA 20', 'MA 50', 'MA 200'],
        'Nilai': [
            f"{stock_data['rsi']:.2f}" if stock_data['rsi'] > 0 else "N/A",
            f"{stock_data['macd']:.2f}" if stock_data['macd'] != 0 else "N/A",
            format_currency(stock_data['ma_20']),
            format_currency(stock_data['ma_50']),
            format_currency(stock_data['ma_200'])
        ]
    }), hide_index=True, use_container_width=True)

# --- Export PDF ---
def export_to_pdf(stock_data, estimates, per_industry, pbv_industry):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    story = []
    styles = getSampleStyleSheet()

    story.append(Paragraph(f"Analisis Saham {stock_data['ticker']}", ParagraphStyle('Title', fontSize=24, alignment=1)))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Dibuat pada: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ParagraphStyle('Date', fontSize=10, alignment=1)))
    story.append(Spacer(1, 12))

    def safe_format(val, formatter):
        try:
            return formatter(val)
        except:
            return "N/A"

    # Harga & EPS
    data = [
        ['Harga Saat Ini', safe_format(stock_data['current_price'], format_currency)],
        ['EPS Saat Ini', f"{stock_data['eps_current']:,.2f}" if stock_data['eps_current'] else "N/A"]
    ]
    table = Table(data, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.grey)]))
    story.append(table)
    story.append(Spacer(1, 12))

    # EPS Historis
    story.append(Paragraph("EPS Historis", styles['Heading2']))
    eps_data = [['Tahun', 'EPS']]
    for yr in sorted(stock_data['eps_history'].keys(), reverse=True):
        val = stock_data['eps_history'][yr]
        eps_data.append([yr, safe_format(val, lambda x: f"{x:,.2f}")])
    t = Table(eps_data, colWidths=[1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(t)
    story.append(Spacer(1, 12))

    # PDF build
    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    b64 = base64.b64encode(pdf).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="analisis_{stock_data["ticker"]}.pdf">📥 Download PDF</a>'
    return href

# --- Grafik (opsional, tidak diubah signifikan) ---
def plot_risk_reward_chart(rendah_high, rendah_low, optimis_high, optimis_low, current_price):
    import plotly.express as px
    rendah = {'Kategori': ['Tinggi (Rendah)', 'Saat Ini', 'Rendah (Rendah)'], 'Harga': [rendah_high, current_price, rendah_low]}
    fig = px.bar(rendah, x='Kategori', y='Harga', title='Skenario Rendah')
    return fig

# --- Main App ---
st.title("📈 Calculator PRHS - Perkiraan Rentang Harga Saham")
st.markdown("Masukkan kode saham (tanpa `.JK`) dan rasio industri untuk mendapatkan prakiraan harga.")

with st.form("prhs_form"):
    ticker_input = st.text_input("Kode Saham (contoh: TLKM)")
    col1, col2 = st.columns(2)
    with col1:
        per_industry = st.number_input("PER Sektor Industri", value=DEFAULT_PER, min_value=0.0, step=0.1)
    with col2:
        pbv_industry = st.number_input("PBV Sektor Industri", value=DEFAULT_PBV, min_value=0.0, step=0.1)
    submit = st.form_submit_button("Hitung")

if submit:
    if not ticker_input.strip():
        st.error("⚠️ Masukkan kode saham.")
    else:
        ticker_symbol = f"{ticker_input.strip().upper()}.JK"
        with st.spinner(f"Mengambil data {ticker_symbol}..."):
            data = get_stock_data(ticker_symbol)
            if data:
                est = calculate_price_estimates(data, per_industry, pbv_industry)
                display_results_table(data, est, per_industry, pbv_industry)
                st.markdown("---")
                st.markdown(export_to_pdf(data, est, per_industry, pbv_industry), unsafe_allow_html=True)
                st.success("✅ Analisis selesai!")
            else:
                st.error("❌ Gagal mengambil data. Pastikan kode saham valid.")
