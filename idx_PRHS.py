import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Calculator PRHS - Perkiraan Rentang Harga Saham",
    page_icon="📈",
    layout="wide"
)

# --- Fungsi Scraping Data Sektor dari IDX ---
def scrape_idx_sector_data(ticker):
    """
    Scraping data PER dan PBV sektor dari idx.co.id
    Jika gagal, return None dan user harus input manual
    """
    try:
        # Simulasi data sektor (ganti dengan scraping nyata jika diperlukan)
        sector_map = {
            'ABMM.JK': {'PER': 15.2, 'PBV': 1.8},
            'BBCA.JK': {'PER': 22.5, 'PBV': 3.1},
            'TLKM.JK': {'PER': 18.7, 'PBV': 2.3},
            'UNVR.JK': {'PER': 35.4, 'PBV': 4.2},
            'BMRI.JK': {'PER': 16.8, 'PBV': 2.0},
        }
        
        sector_data = sector_map.get(ticker, None)
        if sector_data:
            return sector_data['PER'], sector_data['PBV']
        else:
            return None, None
            
    except Exception as e:
        st.warning(f"⚠️ Gagal mengambil data sektor dari IDX: {str(e)}")
        return None, None

# --- Fungsi Ambil Data Saham dari yfinance ---
def get_stock_data(ticker_symbol):
    """
    Ambil data saham dari yfinance
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Ambil data dasar
        current_price = info.get('currentPrice', 0)
        eps_trailing = info.get('trailingEps', 0)
        eps_forward = info.get('forwardEps', 0)
        roe = info.get('returnOnEquity', 0) * 100  # dalam persen
        opm = info.get('operatingMargins', 0) * 100  # dalam persen
        npm = info.get('profitMargins', 0) * 100  # dalam persen
        
        # Ambil data EPS historis (5 tahun terakhir)
        financials = ticker.financials
        quarterly_financials = ticker.quarterly_financials
        
        eps_history = {}
        years = []
        
        # Coba ambil EPS dari laporan keuangan tahunan
        if not financials.empty and 'Diluted EPS' in financials.index:
            eps_row = financials.loc['Diluted EPS']
            for year in eps_row.index:
                year_str = str(year.year)
                eps_history[year_str] = eps_row[year]
                years.append(year_str)
        
        # Jika tidak ada, coba dari quarterly
        if len(eps_history) < 5:
            eps_quarterly = quarterly_financials.loc['Diluted EPS'] if 'Diluted EPS' in quarterly_financials.index else None
            if eps_quarterly is not None:
                # Gabungkan kuartalan menjadi tahunan
                eps_annual = {}
                for date in eps_quarterly.index:
                    year = str(date.year)
                    if year not in eps_annual:
                        eps_annual[year] = 0
                    eps_annual[year] += eps_quarterly[date]
                
                # Ambil 5 tahun terakhir
                sorted_years = sorted(eps_annual.keys(), reverse=True)[:5]
                for y in sorted_years:
                    eps_history[y] = eps_annual[y]
                    years.append(y)
        
        # Pastikan kita punya 5 tahun data
        if len(eps_history) < 5:
            # Isi dengan data dummy atau gunakan trailing/forward
            for i in range(5 - len(eps_history)):
                eps_history[f"Y-{i+1}"] = eps_trailing if eps_trailing > 0 else eps_forward
        
        # Ambil data harga historis (5 tahun terakhir)
        hist = ticker.history(period="5y", interval="1d")
        price_high_low = {}
        
        for year in range(2020, 2025):  # Contoh: 2020-2024
            year_str = str(year)
            year_data = hist[hist.index.year == year]
            if not year_data.empty:
                high = year_data['High'].max()
                low = year_data['Low'].min()
                price_high_low[year_str] = {'high': high, 'low': low}
            else:
                price_high_low[year_str] = {'high': 0, 'low': 0}
        
        # Untuk tahun berjalan, gunakan data tahun ini
        current_year = str(pd.Timestamp.now().year)
        current_year_data = hist[hist.index.year == int(current_year)]
        if not current_year_data.empty:
            current_high = current_year_data['High'].max()
            current_low = current_year_data['Low'].min()
        else:
            current_high = current_price
            current_low = current_price
        
        price_high_low[current_year] = {'high': current_high, 'low': current_low}
        
        # Hitung PER historis
        per_history = {}
        for year in price_high_low:
            if year in eps_history and eps_history[year] > 0:
                per_high = price_high_low[year]['high'] / eps_history[year]
                per_low = price_high_low[year]['low'] / eps_history[year]
                per_history[year] = {'high': per_high, 'low': per_low}
            else:
                per_history[year] = {'high': 0, 'low': 0}
        
        # Hitung rata-rata PER
        per_high_values = [v['high'] for v in per_history.values() if v['high'] > 0]
        per_low_values = [v['low'] for v in per_history.values() if v['low'] > 0]
        
        avg_per_high = np.mean(per_high_values) if per_high_values else 0
        avg_per_low = np.mean(per_low_values) if per_low_values else 0
        
        # Perbaikan: Hitung EPS tahun ini dari data kuartalan
        eps_current = 0
        if not quarterly_financials.empty and 'Diluted EPS' in quarterly_financials.index:
            eps_quarterly = quarterly_financials.loc['Diluted EPS']
            # Ambil 4 kuartal terakhir
            recent_quarters = eps_quarterly.head(4)
            if len(recent_quarters) >= 4:
                eps_current = recent_quarters.sum()
            else:
                # Jika kurang dari 4 kuartal, gunakan trailing atau forward
                eps_current = eps_trailing if eps_trailing > 0 else eps_forward
        else:
            eps_current = eps_trailing if eps_trailing > 0 else eps_forward
        
        return {
            'ticker': ticker_symbol,
            'current_price': current_price,
            'eps_trailing': eps_trailing,
            'eps_forward': eps_forward,
            'eps_current': eps_current,
            'eps_history': eps_history,
            'price_high_low': price_high_low,
            'per_history': per_history,
            'avg_per_high': avg_per_high,
            'avg_per_low': avg_per_low,
            'roe': roe,
            'opm': opm,
            'npm': npm,
            'info': info
        }
        
    except Exception as e:
        st.error(f"❌ Error mengambil data saham: {str(e)}")
        return None

# --- Fungsi Perhitungan Estimasi Harga ---
def calculate_price_estimates(stock_data, per_industry=None, pbv_industry=None):
    """
    Hitung estimasi harga berdasarkan PER dan PBV
    """
    eps_current = stock_data['eps_current']
    avg_per_high = stock_data['avg_per_high']
    avg_per_low = stock_data['avg_per_low']
    
    # Skenario Netral
    netral_high = avg_per_high * eps_current if avg_per_high > 0 else 0
    netral_low = avg_per_low * eps_current if avg_per_low > 0 else 0
    
    # Skenario Optimis (gunakan PER tertinggi dalam sejarah)
    per_high_values = [v['high'] for v in stock_data['per_history'].values() if v['high'] > 0]
    max_per_high = max(per_high_values) if per_high_values else 0
    optimis_high = max_per_high * eps_current if max_per_high > 0 else 0
    
    # Skenario Optimis Low (gunakan PER terendah dalam sejarah)
    per_low_values = [v['low'] for v in stock_data['per_history'].values() if v['low'] > 0]
    min_per_low = min(per_low_values) if per_low_values else 0
    optimis_low = min_per_low * eps_current if min_per_low > 0 else 0
    
    # Perhitungan MoS (Margin of Safety)
    mos = ((netral_high - stock_data['current_price']) / netral_high * 100) if netral_high > 0 else 0
    
    # Perhitungan dari PER Industri
    price_from_per_industry = per_industry * eps_current if per_industry else 0
    
    # Perhitungan dari PBV Industri (jika tersedia)
    price_from_pbv_industry = 0
    if pbv_industry and 'bookValue' in stock_data['info']:
        book_value = stock_data['info']['bookValue']
        price_from_pbv_industry = pbv_industry * book_value
    
    return {
        'netral_high': netral_high,
        'netral_low': netral_low,
        'optimis_high': optimis_high,
        'optimis_low': optimis_low,
        'mos': mos,
        'price_from_per_industry': price_from_per_industry,
        'price_from_pbv_industry': price_from_pbv_industry
    }

# --- Fungsi Tampilkan Bar Chart Horizontal ---
def plot_risk_reward_chart(netral_high, netral_low, optimis_high, optimis_low, current_price):
    """
    Buat bar chart horizontal untuk Risk & Reward
    """
    # Data untuk skenario netral
    netral_data = {
        'Kategori': ['Harga Tinggi (Netral)', 'Harga Saat Ini', 'Harga Rendah (Netral)'],
        'Harga': [netral_high, current_price, netral_low],
        'Warna': ['#2ecc71', '#3498db', '#e74c3c']
    }
    
    # Data untuk skenario optimis
    optimis_data = {
        'Kategori': ['Harga Tinggi (Optimis)', 'Harga Saat Ini', 'Harga Rendah (Optimis)'],
        'Harga': [optimis_high, current_price, optimis_low],
        'Warna': ['#2ecc71', '#3498db', '#e74c3c']
    }
    
    fig_netral = px.bar(netral_data, y='Kategori', x='Harga', color='Warna',
                        title='Risk & Reward - Skenario Netral',
                        labels={'Harga': 'Harga (Rp)'}, height=300)
    fig_netral.update_layout(showlegend=False, xaxis_tickformat=',.0f')
    fig_netral.update_traces(texttemplate='%{x:.0f}', textposition='outside')
    
    fig_optimis = px.bar(optimis_data, y='Kategori', x='Harga', color='Warna',
                         title='Risk & Reward - Skenario Optimis',
                         labels={'Harga': 'Harga (Rp)'}, height=300)
    fig_optimis.update_layout(showlegend=False, xaxis_tickformat=',.0f')
    fig_optimis.update_traces(texttemplate='%{x:.0f}', textposition='outside')
    
    return fig_netral, fig_optimis

# --- Main App ---
st.title("📈 Calculator PRHS - Perkiraan Rentang Harga Saham")
st.markdown("Navigasi dinamika pasar saham menjadi lebih mudah dengan bantuan Humanis PRHS.")

# Input Form
with st.form("prhs_form"):
    st.subheader("📊 Data Saham")
    ticker_input = st.text_input("Nama Saham (contoh: ABMM)", help="Masukkan kode saham tanpa .JK, sistem akan otomatis menambahkan .JK")
    
    submit_button = st.form_submit_button("Hitung Perkiraan Harga")

if submit_button:
    if not ticker_input.strip():
        st.error("⚠️ Silakan masukkan nama saham.")
    else:
        ticker_symbol = f"{ticker_input.strip().upper()}.JK"
        
        with st.spinner(f"Mengambil data saham {ticker_symbol}..."):
            stock_data = get_stock_data(ticker_symbol)
            
            if stock_data is None:
                st.error("❌ Gagal mengambil data saham. Pastikan kode saham benar dan terdaftar di IDX.")
            else:
                # Ambil data sektor
                per_industry, pbv_industry = scrape_idx_sector_data(ticker_symbol)
                
                # Jika scraping gagal, minta input manual
                if per_industry is None or pbv_industry is None:
                    st.warning("⚠️ Data PER & PBV sektor tidak ditemukan. Silakan isi secara manual:")
                    per_industry = st.number_input("PER Industri", min_value=0.0, value=0.0, step=0.1)
                    pbv_industry = st.number_input("PBV Industri", min_value=0.0, value=0.0, step=0.1)
                else:
                    st.success(f"✅ Data sektor berhasil diambil: PER={per_industry}, PBV={pbv_industry}")
                
                # Hitung estimasi harga
                estimates = calculate_price_estimates(stock_data, per_industry, pbv_industry)
                
                # Tampilkan hasil dalam format tabel
                st.header("📈 Perkiraan Rentang Harga Saham")
                
                # Buat tabel untuk skenario netral dan optimis
                table_data = {
                    'Skenario': ['RERATA (Netral)', 'TERTINGGI (Optimis)'],
                    'Harga Tinggi (Rp)': [
                        f"{estimates['netral_high']:,.0f}" if estimates['netral_high'] > 0 else "N/A",
                        f"{estimates['optimis_high']:,.0f}" if estimates['optimis_high'] > 0 else "N/A"
                    ],
                    'Harga Rendah (Rp)': [
                        f"{estimates['netral_low']:,.0f}" if estimates['netral_low'] > 0 else "N/A",
                        f"{estimates['optimis_low']:,.0f}" if estimates['optimis_low'] > 0 else "N/A"
                    ]
                }
                
                table_df = pd.DataFrame(table_data)
                st.table(table_df)
                
                # Tampilkan harga saat ini
                st.subheader("💰 Harga Saat Ini")
                st.metric("", f"Rp {stock_data['current_price']:,.0f}")
                
                # Tampilkan ROE, OPM, NPM
                st.subheader("📊 Profitabilitas")
                col_roe, col_opm, col_npm = st.columns(3)
                col_roe.metric("ROE", f"{stock_data['roe']:.2f}%")
                col_opm.metric("OPM", f"{stock_data['opm']:.2f}%")
                col_npm.metric("NPM", f"{stock_data['npm']:.2f}%")
                
                # Tampilkan MoS
                st.subheader("🛡️ Margin of Safety (MoS)")
                st.metric("", f"{estimates['mos']:.2f}%")
                
                # Tampilkan perkiraan harga dari PER & PBV industri
                st.subheader("🏢 Perkiraan Harga dari PER & PBV Industri")
                col_per, col_pbv = st.columns(2)
                col_per.metric("Dari PER Industri", f"Rp {estimates['price_from_per_industry']:,.0f}" if estimates['price_from_per_industry'] > 0 else "N/A")
                col_pbv.metric("Dari PBV Industri", f"Rp {estimates['price_from_pbv_industry']:,.0f}" if estimates['price_from_pbv_industry'] > 0 else "N/A")
                
                # Tampilkan bar chart Risk & Reward
                st.subheader("📉 Risk & Reward")
                fig_netral, fig_optimis = plot_risk_reward_chart(
                    estimates['netral_high'],
                    estimates['netral_low'],
                    estimates['optimis_high'],
                    estimates['optimis_low'],
                    stock_data['current_price']
                )
                
                st.plotly_chart(fig_netral, use_container_width=True)
                st.plotly_chart(fig_optimis, use_container_width=True)
                
                # Tampilkan data historis EPS dan harga
                st.subheader("📚 Data Historis")
                col_eps, col_price = st.columns(2)
                
                with col_eps:
                    st.write("**EPS 5 Tahun Terakhir**")
                    eps_df = pd.DataFrame.from_dict(stock_data['eps_history'], orient='index', columns=['EPS'])
                    eps_df.index.name = 'Tahun'
                    st.dataframe(eps_df.style.format({'EPS': '{:,.2f}'}))
                
                with col_price:
                    st.write("**Harga High & Low 5 Tahun Terakhir**")
                    price_df = pd.DataFrame.from_dict(stock_data['price_high_low'], orient='index')
                    price_df.index.name = 'Tahun'
                    st.dataframe(price_df.style.format({'high': 'Rp {:,.0f}', 'low': 'Rp {:,.0f}'}))
                
                # Tampilkan PER historis
                st.subheader("📊 PER Historis")
                per_df = pd.DataFrame.from_dict(stock_data['per_history'], orient='index')
                per_df.index.name = 'Tahun'
                st.dataframe(per_df.style.format({'high': '{:.2f}', 'low': '{:.2f}'}))
                
                st.success("✅ Perhitungan selesai!")