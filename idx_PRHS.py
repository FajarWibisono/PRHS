"""
app.py — Streamlit UI for FUNDAMENTAL Stock Future Value Analyzer
Language: Bahasa Indonesia
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data_fetcher import fetch_stock_data
from fcds_t_calculator import calculate_fcds_t

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fundamental Future Value Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helper formatters ──────────────────────────────────────────────────────────

def fmt_currency(val, currency="IDR", decimals=0):
    """Format number as currency string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    prefix = "Rp " if currency == "IDR" else "$ "
    if decimals == 0:
        return f"{prefix}{val:,.0f}"
    return f"{prefix}{val:,.{decimals}f}"


def fmt_pct(val, decimals=2):
    """Format as percentage string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:+.{decimals}f}%"


def fmt_num(val, decimals=2):
    """Format plain number."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:,.{decimals}f}"


def safe_float(val):
    """Return float or NaN for session state."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan Proyeksi")
    projection_years = st.slider("Tahun Proyeksi", min_value=1, max_value=10, value=5, step=1)

    st.subheader("Bobot Komponen FV")
    w_bv = st.number_input("Bobot BV (%)", min_value=0, max_value=100, value=50, step=5)
    w_eps = st.number_input("Bobot EPS (%)", min_value=0, max_value=100, value=40, step=5)
    w_sps = st.number_input("Bobot SPS (%)", min_value=0, max_value=100, value=10, step=5)
    total_w = w_bv + w_eps + w_sps
    if total_w != 100:
        st.warning(f"Total bobot = {total_w}%. Akan dinormalisasi otomatis.")

    st.divider()
    st.caption("**FUNS FUVA** = FUNdamental Stock FUture Value Analysis")
    st.caption("Metode value investing berbasis proyeksi fundamentals 5 tahun.")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 FUNDAMENTAL Stock Future Value Analyzer")
st.markdown(
    "Estimasi **harga wajar masa depan** saham menggunakan pendekatan fundamental"
    "berdasarkan pertumbuhan BV, EPS, dan SPS historis yang dikomposit."
)

# ── Session State Init ─────────────────────────────────────────────────────────
if "fetched" not in st.session_state:
    st.session_state.fetched = {}
if "overrides" not in st.session_state:
    st.session_state.overrides = {}
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = ""
if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0

# ── Input Section ──────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([3, 1], vertical_alignment="bottom")
with col_input:
    ticker_input = st.text_input(
        "Kode Saham",
        placeholder="Contoh: BMRI (IDX) atau NVDA (US)",
        help="Untuk saham IDX, cukup ketik kode tanpa .JK (misal BMRI). Untuk US ketik langsung (NVDA).",
        key=f"ticker_input_{st.session_state.reset_count}",
    )
with col_btn:
    analyze_btn = st.button("🔍 Analisis", use_container_width=True, type="primary")

# ── Fetch on button click ──────────────────────────────────────────────────────
if analyze_btn and ticker_input.strip():
    with st.spinner(f"Mengambil data {ticker_input.upper()}..."):
        data = fetch_stock_data(ticker_input)

    if data.get("error") and not data.get("last_price"):
        st.error(f"❌ Gagal mengambil data untuk **{ticker_input.upper()}**: {data['error']}")
    else:
        st.session_state.fetched = data
        st.session_state.overrides = {}  # reset overrides for new ticker
        st.session_state.current_ticker = ticker_input

# ── Main content (only if data is available) ───────────────────────────────────
if st.session_state.fetched:
    fd = st.session_state.fetched
    currency = fd.get("currency", "IDR")
    dq = fd.get("data_quality", {})

    # Show fetch warnings
    missing_fields = [k for k, v in dq.items() if v == "missing"]
    if missing_fields:
        st.warning(
            f"⚠️ Data berikut tidak tersedia dari yfinance dan perlu diisi manual: "
            f"**{', '.join(missing_fields)}**"
        )
    if fd.get("error"):
        st.info(f"ℹ️ Catatan fetcher: {fd['error']}")

    st.subheader(f"🏢 {fd.get('company_name', '')} ({fd.get('ticker', '')})")

    # ── Editable Input Fields ──────────────────────────────────────────────────
    st.markdown("### 📊 Data Fundamental")
    st.caption("Data diambil otomatis. Anda dapat mengubah nilai jika diperlukan.")

    def get_val(key, default=0.0):
        """Get override or fetched value."""
        ov = st.session_state.overrides.get(key)
        if ov is not None:
            return ov
        v = fd.get(key, np.nan)
        return float(v) if pd.notna(v) else default

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**Data Saham Terkini**")

        last_price = st.number_input(
            "Harga Terakhir",
            value=get_val("last_price"),
            min_value=0.0,
            format="%.2f",
            key="inp_last_price",
            help="Harga penutupan terakhir"
        )
        bv_per_share = st.number_input(
            "Book Value per Saham (BV)",
            value=get_val("bv_per_share"),
            min_value=0.0,
            format="%.2f",
            key="inp_bv",
            help="Ekuitas per saham"
        )
        eps_ttm = st.number_input(
            "EPS (Laba per Saham)",
            value=get_val("eps_ttm"),
            format="%.4f",
            key="inp_eps",
            help="Earnings per share, TTM"
        )
        sps_ttm = st.number_input(
            "SPS (Penjualan per Saham)",
            value=get_val("sps_ttm"),
            min_value=0.0,
            format="%.4f",
            key="inp_sps",
            help="Sales per share, TTM"
        )
        _dpr_raw = get_val("dpr", 0.0)
        _dpr_safe = float(np.clip(_dpr_raw, 0.0, 2.0))
        dpr_val = st.number_input(
            "DPR (Dividend Payout Ratio)",
            value=_dpr_safe,
            min_value=0.0,
            max_value=2.0,
            format="%.4f",
            key="inp_dpr",
            help="Rasio pembagian dividen (0.30 = 30%). Maksimum 2.0 (200%)."
        )

    with right_col:
        st.markdown("**Rata-rata Historis & Pertumbuhan**")

        avg_pbv = st.number_input(
            "Rata-rata PBV historis",
            value=get_val("avg_pbv", 1.0),
            min_value=0.0,
            format="%.2f",
            key="inp_avg_pbv",
            help="Price-to-Book Value rata-rata 5 tahun"
        )
        avg_per = st.number_input(
            "Rata-rata PER historis",
            value=get_val("avg_per", 10.0),
            min_value=0.0,
            format="%.2f",
            key="inp_avg_per",
            help="Price-to-Earnings Ratio rata-rata 5 tahun"
        )
        avg_psr = st.number_input(
            "Rata-rata PSR historis",
            value=get_val("avg_psr", 1.0),
            min_value=0.0,
            format="%.2f",
            key="inp_avg_psr",
            help="Price-to-Sales Ratio rata-rata 5 tahun"
        )

        st.markdown("**ROE**")
        roe_annual = st.number_input(
            "ROE Tahunan (decimal)",
            value=get_val("roe_annual", 0.15),
            format="%.4f",
            key="inp_roe_annual",
            help="Return on Equity tahun terakhir (mis. 0.20 = 20%)"
        )
        roe_5yr = st.number_input(
            "ROE 5 Tahun (rata-rata)",
            value=get_val("roe_5yr", 0.15),
            format="%.4f",
            key="inp_roe_5yr",
        )

        st.markdown("**Pertumbuhan EPS**")
        eps_gr_annual = st.number_input(
            "EPS Growth Tahunan",
            value=get_val("eps_growth_annual", 0.10),
            format="%.4f",
            key="inp_eps_gr_annual",
            help="Pertumbuhan EPS tahun terakhir (mis. 0.10 = 10%)"
        )
        eps_gr_5yr = st.number_input(
            "EPS Growth 5 Tahun (CAGR)",
            value=get_val("eps_growth_5yr", 0.10),
            format="%.4f",
            key="inp_eps_gr_5yr",
        )

        st.markdown("**Pertumbuhan SPS**")
        sps_gr_annual = st.number_input(
            "SPS Growth Tahunan",
            value=get_val("sps_growth_annual", 0.08),
            format="%.4f",
            key="inp_sps_gr_annual",
        )
        sps_gr_5yr = st.number_input(
            "SPS Growth 5 Tahun (CAGR)",
            value=get_val("sps_growth_5yr", 0.08),
            format="%.4f",
            key="inp_sps_gr_5yr",
        )

    # ── Run calculation ────────────────────────────────────────────────────────
    params = {
        "last_price": last_price,
        "bv_per_share": bv_per_share,
        "eps_ttm": eps_ttm,
        "sps_ttm": sps_ttm,
        "roe_annual": roe_annual,
        "roe_5yr": roe_5yr,
        "eps_growth_annual": eps_gr_annual,
        "eps_growth_5yr": eps_gr_5yr,
        "sps_growth_annual": sps_gr_annual,
        "sps_growth_5yr": sps_gr_5yr,
        "avg_pbv": avg_pbv,
        "avg_per": avg_per,
        "avg_psr": avg_psr,
        "dpr": dpr_val,
        "projection_years": projection_years,
        "weight_bv": w_bv / 100,
        "weight_eps": w_eps / 100,
        "weight_sps": w_sps / 100,
    }

    calc = calculate_fcds_t(params)

    if calc.get("error"):
        st.error(f"❌ Error kalkulasi: {calc['error']}")
    else:
        # ── Results Section ────────────────────────────────────────────────────
        st.divider()
        st.markdown("## 🎯 Hasil Analisis FUNS FUVA")

        # Signal badge
        signal = calc.get("signal", "N/A")
        signal_colors = {
            "UNDERVALUED": ("✅", "green", "UNDERVALUED — Harga di Bawah Nilai Wajar"),
            "FAIR VALUE":  ("⚠️", "orange", "FAIR VALUE — Harga Mendekati Nilai Wajar"),
            "OVERVALUED":  ("❌", "red", "OVERVALUED — Harga Di Atas Nilai Wajar"),
            "N/A":         ("❓", "gray", "Data Tidak Cukup"),
        }
        icon, color, label = signal_colors.get(signal, ("❓", "gray", signal))
        st.markdown(
            f"<div style='background-color: {color}; color: white; padding: 12px 20px; "
            f"border-radius: 8px; font-size: 18px; font-weight: bold; text-align: center;'>"
            f"{icon} {label}</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        fv = calc.get("fv_combined", np.nan)
        gl = calc.get("gl_pct", np.nan)
        mos = calc.get("mos_pct", np.nan)
        cagr = calc.get("cagr_pct", np.nan)
        div_yield = calc.get("div_yield_pct", np.nan)

        m1.metric("💰 Harga Sekarang", fmt_currency(last_price, currency))
        m2.metric(
            f"🚀 Future Value ({projection_years}th)",
            fmt_currency(fv, currency),
            delta=fmt_pct(gl) if not np.isnan(gl) else None,
        )
        m3.metric(
            "🛡️ Margin of Safety",
            fmt_pct(mos) if not np.isnan(mos) else "—",
        )
        m4.metric(
            "📈 CAGR Proyeksi",
            fmt_pct(cagr) if not np.isnan(cagr) else "—",
        )

        # Secondary metrics
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("💸 Dividend Yield", fmt_pct(div_yield) if not np.isnan(div_yield) else "—")
        sm2.metric("📦 Akum. Dividen (" + str(projection_years) + "th)",
                   fmt_currency(calc.get("accumulated_dividend", np.nan), currency))
        sm3.metric("🎯 FV tanpa Dividen",
                   fmt_currency(calc.get("fv_combined_no_div", np.nan), currency))

        # Composite growth rates
        with st.expander("📐 Composite Growth Rates (detail kalkulasi)"):
            cgc1, cgc2, cgc3 = st.columns(3)
            cgc1.metric("Composite ROE", fmt_pct(calc.get("composite_roe", np.nan) * 100, 2))
            cgc2.metric("Composite EPS Growth", fmt_pct(calc.get("composite_eps_gr", np.nan) * 100, 2))
            cgc3.metric("Composite SPS Growth", fmt_pct(calc.get("composite_sps_gr", np.nan) * 100, 2))

        # ── Projection Table ───────────────────────────────────────────────────
        st.markdown("### 📅 Tabel Proyeksi Tahunan")
        table = calc.get("projection_table", [])
        if table:
            df_table = pd.DataFrame(table)
            df_display = pd.DataFrame({
                "Tahun": df_table["year"],
                "BV Proyeksi": df_table["bv_fv"].map(lambda x: fmt_num(x, 2)),
                "EPS Proyeksi": df_table["eps_fv"].map(lambda x: fmt_num(x, 4)),
                "SPS Proyeksi": df_table["sps_fv"].map(lambda x: fmt_num(x, 2)),
                f"Harga via BV": df_table["fp_bv"].map(lambda x: fmt_currency(x, currency, 0)),
                f"Harga via EPS": df_table["fp_eps"].map(lambda x: fmt_currency(x, currency, 0)),
                f"Harga via SPS": df_table["fp_sps"].map(lambda x: fmt_currency(x, currency, 0)),
                "FV Gabungan": df_table["fv_combined_no_div"].map(lambda x: fmt_currency(x, currency, 0)),
                "Akum. Dividen": df_table["div_accumulated"].map(lambda x: fmt_currency(x, currency, 0)),
                "FV + Dividen": df_table["fv_combined"].map(lambda x: fmt_currency(x, currency, 0)),
            })
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        # ── Charts ─────────────────────────────────────────────────────────────
        st.markdown("### 📉 Grafik Proyeksi Harga")
        if table:
            df_chart = pd.DataFrame(table)
            from datetime import datetime
            current_year = datetime.now().year

            # Add current year as base row
            base_row = {
                "year": current_year,
                "fp_bv": last_price,
                "fp_eps": last_price,
                "fp_sps": last_price,
                "fv_combined_no_div": last_price,
                "fv_combined": last_price,
            }
            years = [current_year] + list(df_chart["year"])
            fp_bv_vals = [last_price] + list(df_chart["fp_bv"])
            fp_eps_vals = [last_price] + list(df_chart["fp_eps"])
            fp_sps_vals = [last_price] + list(df_chart["fp_sps"])
            fv_vals = [last_price] + list(df_chart["fv_combined_no_div"])
            fv_div_vals = [last_price] + list(df_chart["fv_combined"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=fp_bv_vals, mode="lines+markers",
                                     name="Via BV", line=dict(color="#2196F3", dash="dot")))
            fig.add_trace(go.Scatter(x=years, y=fp_eps_vals, mode="lines+markers",
                                     name="Via EPS", line=dict(color="#4CAF50", dash="dot")))
            fig.add_trace(go.Scatter(x=years, y=fp_sps_vals, mode="lines+markers",
                                     name="Via SPS", line=dict(color="#FF9800", dash="dot")))
            fig.add_trace(go.Scatter(x=years, y=fv_vals, mode="lines+markers",
                                     name="FV Gabungan", line=dict(color="#9C27B0", width=2)))
            fig.add_trace(go.Scatter(x=years, y=fv_div_vals, mode="lines+markers",
                                     name="FV + Dividen", line=dict(color="#F44336", width=3)))
            fig.add_hline(y=last_price, line_dash="dash", line_color="gray",
                          annotation_text="Harga Sekarang")

            y_label = "Harga (Rp)" if currency == "IDR" else "Price ($)"
            fig.update_layout(
                xaxis_title="Tahun",
                yaxis_title=y_label,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Donut chart — weight breakdown
        st.markdown("### 🍩 Bobot Komponen Future Value")
        if total_w > 0:
            fig_donut = px.pie(
                values=[w_bv, w_eps, w_sps],
                names=["Book Value (BV)", "Earnings (EPS)", "Sales (SPS)"],
                hole=0.4,
                color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800"],
            )
            fig_donut.update_layout(height=320)
            st.plotly_chart(fig_donut, use_container_width=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** Analisis ini hanya untuk tujuan referensi dan edukasi, "
        "bukan merupakan rekomendasi investasi. Nilai investasi dapat naik dan turun. "
        "Selalu lakukan riset mandiri (DYOR) sebelum membuat keputusan investasi."
    )

    # ── Reset button ────────────────────────────────────────────────────────────
    st.divider()
    if st.button("🔄 Reset — Analisis Saham Baru", use_container_width=True):
        st.session_state.fetched = {}
        st.session_state.overrides = {}
        st.session_state.current_ticker = ""
        st.session_state.reset_count += 1
        st.rerun()

elif not analyze_btn:
    st.info("👆 Masukkan kode saham dan klik **Analisis** untuk memulai.")
