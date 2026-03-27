import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import re
import time
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PRHS — Perkiraan Rentang Harga Saham",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: #f8f9fb;
        border: 1px solid #e0e4ea;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .output-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a5276;
        margin-bottom: 4px;
    }
    thead tr th {
        background-color: #1a5276 !important;
        color: white !important;
    }
    [data-testid="stToolbarActions"] {display: none !important;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def normalize_ticker(code: str) -> str:
    code = code.strip().upper()
    if "." not in code:
        code += ".JK"
    return code


def fmt_rp(val) -> str:
    if pd.isna(val):
        return "N/A"
    return f"Rp {val:,.2f}"


def fmt_rp_short(val) -> str:
    if pd.isna(val):
        return "N/A"
    return f"Rp {val:,.0f}"


def safe_get(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


# ============================================================
# DATA FETCHING
# ============================================================
def _yf_call(fn, retries=3):
    """Panggil fungsi yfinance dengan retry + backoff saat rate limited."""
    for attempt in range(retries):
        try:
            result = fn()
            time.sleep(0.4)  # jeda sopan antar request
            return result
        except Exception as e:
            msg = str(e).lower()
            if "too many requests" in msg or "rate limit" in msg or "429" in msg:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))  # 3s, 6s, 9s
                    continue
                raise ValueError("Yahoo Finance membatasi akses (rate limit). Tunggu beberapa saat lalu coba lagi.")
            raise


def _parse_financials(fin, shares: float, existing: dict) -> dict:
    """Ekstrak EPS per tahun dari DataFrame financials yfinance."""
    result = dict(existing)
    if fin is None or (hasattr(fin, "empty") and fin.empty):
        return result
    ni_keys = ["Net Income", "Net Income Common Stockholders", "NetIncome",
               "Net Income From Continuing Operations"]
    ni_key = next((k for k in ni_keys if k in fin.index), None)
    if not ni_key:
        return result
    ni = fin.loc[ni_key]
    for col in ni.index:
        try:
            yr = pd.to_datetime(col).year
        except Exception:
            continue
        val = ni[col]
        if pd.notna(val) and yr not in result:
            result[yr] = float(val) / shares
    return result


def _parse_balance_sheet(bs, shares: float, existing: dict) -> dict:
    """Ekstrak BVPS per tahun dari DataFrame balance sheet yfinance."""
    result = dict(existing)
    if bs is None or (hasattr(bs, "empty") and bs.empty):
        return result
    eq_keys = [
        "Stockholders Equity", "Common Stock Equity", "Total Stockholder Equity",
        "Total Equity Gross Minority Interest", "Stockholders Equity (Net Minority Interest)",
    ]
    eq_key = next((k for k in eq_keys if k in bs.index), None)
    if not eq_key:
        return result
    eq = bs.loc[eq_key]
    for col in eq.index:
        try:
            yr = pd.to_datetime(col).year
        except Exception:
            continue
        val = eq[col]
        if pd.notna(val) and yr not in result:
            result[yr] = float(val) / shares
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_data(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)

    # --- Validasi primer via fast_info (ringan & reliable) ---
    fast_price = None
    fast_shares = None
    try:
        fast = _yf_call(lambda: ticker.fast_info)
        fast_price  = getattr(fast, "last_price",  None)
        fast_shares = getattr(fast, "shares",       None)
    except Exception:
        pass

    # --- Full info (bisa lambat / kosong untuk saham tertentu) ---
    try:
        info = _yf_call(lambda: ticker.info)
    except Exception:
        info = {}
    if not isinstance(info, dict):
        info = {}

    # Validasi: cukup punya harga ATAU nama perusahaan — jangan tolak cuma karena quoteType kosong
    has_price = fast_price is not None or \
        safe_get(info, "currentPrice", "regularMarketPrice", "previousClose") is not None
    has_name  = safe_get(info, "longName", "shortName") is not None
    if not has_price and not has_name:
        raise ValueError(
            f"Saham '{symbol}' tidak ditemukan atau tidak tersedia di Yahoo Finance, gagal mendapatkan data."
        )

    company_name  = safe_get(info, "longName", "shortName", default=symbol)
    _cp           = safe_get(info, "currentPrice", "regularMarketPrice", "previousClose")
    current_price = _cp if _cp is not None else fast_price
    current_eps   = safe_get(info, "trailingEps")
    current_bvps  = safe_get(info, "bookValue")
    _sh           = safe_get(info, "sharesOutstanding")
    shares        = _sh if _sh is not None else fast_shares
    sector        = info.get("sector", "")
    industry      = info.get("industry", "")

    # ---- Annual High / Low Prices (5 years) ----
    annual_prices = []
    try:
        hist = _yf_call(lambda: ticker.history(period="5y", interval="1d"))
    except Exception:
        hist = pd.DataFrame()
    if not hist.empty:
        hist.index = pd.to_datetime(hist.index)
        hist["Year"] = hist.index.year
        grp = hist.groupby("Year").agg(High=("High", "max"), Low=("Low", "min"))
        grp = grp[grp.index <= datetime.now().year]
        grp = grp.tail(5)
        for yr, row in grp.iterrows():
            annual_prices.append({"Tahun": int(yr), "Harga Tertinggi": row["High"], "Harga Terendah": row["Low"]})

    # ---- Annual EPS — coba beberapa sumber secara berurutan ----
    eps_by_year: dict[int, float] = {}
    if shares and shares > 0:
        for fin_fn in [
            lambda: ticker.financials,         # annual (API lama)
            lambda: ticker.income_stmt,        # annual (API baru yfinance ≥ 0.2)
            lambda: ticker.quarterly_financials,
            lambda: ticker.quarterly_income_stmt,
        ]:
            try:
                fin = _yf_call(fin_fn)
                eps_by_year = _parse_financials(fin, shares, eps_by_year)
                if len(eps_by_year) >= 2:
                    break
            except Exception:
                continue

    # Fallback: trailingEps sebagai tahun berjalan
    if current_eps is not None and datetime.now().year not in eps_by_year:
        eps_by_year[datetime.now().year] = float(current_eps)

    # ---- Annual BVPS — coba beberapa sumber secara berurutan ----
    bvps_by_year: dict[int, float] = {}
    if shares and shares > 0:
        for bs_fn in [
            lambda: ticker.balance_sheet,          # annual (API lama)
            lambda: ticker.get_balance_sheet(),    # annual (API baru)
            lambda: ticker.quarterly_balance_sheet,
        ]:
            try:
                bs = _yf_call(bs_fn)
                bvps_by_year = _parse_balance_sheet(bs, shares, bvps_by_year)
                if len(bvps_by_year) >= 2:
                    break
            except Exception:
                continue

    if current_bvps is not None and datetime.now().year not in bvps_by_year:
        bvps_by_year[datetime.now().year] = float(current_bvps)

    return {
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "current_price": current_price,
        "current_eps": current_eps,
        "current_bvps": current_bvps,
        "shares": shares,
        "annual_prices": annual_prices,
        "eps_by_year": eps_by_year,
        "bvps_by_year": bvps_by_year,
    }


# ============================================================
# BUILD HISTORICAL TABLE
# ============================================================
def build_hist_table(annual_prices: list, eps_by_year: dict, bvps_by_year: dict) -> pd.DataFrame:
    rows = []
    for item in annual_prices:
        yr   = item["Tahun"]
        high = item["Harga Tertinggi"]
        low  = item["Harga Terendah"]
        eps  = eps_by_year.get(yr)
        bvps = bvps_by_year.get(yr)

        per_high = high / eps  if (eps  and eps  > 0) else np.nan
        per_low  = low  / eps  if (eps  and eps  > 0) else np.nan
        pbv_high = high / bvps if (bvps and bvps > 0) else np.nan
        pbv_low  = low  / bvps if (bvps and bvps > 0) else np.nan

        rows.append({
            "Tahun":           yr,
            "Harga Tertinggi": high,
            "Harga Terendah":  low,
            "EPS":             eps,
            "BVPS":            bvps,
            "PER Tertinggi":   per_high,
            "PER Terendah":    per_low,
            "PBV Tertinggi":   pbv_high,
            "PBV Terendah":    pbv_low,
        })
    return pd.DataFrame(rows)


def recalc_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps_safe  = df["EPS"].where(df["EPS"].notna()  & (df["EPS"]  > 0))
    bvps_safe = df["BVPS"].where(df["BVPS"].notna() & (df["BVPS"] > 0))
    df["PER Tertinggi"] = df["Harga Tertinggi"] / eps_safe
    df["PER Terendah"]  = df["Harga Terendah"]  / eps_safe
    df["PBV Tertinggi"] = df["Harga Tertinggi"] / bvps_safe
    df["PBV Terendah"]  = df["Harga Terendah"]  / bvps_safe
    return df


# ============================================================
# EPS GROWTH ESTIMATION
# ============================================================
def estimate_eps_growth(eps_by_year: dict) -> tuple[float | None, float | None]:
    """Returns (cagr, estimated_eps_next_year)."""
    valid = sorted([(yr, v) for yr, v in eps_by_year.items() if v and v > 0])
    if len(valid) < 2:
        return None, None
    oldest_yr, oldest_val = valid[0]
    latest_yr, latest_val = valid[-1]
    n = latest_yr - oldest_yr
    if n <= 0:
        return None, None
    cagr = (latest_val / oldest_val) ** (1 / n) - 1
    eps_est = latest_val * (1 + cagr)
    return cagr, eps_est


# ============================================================
# SCENARIO CALCULATIONS
# ============================================================
def calculate_scenarios(
    df: pd.DataFrame,
    eps_est: float,
    bvps_current: float,
    per_industri: float | None,
    pbv_industri: float | None,
) -> pd.DataFrame:

    valid_per = df[df["EPS"].notna() & (df["EPS"] > 0) & df["PER Tertinggi"].notna()]
    valid_bv  = df[df["BVPS"].notna() & (df["BVPS"] > 0) & df["PBV Tertinggi"].notna()]

    results = []

    # ── 1. Estimasi Optimis ──────────────────────────────────────────────────
    # Batas Atas  = max(PER Tertinggi 5 thn) × EPS estimated
    # Batas Bawah = max(PER Terendah  5 thn) × EPS estimated
    if not valid_per.empty:
        max_per_h = valid_per["PER Tertinggi"].max()
        max_per_l = valid_per["PER Terendah"].max()
        results.append({
            "Skenario":    "🚀 Estimasi Optimis",
            "Batas Atas":  round(max_per_h * eps_est, 2),
            "Batas Bawah": round(max_per_l * eps_est, 2),
            "_note":       f"PER Tertinggi maks={max_per_h:.1f}x | PER Terendah maks={max_per_l:.1f}x",
        })
    else:
        results.append({"Skenario": "🚀 Estimasi Optimis", "Batas Atas": None, "Batas Bawah": None, "_note": "Data EPS tidak tersedia"})

    # ── 2. Estimasi Netral ───────────────────────────────────────────────────
    # Batas Atas  = avg(PER Tertinggi 5 thn) × EPS estimated
    # Batas Bawah = avg(PER Terendah  5 thn) × EPS estimated
    if not valid_per.empty:
        avg_per_h = valid_per["PER Tertinggi"].mean()
        avg_per_l = valid_per["PER Terendah"].mean()
        results.append({
            "Skenario":    "⚖️ Estimasi Netral",
            "Batas Atas":  round(avg_per_h * eps_est, 2),
            "Batas Bawah": round(avg_per_l * eps_est, 2),
            "_note":       f"PER Tertinggi rata={avg_per_h:.1f}x | PER Terendah rata={avg_per_l:.1f}x",
        })
    else:
        results.append({"Skenario": "⚖️ Estimasi Netral", "Batas Atas": None, "Batas Bawah": None, "_note": "Data EPS tidak tersedia"})

    # ── 3. Rerata BV (Book Value) ────────────────────────────────────────────
    # Batas Atas  = avg(PBV Tertinggi historis) × BVPS saat ini
    # Batas Bawah = avg(PBV Terendah  historis) × BVPS saat ini
    if not valid_bv.empty and bvps_current and bvps_current > 0:
        avg_pbv_h = valid_bv["PBV Tertinggi"].mean()
        avg_pbv_l = valid_bv["PBV Terendah"].mean()
        results.append({
            "Skenario":    "📚 Rerata BV",
            "Batas Atas":  round(avg_pbv_h * bvps_current, 2),
            "Batas Bawah": round(avg_pbv_l * bvps_current, 2),
            "_note":       f"PBV Tertinggi rata={avg_pbv_h:.2f}x | PBV Terendah rata={avg_pbv_l:.2f}x",
        })
    else:
        results.append({"Skenario": "📚 Rerata BV", "Batas Atas": None, "Batas Bawah": None, "_note": "Data BVPS tidak tersedia"})

    # ── 4. Rerata PER Industri ───────────────────────────────────────────────
    # Nilai estimasi = PER Industri × EPS estimated  (satu nilai tunggal)
    if per_industri and eps_est:
        harga_per = round(per_industri * eps_est, 2)
        results.append({
            "Skenario":    "🏭 Rerata PER Industri",
            "Batas Atas":  harga_per,
            "Batas Bawah": harga_per,
            "_note":       f"PER Industri={per_industri:.1f}x × EPS Est. ({fmt_rp(eps_est)})",
        })
    else:
        results.append({"Skenario": "🏭 Rerata PER Industri", "Batas Atas": None, "Batas Bawah": None, "_note": "Isi PER Industri"})

    # ── 5. Rerata PBV Industri ───────────────────────────────────────────────
    # Nilai estimasi = PBV Industri × BVPS saat ini  (satu nilai tunggal)
    if pbv_industri and bvps_current and bvps_current > 0:
        harga_pbv = round(pbv_industri * bvps_current, 2)
        results.append({
            "Skenario":    "📖 Rerata PBV Industri",
            "Batas Atas":  harga_pbv,
            "Batas Bawah": harga_pbv,
            "_note":       f"PBV Industri={pbv_industri:.2f}x × BVPS ({fmt_rp(bvps_current)})",
        })
    else:
        results.append({"Skenario": "📖 Rerata PBV Industri", "Batas Atas": None, "Batas Bawah": None, "_note": "Isi PBV Industri & BVPS"})

    return pd.DataFrame(results)


# ============================================================
# CHART
# ============================================================
def build_range_chart(scenarios_df: pd.DataFrame, current_price: float | None) -> go.Figure:
    valid = scenarios_df[scenarios_df["Batas Atas"].notna() & scenarios_df["Batas Bawah"].notna()]
    if valid.empty:
        return None

    labels = valid["Skenario"].tolist()
    highs  = valid["Batas Atas"].tolist()
    lows   = valid["Batas Bawah"].tolist()

    fig = go.Figure()

    colors = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]
    for i, (label, h, l) in enumerate(zip(labels, highs, lows)):
        color = colors[i % len(colors)]
        fig.add_trace(go.Bar(
            name=label,
            x=[label],
            y=[h - l],
            base=[l],
            marker_color=color,
            opacity=0.75,
            text=[f"{fmt_rp_short(l)} – {fmt_rp_short(h)}"],
            textposition="outside" if h == l else "inside",
            hovertemplate=f"<b>{label}</b><br>Batas Atas: {fmt_rp_short(h)}<br>Batas Bawah: {fmt_rp_short(l)}<extra></extra>",
        ))

    if current_price:
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Harga Saat Ini: {fmt_rp_short(current_price)}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Perkiraan Rentang Harga — Perbandingan Skenario",
        yaxis_title="Harga (Rp)",
        xaxis_title="",
        showlegend=False,
        height=420,
        barmode="overlay",
        plot_bgcolor="#f8f9fb",
        paper_bgcolor="white",
    )
    return fig


# ============================================================
# PDF GENERATION
# ============================================================
def _strip_emoji(text: str) -> str:
    """Remove non-ASCII characters (emoji) so reportlab Helvetica doesn't choke."""
    return re.sub(r'[^\x00-\x7F]+', '', str(text)).strip()


def generate_pdf(
    data: dict,
    hist_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    eps_estimated: float,
    bvps_input: float,
    per_industri: float,
    pbv_industri: float,
    cagr: float | None,
    cp: float | None,
    fig: go.Figure | None,
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle,
        Paragraph, Spacer, Image, HRFlowable,
    )

    NAVY  = colors.HexColor('#1a5276')
    GRAY  = colors.HexColor('#cccccc')
    LIGHT = colors.HexColor('#f0f4f8')
    GREEN = colors.HexColor('#1e8449')
    RED   = colors.HexColor('#c0392b')

    buffer = BytesIO()
    W_PAGE, _ = A4
    MARGIN = 2 * cm
    W = W_PAGE - 2 * MARGIN

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=MARGIN, leftMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    base  = getSampleStyleSheet()
    title_s   = ParagraphStyle('t', parent=base['Title'],   fontSize=15, textColor=NAVY, spaceAfter=2)
    heading_s = ParagraphStyle('h', parent=base['Heading2'], fontSize=10, textColor=NAVY, spaceAfter=3, spaceBefore=8)
    normal_s  = base['Normal']
    small_s   = ParagraphStyle('s', parent=base['Normal'],  fontSize=7.5, textColor=colors.grey)

    def tbl_style(extra=None):
        cmds = [
            ('BACKGROUND',   (0, 0), (-1, 0),  NAVY),
            ('TEXTCOLOR',    (0, 0), (-1, 0),  colors.white),
            ('FONTNAME',     (0, 0), (-1, 0),  'Helvetica-Bold'),
            ('FONTSIZE',     (0, 0), (-1, -1), 8.5),
            ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID',         (0, 0), (-1, -1), 0.4, GRAY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
            ('TOPPADDING',   (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
        ]
        if extra:
            cmds.extend(extra)
        return TableStyle(cmds)

    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    sub = f" · {data['sector']} / {data['industry']}" if data.get('sector') else ''
    story.append(Paragraph("Perkiraan Rentang Harga Saham (PRHS)", title_s))
    story.append(Paragraph(f"<b>{data['company_name']}</b>{sub}", normal_s))
    story.append(Paragraph(
        f"Tanggal analisis: {datetime.now().strftime('%d %B %Y')}  ·  Data: Yahoo Finance / IDX",
        small_s,
    ))
    story.append(HRFlowable(width=W, thickness=1, color=NAVY, spaceAfter=6))

    # ── Key Metrics ───────────────────────────────────────────────────────────
    story.append(Paragraph("Metrik Utama", heading_s))
    eps_v = data['current_eps']
    bv_v  = data['current_bvps']
    met_data = [
        ["Harga Saat Ini", "EPS (TTM)", "BVPS", "PER Saat Ini", "PBV Saat Ini"],
        [
            fmt_rp_short(cp)   if cp  is not None else "N/A",
            fmt_rp(eps_v)      if eps_v is not None else "N/A",
            fmt_rp(bv_v)       if bv_v  is not None else "N/A",
            f"{cp/eps_v:.2f}x" if (cp is not None and eps_v is not None and eps_v > 0) else "N/A",
            f"{cp/bv_v:.2f}x"  if (cp is not None and bv_v  is not None and bv_v  > 0) else "N/A",
        ],
    ]
    story.append(Table(met_data, colWidths=[W/5]*5, style=tbl_style()))
    story.append(Spacer(1, 0.3*cm))

    # ── Historical Data ───────────────────────────────────────────────────────
    story.append(Paragraph("Data Historis 5 Tahun Terakhir", heading_s))
    hist_headers = ["Tahun", "H. Tertinggi", "H. Terendah", "EPS", "BVPS",
                    "PER Tert.", "PER Ter.", "PBV Tert.", "PBV Ter."]
    hist_rows = [hist_headers]
    for _, row in hist_df.iterrows():
        hist_rows.append([
            str(int(row["Tahun"])),
            fmt_rp_short(row["Harga Tertinggi"]) if pd.notna(row["Harga Tertinggi"]) else "N/A",
            fmt_rp_short(row["Harga Terendah"])  if pd.notna(row["Harga Terendah"])  else "N/A",
            fmt_rp(row["EPS"])  if pd.notna(row["EPS"])  else "N/A",
            fmt_rp(row["BVPS"]) if pd.notna(row["BVPS"]) else "N/A",
            f"{row['PER Tertinggi']:.2f}x" if pd.notna(row["PER Tertinggi"]) else "N/A",
            f"{row['PER Terendah']:.2f}x"  if pd.notna(row["PER Terendah"])  else "N/A",
            f"{row['PBV Tertinggi']:.2f}x" if pd.notna(row["PBV Tertinggi"]) else "N/A",
            f"{row['PBV Terendah']:.2f}x"  if pd.notna(row["PBV Terendah"])  else "N/A",
        ])
    cw = W / len(hist_headers)
    story.append(Table(hist_rows, colWidths=[cw]*len(hist_headers), style=tbl_style()))
    story.append(Spacer(1, 0.3*cm))

    # ── Parameters ───────────────────────────────────────────────────────────
    story.append(Paragraph("Parameter Estimasi", heading_s))
    cagr_str = f"{cagr*100:.1f}% / tahun" if cagr is not None else "N/A"
    par_data = [
        ["Parameter", "Nilai"],
        ["CAGR EPS Historis", cagr_str],
        ["EPS Estimasi Tahun Depan", fmt_rp(eps_estimated)],
        ["BVPS Saat Ini (input)", fmt_rp(bvps_input)],
        ["Rata-rata P/E Ratio Industri", f"{per_industri:.2f}x" if per_industri > 0 else "Tidak diisi"],
        ["Price to BV Industri",         f"{pbv_industri:.2f}x" if pbv_industri > 0 else "Tidak diisi"],
    ]
    story.append(Table(par_data, colWidths=[W*0.55, W*0.45], style=tbl_style()))
    story.append(Spacer(1, 0.3*cm))

    # ── Scenarios ────────────────────────────────────────────────────────────
    story.append(Paragraph(f"Perkiraan Rentang Harga — Tahun {datetime.now().year + 1}", heading_s))
    scen_rows = [["Skenario", "Batas Atas (Rp)", "Batas Bawah (Rp)", "Formula / Keterangan"]]
    for _, row in scenarios_df.iterrows():
        ba = row["Batas Atas"]
        bb = row["Batas Bawah"]
        scen_rows.append([
            _strip_emoji(row["Skenario"]),
            fmt_rp_short(ba) if (ba is not None and pd.notna(ba)) else "N/A",
            fmt_rp_short(bb) if (bb is not None and pd.notna(bb)) else "N/A",
            str(row.get("_note", "")),
        ])
    story.append(Table(
        scen_rows,
        colWidths=[W*0.22, W*0.17, W*0.17, W*0.44],
        style=tbl_style(extra=[('ALIGN', (0, 1), (0, -1), 'LEFT'),
                                ('ALIGN', (3, 1), (3, -1), 'LEFT'),
                                ('FONTSIZE', (3, 1), (3, -1), 7.5)]),
    ))
    story.append(Spacer(1, 0.3*cm))

    # ── MoS ──────────────────────────────────────────────────────────────────
    netral_row = scenarios_df[scenarios_df["Skenario"].str.contains("Netral")]
    if not netral_row.empty and cp is not None:
        nu = netral_row.iloc[0]["Batas Atas"]
        nl = netral_row.iloc[0]["Batas Bawah"]
        if nu is not None and pd.notna(nu):
            mos_u = (nu - cp) / nu * 100
            mos_l = (nl - cp) / nl * 100 if (nl is not None and nl != 0) else None
            story.append(Paragraph("Margin of Safety (MoS) — Estimasi Netral", heading_s))
            mos_data = [["Harga Saat Ini", "MoS dari Batas Atas Netral", "MoS dari Batas Bawah Netral"]]
            mos_data.append([
                fmt_rp_short(cp),
                f"{mos_u:.1f}% ({'Undervalued' if mos_u > 0 else 'Overvalued'})",
                f"{mos_l:.1f}% ({'Undervalued' if mos_l > 0 else 'Overvalued'})" if mos_l is not None else "N/A",
            ])
            extra_mos = [
                ('TEXTCOLOR', (1, 1), (1, 1), GREEN if mos_u > 0 else RED),
                ('TEXTCOLOR', (2, 1), (2, 1), (GREEN if mos_l > 0 else RED) if mos_l is not None else colors.grey),
                ('FONTNAME',  (1, 1), (2, 1), 'Helvetica-Bold'),
            ]
            story.append(Table(mos_data, colWidths=[W/3]*3, style=tbl_style(extra=extra_mos)))
            story.append(Spacer(1, 0.3*cm))

    # ── Chart ─────────────────────────────────────────────────────────────────
    if fig is not None:
        try:
            import plotly.io as pio
            img_bytes = pio.to_image(fig, format="png", width=750, height=380, scale=1.5)
            chart_img = Image(BytesIO(img_bytes), width=W, height=W * 0.5)
            story.append(Paragraph("Visualisasi Rentang Harga", heading_s))
            story.append(chart_img)
            story.append(Spacer(1, 0.3*cm))
        except Exception:
            pass  # kaleido not installed — skip chart

    # ── Disclaimer ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=GRAY, spaceAfter=4))
    story.append(Paragraph(
        "Dokumen ini dibuat otomatis oleh PRHS App. Bukan merupakan rekomendasi investasi. "
        "Data bersumber dari Yahoo Finance dan dapat berbeda dengan data IDX resmi.",
        small_s,
    ))

    doc.build(story)
    return buffer.getvalue()


# ============================================================
# MAIN APP
# ============================================================
st.title("📈 Perkiraan Rentang Harga Saham (PRHS)")
st.markdown(
    "**Humanis PRHS** — Estimasi harga saham tahun depan berdasarkan data historis 5 tahun | "
    "Data otomatis dari Yahoo Finance · IDX"
)
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────
col_a, col_b, _ = st.columns([3, 1, 3])  # third column is intentional whitespace
with col_a:
    ticker_raw = st.text_input(
        "Kode Saham",
        placeholder="Contoh: BBCA  /  TLKM  /  BBRI",
        help="Kode saham IDX. Suffix .JK ditambahkan otomatis.",
        key="ticker_input",
    )
with col_b:
    st.write("")
    st.write("")
    fetch_btn = st.button("🔍 Ambil Data", type="primary", use_container_width=True)

# ── Session State ─────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data     = None
    st.session_state.hist_df  = None
    st.session_state.symbol   = None

if fetch_btn and ticker_raw.strip():
    symbol = normalize_ticker(ticker_raw)
    with st.spinner(f"Mengambil data {symbol} …"):
        try:
            data = fetch_stock_data(symbol)
            hist_df = build_hist_table(
                data["annual_prices"],
                data["eps_by_year"],
                data["bvps_by_year"],
            )
            st.session_state.data    = data
            st.session_state.hist_df = hist_df
            st.session_state.symbol  = symbol
            st.success(f"✅ Data **{data['company_name']}** berhasil diambil!")
        except Exception as e:
            st.error(f"❌ {e}")
            st.session_state.data = None

# ── Display ────────────────────────────────────────────────────────────────
if st.session_state.data is not None:
    data    = st.session_state.data
    hist_df = st.session_state.hist_df.copy()

    # Company header
    sub = f" · {data['sector']} / {data['industry']}" if data.get("sector") else ""
    st.subheader(f"📊 {data['company_name']}{sub}")

    # Key metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    cp  = data["current_price"]
    eps = data["current_eps"]
    bv  = data["current_bvps"]

    m1.metric("Harga Saat Ini",  fmt_rp_short(cp)   if cp  is not None else "N/A")
    m2.metric("EPS (TTM)",       fmt_rp(eps)         if eps is not None else "N/A")
    m3.metric("BVPS",            fmt_rp(bv)          if bv  is not None else "N/A")
    m4.metric("PER Saat Ini",    f"{cp/eps:.2f}x"    if (cp is not None and eps is not None and eps > 0) else "N/A")
    m5.metric("PBV Saat Ini",    f"{cp/bv:.2f}x"     if (cp is not None and bv  is not None and bv  > 0) else "N/A")

    st.divider()

    # ── Historical Data Table (Editable) ─────────────────────────────────
    st.subheader("📅 Data Historis 5 Tahun Terakhir")
    st.caption(
        "💡 EPS dan BVPS diambil otomatis dari laporan keuangan. "
        "Anda dapat mengedit nilainya jika data belum tersedia atau perlu koreksi."
    )

    edited_df = st.data_editor(
        hist_df,
        column_config={
            "Tahun":           st.column_config.NumberColumn("Tahun", disabled=True, format="%d"),
            "Harga Tertinggi": st.column_config.NumberColumn("Harga Tertinggi (Rp)", format="%.0f", disabled=True),
            "Harga Terendah":  st.column_config.NumberColumn("Harga Terendah (Rp)",  format="%.0f", disabled=True),
            "EPS":             st.column_config.NumberColumn("EPS ✏️",   format="%.2f", help="Earnings Per Share — edit jika perlu"),
            "BVPS":            st.column_config.NumberColumn("BVPS ✏️",  format="%.2f", help="Book Value Per Share — edit jika perlu"),
            "PER Tertinggi":   st.column_config.NumberColumn("PER Tertinggi", format="%.2f", disabled=True),
            "PER Terendah":    st.column_config.NumberColumn("PER Terendah",  format="%.2f", disabled=True),
            "PBV Tertinggi":   st.column_config.NumberColumn("PBV Tertinggi", format="%.2f", disabled=True),
            "PBV Terendah":    st.column_config.NumberColumn("PBV Terendah",  format="%.2f", disabled=True),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="hist_editor",
    )

    # Recalculate derived columns after any edit
    edited_df = recalc_derived(edited_df)

    # ── Peringatan EPS negatif ─────────────────────────────────────────────
    neg_eps_years = [
        int(row["Tahun"]) for _, row in edited_df.iterrows()
        if pd.notna(row["EPS"]) and row["EPS"] < 0
    ]
    if neg_eps_years:
        years_str = ", ".join(str(y) for y in sorted(neg_eps_years))
        st.warning(
            f"⚠️ **EPS Negatif Terdeteksi** — tahun: **{years_str}**\n\n"
            "Metode PRHS mengasumsikan EPS selalu positif. "
            "Tahun dengan EPS negatif **dikecualikan** dari perhitungan CAGR dan skenario harga. "
            "Hasil estimasi mungkin **kurang akurat** — pertimbangkan untuk memverifikasi "
            "data dari laporan keuangan resmi di IDX.co.id."
        )

    st.divider()

    # ── Parameters ────────────────────────────────────────────────────────
    st.subheader("⚙️ Parameter Estimasi")

    eps_hist = {
        int(row["Tahun"]): row["EPS"]
        for _, row in edited_df.iterrows()
        if pd.notna(row["EPS"]) and row["EPS"] > 0
    }
    cagr, eps_auto = estimate_eps_growth(eps_hist)
    latest_eps = None
    if eps_hist:
        latest_year = max(eps_hist.keys())
        latest_eps  = eps_hist[latest_year]

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**EPS & BVPS**")
        if cagr is not None and eps_auto is not None:
            st.info(
                f"📈 CAGR EPS historis: **{cagr*100:.1f}% / tahun**  |  "
                f"EPS terakhir: **{fmt_rp(latest_eps)}**  →  "
                f"EPS est. otomatis: **{fmt_rp(eps_auto)}**"
            )
        else:
            st.warning("⚠️ Data EPS historis tidak lengkap — masukkan EPS estimated secara manual.")

        eps_estimated = st.number_input(
            "EPS Estimasi Tahun Depan (Rp) ✏️",
            min_value=0.0,
            value=float(round(eps_auto, 2)) if (eps_auto is not None and eps_auto > 0) else 0.0,
            step=10.0,
            format="%.2f",
            help="Dihitung otomatis dengan CAGR historis. Anda bisa mengubahnya.",
        )

        bvps_input = st.number_input(
            "BVPS Saat Ini (Rp) ✏️",
            min_value=0.0,
            value=float(round(data["current_bvps"], 2)) if data["current_bvps"] is not None else 0.0,
            step=10.0,
            format="%.2f",
            help="Book Value Per Share dari laporan keuangan terakhir.",
        )

    with col_right:
        st.markdown(
            "**Data Industri**  "
            "*(IDX.co.id → Data Pasar → Laporan Statistik → Digital Statistik → "
            "Summary Financial Ratio by Industry)*"
        )
        st.markdown('<p style="color:red; font-weight:600; margin-bottom:0;">Rata-rata P/E Ratio Industri ✏️</p>', unsafe_allow_html=True)
        per_industri = st.number_input(
            "Rata-rata P/E Ratio Industri",
            min_value=0.0,
            value=0.0,
            step=0.5,
            format="%.2f",
            help="Price/Earnings Ratio rata-rata industri. Isi manual dari IDX.",
            label_visibility="collapsed",
        )
        st.markdown('<p style="color:red; font-weight:600; margin-bottom:0;">Price to BV Industri ✏️</p>', unsafe_allow_html=True)
        pbv_industri = st.number_input(
            "Price to BV Industri",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            help="Price/Book Value rata-rata industri. Isi manual dari IDX.",
            label_visibility="collapsed",
        )

    st.divider()

    # ── Output Table ──────────────────────────────────────────────────────
    st.subheader(f"🎯 Perkiraan Rentang Harga Saham — Tahun {datetime.now().year + 1}")

    if eps_estimated > 0:
        scenarios_df = calculate_scenarios(
            df=edited_df,
            eps_est=eps_estimated,
            bvps_current=bvps_input if bvps_input > 0 else None,
            per_industri=per_industri if per_industri > 0 else None,
            pbv_industri=pbv_industri if pbv_industri > 0 else None,
        )

        # ── Table display
        display_df = scenarios_df.copy()
        display_df["Batas Atas (Rp)"]  = display_df["Batas Atas"].apply(fmt_rp_short)
        display_df["Batas Bawah (Rp)"] = display_df["Batas Bawah"].apply(fmt_rp_short)
        display_df["Formula / Keterangan"] = display_df["_note"]

        st.dataframe(
            display_df[["Skenario", "Batas Atas (Rp)", "Batas Bawah (Rp)", "Formula / Keterangan"]],
            use_container_width=True,
            hide_index=True,
        )

        # ── MoS section
        netral_row = scenarios_df[scenarios_df["Skenario"].str.contains("Netral")]
        if not netral_row.empty:
            netral_upper = netral_row.iloc[0]["Batas Atas"]
            netral_lower = netral_row.iloc[0]["Batas Bawah"]
            if cp is not None and netral_upper is not None and pd.notna(netral_upper):
                mos_upper = (netral_upper - cp) / netral_upper * 100
                mos_lower = (netral_lower - cp) / netral_lower * 100 if (netral_lower is not None and netral_lower != 0) else None

                mos_col1, mos_col2, mos_col3 = st.columns(3)
                mos_col1.metric("Harga Saat Ini", fmt_rp_short(cp))
                color_u = "normal" if mos_upper > 0 else "inverse"
                mos_col2.metric(
                    "MoS dari Batas Atas Netral",
                    f"{mos_upper:.1f}%",
                    delta=f"{'Undervalued' if mos_upper > 0 else 'Overvalued'}",
                    delta_color=color_u,
                )
                if mos_lower is not None:
                    color_l = "normal" if mos_lower > 0 else "inverse"
                    mos_col3.metric(
                        "MoS dari Batas Bawah Netral",
                        f"{mos_lower:.1f}%",
                        delta=f"{'Undervalued' if mos_lower > 0 else 'Overvalued'}",
                        delta_color=color_l,
                    )

        # ── Chart
        st.markdown("#### Visualisasi Rentang Harga")
        fig = build_range_chart(scenarios_df, cp)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # ── PDF Download ──────────────────────────────────────────────────────
        st.divider()
        ticker_clean = (st.session_state.get("symbol") or "SAHAM").replace(".JK", "")
        pdf_filename = f"{ticker_clean}_PRHS_{datetime.now().strftime('%Y%m%d')}.pdf"

        # Auto-generate PDF ketika parameter berubah (cache by key)
        pdf_cache_key = (
            f"{st.session_state.get('symbol')}|{eps_estimated}|{bvps_input}"
            f"|{per_industri}|{pbv_industri}|{hash(edited_df.to_json())}"
        )
        if st.session_state.get("_pdf_key") != pdf_cache_key:
            with st.spinner("Menyiapkan laporan PDF …"):
                try:
                    st.session_state._pdf_bytes = generate_pdf(
                        data=data,
                        hist_df=edited_df,
                        scenarios_df=scenarios_df,
                        eps_estimated=eps_estimated,
                        bvps_input=bvps_input,
                        per_industri=per_industri,
                        pbv_industri=pbv_industri,
                        cagr=cagr,
                        cp=cp,
                        fig=fig,
                    )
                    st.session_state._pdf_key = pdf_cache_key
                except ImportError:
                    st.session_state._pdf_bytes = None
                    st.warning("⚠️ PDF tidak tersedia — install `reportlab`: `pip install reportlab`")
                except Exception as pdf_err:
                    st.session_state._pdf_bytes = None
                    st.warning(f"⚠️ Gagal membuat PDF: {pdf_err}")

        if st.session_state.get("_pdf_bytes"):
            st.download_button(
                "⬇️ Download as PDF",
                data=st.session_state._pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

    else:
        st.warning("⚠️ Masukkan **EPS Estimasi Tahun Depan** untuk menghitung perkiraan harga.")

    # ── Footer notes
    st.divider()
    st.caption(
        "📌 **Catatan:**  \n"
        "- Valid hanya untuk saham dengan **EPS selalu positif** dalam 5 tahun terakhir.  \n"
        "- Data harga historis dan EPS dari **Yahoo Finance** — dapat berbeda dengan data IDX resmi.  \n"
        "- Untuk akurasi lebih tinggi, verifikasi EPS dari laporan keuangan di "
        "[IDX.co.id](https://www.idx.co.id) atau [RTI.co.id](https://www.rti.co.id).  \n"
        "- PER & PBV Industri: IDX.co.id → Data Pasar → Laporan Statistik → Digital Statistik → "
        "Listed Company → Summary Financial Ratio by Industry.  \n"
        "- Aplikasi ini **bukan merupakan rekomendasi investasi**."
    )

    # ── Reset Button ──────────────────────────────────────────────────────
    st.divider()
    if st.button("🔄 Reset — Mulai Analisis Baru", type="secondary", use_container_width=True):
        for key in ["data", "hist_df", "symbol", "hist_editor", "_pdf_bytes", "_pdf_key"]:
            st.session_state.pop(key, None)
        st.session_state["ticker_input"] = ""
        st.rerun()

else:
    # Landing state
    st.info("👆 Masukkan kode saham IDX di atas (contoh: **BBCA**, **TLKM**, **BBRI**) lalu klik **Ambil Data**.")

    with st.expander("ℹ️ Cara Menggunakan Aplikasi"):
        st.markdown("""
**Langkah-langkah:**
1. **Masukkan kode saham** IDX (tanpa .JK) → klik *Ambil Data*
2. **Periksa tabel historis** — edit EPS/BVPS jika data belum tersedia atau salah
3. **Sesuaikan EPS Estimasi** tahun depan (otomatis dari CAGR, bisa diedit)
4. **Isi PER & PBV Industri** dari IDX.co.id untuk skenario Rerata Industri
5. **Baca tabel output** — 4 skenario perkiraan rentang harga

**Formula per skenario:**
| Skenario | Batas Atas | Batas Bawah |
|---|---|---|
| 🚀 Optimis | max(PER Tertinggi 5 thn) × EPS Est. | max(PER Terendah 5 thn) × EPS Est. |
| ⚖️ Netral | avg(PER Tertinggi 5 thn) × EPS Est. | avg(PER Terendah 5 thn) × EPS Est. |
| 📚 Rerata BV | avg(PBV Tertinggi 5 thn) × BVPS | avg(PBV Terendah 5 thn) × BVPS |
| 🏭 Rerata PER Industri | PER Industri × EPS Est. | *(nilai tunggal)* |
| 📖 Rerata PBV Industri | PBV Industri × BVPS | *(nilai tunggal)* |
        """)
