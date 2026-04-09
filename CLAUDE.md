# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stock valuation tool for the Indonesian stock market using the **PRHS (Perkiraan Rentang Harga Saham — Estimated Stock Price Range)** methodology.

**Files:**
- `Calc_PRHS.xlsx` — The main calculation spreadsheet containing all valuation formulas
- `lay_outPRHS.jpeg` — Visual reference showing the tool's interface layout

## PRHS Methodology

The spreadsheet calculates stock price ranges based on:

- **Inputs**: Stock name, current price, BVPS (profit margin), share count, PER, PBV industry benchmarks, last 5 years of EPS data, and last 5 years of high/low price history
- **Outputs**: Price range estimates using RERATA (average) and TERTINGGI (highest/optimal) methods, ROE and ROE Adjusted values, Margin of Safety (MoS) from Tertinggi Harga Netral, and price ranges from PER and PBV industry comparisons

**Data sources referenced in the tool:**
- [IDX.co.id](https://idx.co.id) (Indonesian Stock Exchange) — financial reports and statistics
- Tradingyates.com — historical stock price data

## Python App (Streamlit)

`app.py` — single-file Streamlit app implementing the PRHS methodology.

**Run the app:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Dependencies:** `streamlit`, `yfinance`, `pandas`, `numpy`, `plotly`

**App flow:**
1. User enters an IDX ticker (e.g. `BBCA`) — `.JK` suffix added automatically
2. `yfinance` fetches: current price, EPS (TTM), BVPS, 5-year daily OHLC → resampled to annual High/Low, annual EPS from `ticker.financials`, annual BVPS from `ticker.balance_sheet`
3. User can edit EPS/BVPS in the historical table if yfinance data is missing
4. EPS Estimated is auto-calculated using CAGR of historical EPS (editable)
5. User manually enters PER Industri and PBV Industri

**Output table — 4 scenarios:**
| Skenario | Batas Atas | Batas Bawah |
|---|---|---|
| Estimasi Optimis | max(PER Tertinggi 5 thn) × EPS Est. | max(PER Terendah 5 thn) × EPS Est. |
| Estimasi Netral | avg(PER Tertinggi 5 thn) × EPS Est. | avg(PER Terendah 5 thn) × EPS Est. |
| Rerata BV | avg(PBV Tertinggi hist.) × BVPS | avg(PBV Terendah hist.) × BVPS |
| Rerata PER Industri | PER Industri × EPS Est. | *(nilai tunggal)* |
| Rerata PBV Industri | PBV Industri × BVPS | *(nilai tunggal)* |

Where `PER Tertinggi[y] = Harga Tertinggi[y] / EPS[y]` and `PBV Tertinggi[y] = Harga Tertinggi[y] / BVPS[y]`.
