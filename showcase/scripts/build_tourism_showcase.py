"""Destination Performance & Tourism Growth Dashboard.

This script reproduces the core analysis used in the portfolio showcase.
It combines official Singapore tourism datasets with a transparent synthetic
Sentosa-style destination layer.

Source data labels:
1. Official public data: International Visitor Arrivals by Place of Residence, Monthly
   Dataset ID: d_d1c33009b674dcf70b8e8c790b793f28
   Link: https://data.gov.sg/datasets/d_d1c33009b674dcf70b8e8c790b793f28/view
2. Official public data: Monthly Hotel Statistics
   Dataset ID: d_8e62605f0c1c948702b6ea0fe45242d3
   Link: https://data.gov.sg/datasets/d_8e62605f0c1c948702b6ea0fe45242d3/view
3. Official public data: Tourism Receipts by Major Components, Year-to-Date Quarterly
   Dataset ID: d_248d4c6574b5ac87cd31851ed3f697d6
   Link: https://data.gov.sg/datasets/d_248d4c6574b5ac87cd31851ed3f697d6/view
4. Synthetic destination data: destination visitors, revenue, campaign ROI and
   guest feedback generated inside this script because attraction-level SDC-style
   operating data is not publicly available.
"""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tourism_analytics_data"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "visitor_arrivals_monthly": {
        "id": "d_d1c33009b674dcf70b8e8c790b793f28",
        "label": "International Visitor Arrivals by Place of Residence, Monthly",
        "url": "https://data.gov.sg/datasets/d_d1c33009b674dcf70b8e8c790b793f28/view",
    },
    "hotel_statistics_monthly": {
        "id": "d_8e62605f0c1c948702b6ea0fe45242d3",
        "label": "Monthly Hotel Statistics",
        "url": "https://data.gov.sg/datasets/d_8e62605f0c1c948702b6ea0fe45242d3/view",
    },
    "tourism_receipts_ytd_quarterly": {
        "id": "d_248d4c6574b5ac87cd31851ed3f697d6",
        "label": "Tourism Receipts by Major Components (Year-to-Date), Quarterly",
        "url": "https://data.gov.sg/datasets/d_248d4c6574b5ac87cd31851ed3f697d6/view",
    },
}

MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
REGION_ROWS = {"Total International Visitor Arrivals By Place Of Residence", "Southeast Asia", "Greater China", "North Asia", "South Asia", "West Asia", "Europe", "Americas", "Oceania", "Africa", "Others"}


def api_url(resource_id: str, limit: int = 5000, offset: int = 0) -> str:
    query = urllib.parse.urlencode({"resource_id": resource_id, "limit": limit, "offset": offset})
    return f"https://data.gov.sg/api/action/datastore_search?{query}"


def fetch_resource(resource_id: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        with urllib.request.urlopen(api_url(resource_id, offset=offset), timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError(payload)
        batch = payload["result"]["records"]
        rows.extend(batch)
        if len(rows) >= payload["result"].get("total", len(rows)) or not batch:
            break
        offset += len(batch)
    return pd.DataFrame(rows)


def parse_month(col: str):
    if len(col) < 7 or not col[:4].isdigit() or col[4:] not in MONTHS:
        return None
    return pd.Timestamp(year=int(col[:4]), month=MONTHS[col[4:]], day=1)


def wide_monthly_to_long(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    month_cols = [c for c in df.columns if parse_month(c) is not None]
    long = df.melt(id_vars=["DataSeries"], value_vars=month_cols, var_name="period_raw", value_name=value_col)
    long["date"] = long["period_raw"].map(parse_month)
    long["series"] = long["DataSeries"].astype(str).str.strip()
    long[value_col] = pd.to_numeric(long[value_col], errors="coerce")
    return long[["date", "series", value_col]].dropna()


def prepare_arrivals(raw: pd.DataFrame):
    arrivals = wide_monthly_to_long(raw, "arrivals")
    total = arrivals[arrivals["series"].eq("Total International Visitor Arrivals By Place Of Residence")].copy()
    total["year"] = total["date"].dt.year
    countries = arrivals[~arrivals["series"].isin(REGION_ROWS)].copy()
    countries["year"] = countries["date"].dt.year
    return total, countries


def prepare_hotel(raw: pd.DataFrame) -> pd.DataFrame:
    hotel = wide_monthly_to_long(raw, "value").pivot_table(index="date", columns="series", values="value", aggfunc="first").reset_index()
    hotel.columns.name = None
    hotel["year"] = hotel["date"].dt.year
    return hotel


def make_synthetic_destination(total: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    random.seed(42)
    base = total[(total["date"] >= "2023-01-01") & (total["date"] <= "2026-03-01")].copy().sort_values("date")
    top_markets = countries[countries["year"].eq(2025)].groupby("series")["arrivals"].sum().sort_values(ascending=False).head(6).index.tolist()
    rows = []
    for row in base.itertuples(index=False):
        season = 1.10 if row.date.month in [7, 8, 12] else 1.04 if row.date.month in [3, 6, 11] else 0.96
        visitors = int(row.arrivals * random.uniform(0.125, 0.155) * season)
        tourist_share = 0.48 + min(0.13, (row.arrivals - base["arrivals"].min()) / (base["arrivals"].max() - base["arrivals"].min()) * 0.16)
        tourist_visitors = int(visitors * tourist_share)
        local_visitors = visitors - tourist_visitors
        revenue = visitors * random.uniform(72, 91)
        campaign_spend = random.uniform(180000, 360000) * (1.45 if row.date.month in [6, 11, 12] else 1.0)
        campaign_roi = random.uniform(1.8, 4.4) * (1.18 if row.date.month in [6, 11, 12] else 1.0)
        crowd_penalty = max(0, visitors - base["arrivals"].median() * 0.145) / 40000
        nps = max(24, min(68, random.gauss(52, 5) - crowd_penalty * 4))
        negative_feedback = max(8, min(32, random.gauss(18, 4) + crowd_penalty * 2.3))
        rows.append({
            "date": row.date,
            "destination_visitors": visitors,
            "tourist_visitors": tourist_visitors,
            "local_visitors": local_visitors,
            "tourist_share": tourist_share,
            "total_revenue": revenue,
            "revenue_per_visitor": revenue / visitors,
            "campaign_spend": campaign_spend,
            "campaign_roi": campaign_roi,
            "nps": nps,
            "negative_feedback_pct": negative_feedback,
            "priority_market": random.choice(top_markets[:4]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    raw = {name: fetch_resource(meta["id"]) for name, meta in DATASETS.items()}
    arrivals_total, arrivals_by_market = prepare_arrivals(raw["visitor_arrivals_monthly"])
    hotel = prepare_hotel(raw["hotel_statistics_monthly"])
    destination = make_synthetic_destination(arrivals_total, arrivals_by_market)

    annual = arrivals_total.groupby("year", as_index=False)["arrivals"].sum()
    arrivals_2019 = float(annual.loc[annual["year"].eq(2019), "arrivals"].iloc[0])
    arrivals_2025 = float(annual.loc[annual["year"].eq(2025), "arrivals"].iloc[0])
    recovery = arrivals_2025 / arrivals_2019 * 100

    top_markets = arrivals_by_market[arrivals_by_market["year"].eq(2025)].groupby("series", as_index=False)["arrivals"].sum().sort_values("arrivals", ascending=False)
    top5_share = top_markets.head(5)["arrivals"].sum() / top_markets["arrivals"].sum() * 100

    hotel_2025 = hotel[hotel["year"].eq(2025)]["Room Revenue"].sum()
    hotel_2019 = hotel[hotel["year"].eq(2019)]["Room Revenue"].sum()
    hotel_recovery = hotel_2025 / hotel_2019 * 100

    summary = {
        "kpis": {
            "2025 arrivals": f"{arrivals_2025 / 1_000_000:.1f}m",
            "recovery_vs_2019": f"{recovery:.0f}%",
            "top_5_market_share": f"{top5_share:.0f}%",
            "hotel_revenue_vs_2019": f"{hotel_recovery:.0f}%",
        },
        "top_source_markets_2025": top_markets.head(10).to_dict(orient="records"),
        "sdc_management_questions": [
            "Which source markets should SDC prioritize for growth?",
            "Is tourism recovery translating into destination value, not just footfall?",
            "Which under-recovered markets show campaign upside?",
            "What guest experience risks could constrain repeat visitation?",
        ],
        "sources": DATASETS,
        "synthetic_data_note": "Destination-level revenue, campaign ROI and guest feedback are synthetic and used only to demonstrate analytical approach.",
    }

    arrivals_total.to_csv(OUT / "singapore_monthly_arrivals_total.csv", index=False)
    top_markets.head(10).to_csv(OUT / "top_source_markets_2025.csv", index=False)
    destination.to_csv(OUT / "synthetic_destination_monthly_kpis.csv", index=False)
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
