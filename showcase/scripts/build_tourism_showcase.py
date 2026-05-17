from __future__ import annotations

import json
import math
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ASSET_DIR = ROOT / "assets"

DATASETS = {
    "visitor_arrivals_monthly": {
        "id": "d_d1c33009b674dcf70b8e8c790b793f28",
        "label": "International Visitor Arrivals by Place of Residence, Monthly",
        "url": "https://data.gov.sg/datasets?query=International+Visitor+Arrivals+by+Place+of+Residence,+Monthly&resultId=d_d1c33009b674dcf70b8e8c790b793f28",
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

# Source data labels used in the website read-up and generated README.
# Public data is Singapore-level tourism demand and benchmarking data.
# Destination-level operating metrics are synthetic and are generated below
# because SDC-style attraction revenue, guest feedback and campaign ROAS are
# not publicly available at destination level.
SOURCE_DATA_LABELS = {
    "PUBLIC_1": "Official public data: International Visitor Arrivals by Place of Residence, Monthly",
    "PUBLIC_2": "Official public data: Monthly Hotel Statistics",
    "PUBLIC_3": "Official public data: Tourism Receipts by Major Components, Year-to-Date Quarterly",
    "SYNTHETIC_1": "Synthetic destination data: destination visitors, revenue, campaign ROAS and guest feedback",
}

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

COLORS = {
    "ink": "#17202a",
    "muted": "#68737d",
    "line": "#d8dee5",
    "blue": "#1f77b4",
    "teal": "#129c9a",
    "green": "#2b8a3e",
    "amber": "#d4860b",
    "red": "#c0392b",
    "purple": "#6f4dbf",
    "bg": "#f7f9fb",
    "panel": "#ffffff",
}


def ensure_dirs() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, ASSET_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def datastore_url(resource_id: str, limit: int = 5000, offset: int = 0) -> str:
    params = urllib.parse.urlencode(
        {"resource_id": resource_id, "limit": limit, "offset": offset}
    )
    return f"https://data.gov.sg/api/action/datastore_search?{params}"


def fetch_datastore(resource_id: str) -> pd.DataFrame:
    records: list[dict] = []
    offset = 0
    limit = 5000
    while True:
        with urllib.request.urlopen(datastore_url(resource_id, limit, offset), timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError(f"Failed to fetch resource {resource_id}: {payload}")
        batch = payload["result"]["records"]
        records.extend(batch)
        total = payload["result"].get("total", len(records))
        if len(records) >= total or not batch:
            break
        offset += limit
    return pd.DataFrame(records)


def save_raw() -> dict[str, pd.DataFrame]:
    raw = {}
    for key, meta in DATASETS.items():
        df = fetch_datastore(meta["id"])
        raw[key] = df
        df.to_csv(RAW_DIR / f"{key}.csv", index=False)
    return raw


def parse_month_column(col: str) -> pd.Timestamp | None:
    if len(col) < 7:
        return None
    year = col[:4]
    mon = col[4:]
    if not year.isdigit() or mon not in MONTHS:
        return None
    return pd.Timestamp(year=int(year), month=MONTHS[mon], day=1)


def wide_monthly_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    month_cols = [c for c in df.columns if parse_month_column(c) is not None]
    long = df.melt(
        id_vars=["DataSeries"],
        value_vars=month_cols,
        var_name="period_raw",
        value_name=value_name,
    )
    long["date"] = long["period_raw"].map(parse_month_column)
    long["series"] = long["DataSeries"].astype(str).str.strip()
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long[["date", "series", value_name]].dropna(subset=["date", value_name])


def prepare_arrivals(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arrivals = wide_monthly_to_long(raw, "arrivals")
    total = arrivals[
        arrivals["series"].eq("Total International Visitor Arrivals By Place Of Residence")
    ].copy()
    total["year"] = total["date"].dt.year
    total["month"] = total["date"].dt.month
    total["month_name"] = total["date"].dt.strftime("%b")

    countries = arrivals[
        ~arrivals["series"].isin(
            [
                "Total International Visitor Arrivals By Place Of Residence",
                "Southeast Asia",
                "Greater China",
                "North Asia",
                "South Asia",
                "West Asia",
                "Europe",
                "Americas",
                "Oceania",
                "Africa",
                "Others",
            ]
        )
    ].copy()
    countries["year"] = countries["date"].dt.year
    countries["month"] = countries["date"].dt.month
    return total, countries


def prepare_hotel(raw: pd.DataFrame) -> pd.DataFrame:
    hotel = wide_monthly_to_long(raw, "value")
    hotel = hotel.pivot_table(index="date", columns="series", values="value", aggfunc="first")
    hotel = hotel.reset_index()
    hotel.columns.name = None
    hotel["year"] = hotel["date"].dt.year
    hotel["month"] = hotel["date"].dt.month
    return hotel


def parse_quarter_column(col: str) -> tuple[int, int] | None:
    if len(col) != 6 or not col[:4].isdigit() or col[4] not in "1234" or col[5] != "Q":
        return None
    return int(col[:4]), int(col[4])


def prepare_receipts(raw: pd.DataFrame) -> pd.DataFrame:
    q_cols = [c for c in raw.columns if parse_quarter_column(c) is not None]
    long = raw.melt(
        id_vars=["DataSeries"],
        value_vars=q_cols,
        var_name="period_raw",
        value_name="tourism_receipts_m",
    )
    long["year"] = long["period_raw"].map(lambda x: parse_quarter_column(x)[0])
    long["quarter"] = long["period_raw"].map(lambda x: parse_quarter_column(x)[1])
    long["component"] = long["DataSeries"].astype(str).str.strip()
    long["tourism_receipts_m"] = pd.to_numeric(long["tourism_receipts_m"], errors="coerce")
    return long[["year", "quarter", "component", "tourism_receipts_m"]].dropna()


def annual_arrivals(total: pd.DataFrame) -> pd.DataFrame:
    annual = (
        total.groupby("year", as_index=False)["arrivals"]
        .sum()
        .sort_values("year")
    )
    annual["arrivals_m"] = annual["arrivals"] / 1_000_000
    return annual


def format_num(value: float, suffix: str = "") -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}m{suffix}"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k{suffix}"
    return f"{value:.0f}{suffix}"


def pct(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}%"


def scale(value, domain_min, domain_max, range_min, range_max):
    if domain_max == domain_min:
        return (range_min + range_max) / 2
    return range_min + (value - domain_min) * (range_max - range_min) / (domain_max - domain_min)


def svg_wrap(width, height, body, title=None):
    title_el = f"<title>{title}</title>" if title else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">
  {title_el}
  <rect width="{width}" height="{height}" rx="12" fill="{COLORS['panel']}"/>
  {body}
</svg>
"""


def text(x, y, content, size=13, color=None, weight="400", anchor="start"):
    color = color or COLORS["ink"]
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{content}</text>'
    )


def axis_grid(x0, y0, w, h, ticks=4):
    parts = []
    for i in range(ticks + 1):
        y = y0 + h * i / ticks
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" stroke="{COLORS["line"]}" stroke-width="1"/>')
    return "\n".join(parts)


def line_path(points):
    return " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))


def make_recovery_chart(annual: pd.DataFrame, out: Path) -> None:
    df = annual[(annual["year"] >= 2015) & (annual["year"] <= 2025)].copy()
    x0, y0, w, h = 72, 70, 720, 260
    ymin, ymax = 0, max(df["arrivals_m"]) * 1.12
    pts = [
        (
            scale(row.year, df["year"].min(), df["year"].max(), x0, x0 + w),
            scale(row.arrivals_m, ymin, ymax, y0 + h, y0),
        )
        for row in df.itertuples()
    ]
    parts = [
        text(36, 32, "Singapore visitor arrivals: recovery curve", 20, weight="700"),
        text(36, 54, "Annual arrivals remain close to the 2019 benchmark, with 2025 still below the pre-COVID peak.", 12, COLORS["muted"]),
        axis_grid(x0, y0, w, h),
        f'<path d="{line_path(pts)}" fill="none" stroke="{COLORS["blue"]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>',
    ]
    for row, (x, y) in zip(df.itertuples(), pts):
        if row.year in [2015, 2019, 2020, 2025]:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS["blue"]}"/>')
            parts.append(text(x, y - 10, f"{int(row.year)}: {row.arrivals_m:.1f}m", 11, COLORS["ink"], "700", "middle"))
    for year in range(int(df["year"].min()), int(df["year"].max()) + 1, 2):
        x = scale(year, df["year"].min(), df["year"].max(), x0, x0 + w)
        parts.append(text(x, y0 + h + 26, str(year), 11, COLORS["muted"], anchor="middle"))
    for tick in range(0, int(math.ceil(ymax)) + 1, 5):
        y = scale(tick, ymin, ymax, y0 + h, y0)
        parts.append(text(x0 - 12, y + 4, f"{tick}m", 10, COLORS["muted"], anchor="end"))
    out.write_text(svg_wrap(840, 380, "\n".join(parts), "Visitor arrivals recovery chart"), encoding="utf-8")


def make_market_mix_chart(countries: pd.DataFrame, out: Path) -> pd.DataFrame:
    yearly = countries[countries["year"].eq(2025)].groupby("series", as_index=False)["arrivals"].sum()
    yearly = yearly.sort_values("arrivals", ascending=False).head(10)
    total = yearly["arrivals"].sum()
    x0, y0, w, row_h = 210, 70, 560, 25
    parts = [
        text(36, 32, "Top source countries: concentration and mix", 20, weight="700"),
        text(36, 54, "Top 10 source countries in 2025 show where growth and concentration risk need active monitoring.", 12, COLORS["muted"]),
    ]
    maxv = yearly["arrivals"].max()
    for tick in range(0, int(math.floor(maxv / 500_000)) * 500_000 + 1, 500_000):
        x = x0 + scale(tick, 0, maxv, 0, w)
        parts.append(f'<line x1="{x:.1f}" y1="{y0 - 8}" x2="{x:.1f}" y2="{y0 + row_h * len(yearly)}" stroke="{COLORS["line"]}" stroke-width="1"/>')
        parts.append(text(x, y0 + row_h * len(yearly) + 20, f"{tick/1_000_000:.1f}m", 10, COLORS["muted"], anchor="middle"))
    for i, row in enumerate(yearly.itertuples(index=False)):
        y = y0 + i * row_h
        bw = scale(row.arrivals, 0, maxv, 0, w)
        color = COLORS["teal"] if i < 5 else COLORS["blue"]
        parts.append(text(36, y + 16, row.series, 12, COLORS["ink"], "600"))
        parts.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="17" rx="4" fill="{color}" opacity="0.86"/>')
        parts.append(text(x0 + bw + 8, y + 14, f"{row.arrivals/1_000_000:.2f}m", 11, COLORS["muted"]))
    parts.append(text(x0 + w / 2, 352, "2025 international visitor arrivals", 11, COLORS["muted"], anchor="middle"))
    out.write_text(svg_wrap(840, 360, "\n".join(parts), "Top source countries chart"), encoding="utf-8")
    yearly["share_of_top10"] = yearly["arrivals"] / total
    return yearly


def make_hotel_alignment_chart(total: pd.DataFrame, hotel: pd.DataFrame, out: Path) -> pd.DataFrame:
    arrivals_monthly = total[["date", "arrivals"]].copy()
    merged = arrivals_monthly.merge(hotel[["date", "Room Revenue", "Average Hotel Occupancy Rate"]], on="date", how="inner")
    merged = merged[(merged["date"] >= "2019-01-01") & (merged["date"] <= "2026-03-01")].copy()
    merged["arrivals_index"] = merged["arrivals"] / merged.loc[merged["date"].dt.year.eq(2019), "arrivals"].mean() * 100
    merged["room_revenue_index"] = merged["Room Revenue"] / merged.loc[merged["date"].dt.year.eq(2019), "Room Revenue"].mean() * 100
    x0, y0, w, h = 70, 76, 720, 245
    y_max = max(merged["arrivals_index"].max(), merged["room_revenue_index"].max()) * 1.08
    x_min = merged["date"].min().value
    x_max = merged["date"].max().value
    arr_pts = [(scale(d.value, x_min, x_max, x0, x0 + w), scale(v, 0, y_max, y0 + h, y0)) for d, v in zip(merged["date"], merged["arrivals_index"])]
    rev_pts = [(scale(d.value, x_min, x_max, x0, x0 + w), scale(v, 0, y_max, y0 + h, y0)) for d, v in zip(merged["date"], merged["room_revenue_index"])]
    parts = [
        text(36, 32, "Tourism demand vs hotel revenue", 20, weight="700"),
        text(36, 54, "Indexed to 2019 monthly average: hotel revenue has recovered faster than visitor volume.", 12, COLORS["muted"]),
        axis_grid(x0, y0, w, h),
        f'<path d="{line_path(arr_pts)}" fill="none" stroke="{COLORS["blue"]}" stroke-width="3" stroke-linejoin="round"/>',
        f'<path d="{line_path(rev_pts)}" fill="none" stroke="{COLORS["amber"]}" stroke-width="3" stroke-linejoin="round"/>',
        f'<rect x="562" y="28" width="12" height="12" fill="{COLORS["blue"]}"/>{text(580, 39, "Visitor arrivals index", 11, COLORS["muted"])}',
        f'<rect x="694" y="28" width="12" height="12" fill="{COLORS["amber"]}"/>{text(712, 39, "Room revenue index", 11, COLORS["muted"])}',
    ]
    for tick in [0, 50, 100, 150, 200]:
        if tick <= y_max:
            y = scale(tick, 0, y_max, y0 + h, y0)
            parts.append(text(x0 - 10, y + 4, f"{tick}", 10, COLORS["muted"], anchor="end"))
    parts.append(text(24, y0 - 8, "Index, 2019 monthly avg = 100", 10, COLORS["muted"]))
    for year in [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]:
        d = pd.Timestamp(year=year, month=1, day=1)
        x = scale(d.value, x_min, x_max, x0, x0 + w)
        parts.append(text(x, y0 + h + 26, str(year), 10, COLORS["muted"], anchor="middle"))
    out.write_text(svg_wrap(840, 370, "\n".join(parts), "Demand and hotel revenue index"), encoding="utf-8")
    return merged


def synthetic_destination_data(total: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    random.seed(42)
    base = total[(total["date"] >= "2023-01-01") & (total["date"] <= "2026-03-01")][["date", "arrivals"]].copy()
    base = base.sort_values("date")
    top_markets = (
        countries[countries["year"].eq(2025)].groupby("series")["arrivals"].sum().sort_values(ascending=False).head(6).index.tolist()
    )
    rows = []
    for row in base.itertuples(index=False):
        month = row.date.month
        season = 1.10 if month in [7, 8, 12] else 1.04 if month in [3, 6, 11] else 0.96
        destination_visitors = int(row.arrivals * random.uniform(0.125, 0.155) * season)
        tourist_share = 0.48 + min(0.13, (row.arrivals - base["arrivals"].min()) / (base["arrivals"].max() - base["arrivals"].min()) * 0.16)
        tourist_visitors = int(destination_visitors * tourist_share)
        local_visitors = destination_visitors - tourist_visitors
        tourist_daily_spend = 100.0
        local_daily_spend = 50.0
        attraction = tourist_visitors * 30.0 + local_visitors * 10.0
        fnb = tourist_visitors * 35.0 + local_visitors * 25.0
        retail = tourist_visitors * 25.0 + local_visitors * 10.0
        events = tourist_visitors * 10.0 + local_visitors * 5.0
        revenue = attraction + fnb + retail + events
        campaign_spend = random.uniform(100_000, 200_000) * (1.25 if month in [6, 11, 12] else 1.0)
        campaign_roas = random.uniform(2.0, 3.0)
        campaign_revenue = campaign_spend * campaign_roas
        crowd_penalty = max(0, destination_visitors - base["arrivals"].median() * 0.145) / 40_000
        nps = max(35, min(65, random.gauss(50, 3) - crowd_penalty * 3))
        negative_feedback = max(10, min(30, random.gauss(20, 3) + crowd_penalty * 2))
        theme_weights = [
            random.uniform(0.26, 0.32),  # transport
            random.uniform(0.22, 0.28),  # crowding
            random.uniform(0.18, 0.24),  # service / queueing
            random.uniform(0.16, 0.22),  # wayfinding / facilities
        ]
        theme_total = sum(theme_weights)
        transport_complaints = negative_feedback * theme_weights[0] / theme_total
        crowding_complaints = negative_feedback * theme_weights[1] / theme_total
        service_queueing_complaints = negative_feedback * theme_weights[2] / theme_total
        facilities_wayfinding_complaints = negative_feedback * theme_weights[3] / theme_total
        top_market = random.choice(top_markets[:4])
        rows.append(
            {
                "date": row.date,
                "destination_visitors": destination_visitors,
                "tourist_visitors": tourist_visitors,
                "local_visitors": local_visitors,
                "tourist_share": tourist_visitors / destination_visitors,
                "tourist_daily_spend": tourist_daily_spend,
                "local_daily_spend": local_daily_spend,
                "attraction_revenue": attraction,
                "fnb_revenue": fnb,
                "retail_revenue": retail,
                "events_revenue": events,
                "total_revenue": revenue,
                "revenue_per_visitor": revenue / destination_visitors,
                "campaign_spend": campaign_spend,
                "campaign_revenue": campaign_revenue,
                "campaign_roas": campaign_roas,
                "nps": nps,
                "negative_feedback_pct": negative_feedback,
                "transport_complaints_pct": transport_complaints,
                "crowding_complaints_pct": crowding_complaints,
                "service_queueing_complaints_pct": service_queueing_complaints,
                "facilities_wayfinding_complaints_pct": facilities_wayfinding_complaints,
                "priority_market": top_market,
            }
        )
    return pd.DataFrame(rows)


def make_destination_dashboard(dest: pd.DataFrame, out: Path) -> None:
    latest = dest.sort_values("date").tail(12)
    x0, y0, w, h = 60, 162, 720, 160
    y_max = latest["destination_visitors"].max() * 1.12
    pts = [
        (
            scale(i, 0, len(latest) - 1, x0, x0 + w),
            scale(v, 0, y_max, y0 + h, y0),
        )
        for i, v in enumerate(latest["destination_visitors"])
    ]
    aggregate_roas = latest["campaign_revenue"].sum() / latest["campaign_spend"].sum()
    kpis = [
        ("Visitors", format_num(latest["destination_visitors"].sum())),
        ("Revenue", f"${latest['total_revenue'].sum()/1_000_000:.1f}m"),
        ("Tourist mix", pct(latest["tourist_share"].mean() * 100, 0)),
        ("Campaign ROAS", f"{aggregate_roas:.1f}x"),
        ("NPS", f"{latest['nps'].mean():.0f}"),
    ]
    parts = [
        text(36, 34, "Synthetic destination layer: management dashboard", 20, weight="700"),
        text(36, 56, "Simulates how a destination operator could link demand, revenue, campaigns and guest experience.", 12, COLORS["muted"]),
    ]
    for i, (label, value) in enumerate(kpis):
        x = 36 + i * 158
        parts.append(f'<rect x="{x}" y="78" width="142" height="58" rx="8" fill="{COLORS["bg"]}"/>')
        parts.append(text(x + 14, 101, label, 11, COLORS["muted"], "600"))
        parts.append(text(x + 14, 126, value, 20, COLORS["ink"], "700"))
    parts.append(axis_grid(x0, y0, w, h, ticks=3))
    for tick in range(0, int(math.ceil(y_max / 50_000)) * 50_000 + 1, 50_000):
        y = scale(tick, 0, y_max, y0 + h, y0)
        parts.append(text(x0 - 10, y + 4, f"{tick/1000:.0f}k", 10, COLORS["muted"], anchor="end"))
    parts.append(f'<path d="{line_path(pts)}" fill="none" stroke="{COLORS["teal"]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    for i, row in enumerate(latest.itertuples(index=False)):
        if i in [0, len(latest) - 1]:
            x, y = pts[i]
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS["teal"]}"/>')
    parts.append(text(60.0, 326, "Apr", 10, COLORS["muted"], anchor="middle"))
    parts.append(text(256.4, 326, "Jul", 10, COLORS["muted"], anchor="middle"))
    parts.append(text(452.7, 326, "Oct", 10, COLORS["muted"], anchor="middle"))
    parts.append(text(649.1, 326, "Jan", 10, COLORS["muted"], anchor="middle"))
    parts.append(text(x0 + w / 2, 368, "12 months synthetic destination visitors", 11, COLORS["muted"], anchor="middle"))
    parts.append(f'<text x="18" y="242" transform="rotate(-90 18 242)" font-family="Arial, sans-serif" font-size="10" font-weight="400" fill="{COLORS["muted"]}" text-anchor="middle">Monthly visitors</text>')
    out.write_text(svg_wrap(840, 385, "\n".join(parts), "Synthetic destination dashboard"), encoding="utf-8")


def make_opportunity_matrix(countries: pd.DataFrame, out: Path) -> pd.DataFrame:
    yearly = countries.groupby(["year", "series"], as_index=False)["arrivals"].sum()
    piv = yearly.pivot(index="series", columns="year", values="arrivals")
    for col in [2019, 2024, 2025]:
        if col not in piv.columns:
            raise ValueError(f"Missing year {col} in arrivals data")
    opp = piv[[2019, 2024, 2025]].dropna().copy()
    opp = opp[(opp[2019] > 80_000) & (opp[2025] > 50_000)]
    opp["recovery_vs_2019"] = opp[2025] / opp[2019] * 100
    opp["yoy_growth_2025"] = (opp[2025] / opp[2024] - 1) * 100
    opp["arrivals_2025"] = opp[2025]
    opp = opp.sort_values("arrivals_2025", ascending=False).head(18).reset_index()
    x0, y0, w, h = 90, 74, 640, 245
    xmin, xmax = min(40, opp["recovery_vs_2019"].min() - 5), max(130, opp["recovery_vs_2019"].max() + 5)
    ymin, ymax = min(-20, opp["yoy_growth_2025"].min() - 5), max(35, opp["yoy_growth_2025"].max() + 5)
    parts = [
        text(36, 32, "Growth opportunity matrix", 20, weight="700"),
        text(36, 54, "Markets above the horizontal line are growing; markets left of 100% are still below 2019 baseline.", 12, COLORS["muted"]),
        axis_grid(x0, y0, w, h),
    ]
    x100 = scale(100, xmin, xmax, x0, x0 + w)
    y0line = scale(0, ymin, ymax, y0 + h, y0)
    parts.append(f'<line x1="{x100:.1f}" y1="{y0}" x2="{x100:.1f}" y2="{y0+h}" stroke="{COLORS["muted"]}" stroke-dasharray="4 4"/>')
    parts.append(f'<line x1="{x0}" y1="{y0line:.1f}" x2="{x0+w}" y2="{y0line:.1f}" stroke="{COLORS["muted"]}" stroke-dasharray="4 4"/>')
    for tick in [50, 75, 100, 125, 150]:
        if xmin <= tick <= xmax:
            x = scale(tick, xmin, xmax, x0, x0 + w)
            parts.append(text(x, y0 + h + 20, f"{tick}%", 10, COLORS["muted"], anchor="middle"))
    for tick in [-20, 0, 20, 40]:
        if ymin <= tick <= ymax:
            y = scale(tick, ymin, ymax, y0 + h, y0)
            parts.append(text(x0 - 10, y + 4, f"{tick}%", 10, COLORS["muted"], anchor="end"))
    max_size = opp["arrivals_2025"].max()
    for row in opp.itertuples(index=False):
        x = scale(row.recovery_vs_2019, xmin, xmax, x0, x0 + w)
        y = scale(row.yoy_growth_2025, ymin, ymax, y0 + h, y0)
        r = scale(math.sqrt(row.arrivals_2025), 0, math.sqrt(max_size), 4, 18)
        color = COLORS["green"] if row.yoy_growth_2025 > 0 and row.recovery_vs_2019 < 100 else COLORS["blue"]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" fill-opacity="0.72" stroke="#fff" stroke-width="1.5"/>')
        if row.arrivals_2025 > 350_000 or (row.yoy_growth_2025 > 10 and row.recovery_vs_2019 < 100):
            parts.append(text(x + r + 3, y + 4, row.series, 10, COLORS["ink"], "600"))
    parts.append(text(x0, y0 + h + 32, "Recovery vs 2019 baseline", 11, COLORS["muted"]))
    parts.append(text(18, y0 - 8, "2025 YoY growth", 11, COLORS["muted"]))
    out.write_text(svg_wrap(840, 380, "\n".join(parts), "Growth opportunity matrix"), encoding="utf-8")
    return opp


def make_experience_chart(dest: pd.DataFrame, out: Path) -> None:
    latest = dest.sort_values("date").tail(12).copy()
    x0, y0, w, h = 62, 74, 720, 230
    x_min = latest["date"].min().value
    x_max = latest["date"].max().value
    nps_pts = [(scale(d.value, x_min, x_max, x0, x0 + w), scale(v, 20, 70, y0 + h, y0)) for d, v in zip(latest["date"], latest["nps"])]
    neg_pts = [(scale(d.value, x_min, x_max, x0, x0 + w), scale(v, 0, 40, y0 + h, y0)) for d, v in zip(latest["date"], latest["negative_feedback_pct"])]
    parts = [
        text(36, 32, "Guest experience early-warning view", 20, weight="700"),
        text(36, 54, "Dual-axis view for the latest 12 months: NPS on the left axis, negative feedback on the right axis.", 12, COLORS["muted"]),
        axis_grid(x0, y0, w, h),
        f'<path d="{line_path(nps_pts)}" fill="none" stroke="{COLORS["green"]}" stroke-width="3"/>',
        f'<path d="{line_path(neg_pts)}" fill="none" stroke="{COLORS["red"]}" stroke-width="3"/>',
        f'<rect x="560" y="28" width="12" height="12" fill="{COLORS["green"]}"/>{text(578, 39, "NPS (LHS)", 11, COLORS["muted"])}',
        f'<rect x="646" y="28" width="12" height="12" fill="{COLORS["red"]}"/>{text(664, 39, "Negative feedback % (RHS)", 11, COLORS["muted"])}',
    ]
    for tick in [20, 30, 40, 50, 60, 70]:
        y = scale(tick, 20, 70, y0 + h, y0)
        parts.append(text(x0 - 10, y + 4, f"{tick}", 10, COLORS["muted"], anchor="end"))
    for tick in [0, 10, 20, 30, 40]:
        y = scale(tick, 0, 40, y0 + h, y0)
        parts.append(text(x0 + w + 12, y + 4, f"{tick}%", 10, COLORS["muted"], anchor="start"))
    parts.append(text(24, y0 - 8, "NPS", 10, COLORS["muted"]))
    parts.append(text(x0 + w + 12, y0 - 8, "Negative feedback %", 10, COLORS["muted"]))
    for row in latest.itertuples(index=False):
        if row.date.month in [1, 4, 7, 10]:
            x = scale(row.date.value, x_min, x_max, x0, x0 + w)
            parts.append(text(x, y0 + h + 22, row.date.strftime("%b"), 10, COLORS["muted"], anchor="middle"))
    complaints = [
        ("Transport", latest["transport_complaints_pct"].mean(), COLORS["amber"]),
        ("Crowding", latest["crowding_complaints_pct"].mean(), COLORS["purple"]),
        ("Service / queueing", latest["service_queueing_complaints_pct"].mean(), COLORS["blue"]),
        ("Wayfinding / facilities", latest["facilities_wayfinding_complaints_pct"].mean(), COLORS["muted"]),
    ]
    parts.append(f'<rect x="36" y="328" width="768" height="48" rx="8" fill="{COLORS["bg"]}" stroke="{COLORS["line"]}"/>')
    parts.append(text(52, 346, "Average negative feedback composition", 11, COLORS["muted"], "600"))
    parts.append(text(52, 362, "The average complaint rate is split into modeled issue themes for the displayed 12-month period.", 10, COLORS["muted"]))
    sx = 52
    for label, value, color in complaints:
        bw = value * 12
        parts.append(f'<rect x="{sx}" y="382" width="{bw:.1f}" height="16" rx="4" fill="{color}" opacity="0.84"/>')
        parts.append(text(sx, 412, f"{label} {value:.1f} pts", 10, COLORS["muted"], anchor="start"))
        sx += bw + 8
    parts.append(text(52, 428, f"Displayed period averages: NPS {latest['nps'].mean():.1f}, negative feedback {latest['negative_feedback_pct'].mean():.1f}%.", 10, COLORS["muted"]))
    out.write_text(svg_wrap(840, 450, "\n".join(parts), "Guest experience early warning chart"), encoding="utf-8")


def write_html(summary: dict, insights: list[str]) -> None:
    cards = "\n".join(
        f'<article class="metric"><span>{k}</span><strong>{v}</strong></article>'
        for k, v in summary["kpis"].items()
    )
    insight_items = "\n".join(f"<li>{item}</li>" for item in insights)
    actions = [
        (
            "Prioritize country-specific activation",
            "China, Indonesia, Malaysia, Australia and India account for the largest share of arrivals.",
            "Build source-country views for campaign planning, language localization, travel-trade partnerships and visitor-mix monitoring.",
            "Visitors by source country, campaign ROAS, revenue per visitor"
        ),
        (
            "Separate volume recovery from value recovery",
            "Hotel room revenue has recovered faster than visitor arrivals.",
            "Track whether destination visitors are converting into higher-yield attraction, F&B, retail and event spend rather than only measuring footfall.",
            "Revenue per visitor, average spend, business-unit revenue mix"
        ),
        (
            "Use recovery gaps as a growth pipeline",
            "Japan, Germany and Taiwan show positive growth but remain below the 2019 baseline.",
            "Test tactical campaigns, airline/travel-agent partnerships and event bundles for under-recovered but improving countries.",
            "Recovery vs 2019, YoY growth, campaign bookings"
        ),
        (
            "Protect experience during peak demand",
            "Synthetic guest-experience modeling links high visitorship to crowding and transport complaint pressure.",
            "Pair demand forecasts with transport nudges, crowd-level dashboards, staffing plans and peak-period communication.",
            "NPS, negative feedback %, crowding/transport complaints"
        ),
    ]
    action_rows = "\n".join(
        f"<tr><td>{a}</td><td>{e}</td><td>{r}</td><td>{m}</td></tr>"
        for a, e, r, m in actions
    )
    sources = "\n".join(
        f'<li><a href="{meta["url"]}">{meta["label"]}</a> <code>{meta["id"]}</code></li>'
        for meta in DATASETS.values()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Destination Performance & Tourism Growth Dashboard</title>
  <style>
    :root {{
      --ink:#17202a; --muted:#68737d; --line:#d8dee5; --bg:#f7f9fb;
      --panel:#ffffff; --blue:#1f77b4; --teal:#129c9a; --green:#2b8a3e;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Arial, sans-serif; color:var(--ink); background:var(--bg); }}
    main {{ width:min(1120px, calc(100% - 32px)); margin:0 auto; padding:38px 0 54px; }}
    .eyebrow {{ color:var(--teal); font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
    h1 {{ font-size:38px; line-height:1.08; margin:10px 0 12px; max-width:850px; }}
    .lead {{ max-width:850px; color:var(--muted); font-size:17px; line-height:1.5; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin:28px 0; }}
    .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .metric span {{ color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; }}
    .metric strong {{ display:block; margin-top:8px; font-size:24px; }}
    .section {{ margin-top:24px; padding-top:10px; border-top:1px solid var(--line); }}
    h2 {{ font-size:23px; margin:20px 0 8px; }}
    p, li {{ line-height:1.55; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }}
    figure {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }}
    figure img {{ display:block; width:100%; height:auto; }}
    .story {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px 22px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th, td {{ text-align:left; vertical-align:top; padding:12px; border-bottom:1px solid var(--line); font-size:14px; line-height:1.45; }}
    th {{ background:#edf2f7; font-size:12px; text-transform:uppercase; color:var(--muted); }}
    tr:last-child td {{ border-bottom:0; }}
    .sources a {{ color:var(--blue); }}
    code {{ font-size:12px; background:#edf2f7; padding:2px 4px; border-radius:4px; }}
    @media (max-width: 860px) {{
      h1 {{ font-size:30px; }}
      .metrics, .grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Portfolio Case Study</div>
    <h1>Destination Performance & Tourism Growth Dashboard</h1>
    <p class="lead">This showcase combines official Singapore tourism and hotel datasets with a transparent synthetic destination operating layer to show how a destination operator can connect visitorship trends, destination value, campaign ROAS and guest experience in one management view.</p>
    <section class="metrics">{cards}</section>

    <section class="section story">
      <h2>Executive Narrative for SDC</h2>
      <ul>{insight_items}</ul>
    </section>

    <section class="section">
      <h2>Step 1: Public Tourism Demand</h2>
      <p>Monthly visitor arrivals are used as the core demand signal. For SDC, this identifies which visitor source countries may matter most for partnership strategy, campaign localization and destination programming.</p>
      <div class="grid">
        <figure><img src="assets/arrivals_recovery.svg" alt="Singapore visitor arrivals recovery chart"></figure>
        <figure><img src="assets/source_market_mix.svg" alt="Top Singapore visitor source countries chart"></figure>
      </div>
    </section>

    <section class="section">
      <h2>Step 2: Tourism Benchmarking</h2>
      <p>Hotel room revenue, average room rate and occupancy provide a business-quality proxy: not just whether visitor volumes are back, but whether demand is translating into yield. This is useful for SDC because destination management should optimize for visitor value, not footfall alone.</p>
      <div class="grid">
        <figure><img src="assets/hotel_revenue_alignment.svg" alt="Demand versus hotel room revenue index chart"></figure>
        <figure><img src="assets/growth_opportunity_matrix.svg" alt="Source country growth opportunity matrix"></figure>
      </div>
    </section>

    <section class="section">
      <h2>Step 3: Synthetic Destination Layer</h2>
      <p>The destination layer is synthetic by design, because attraction-level revenue, guest satisfaction and campaign ROAS are not public. It creates a management view that links public tourism demand to simulated destination operations.</p>
      <div class="grid">
        <figure><img src="assets/destination_dashboard.svg" alt="Synthetic destination dashboard"></figure>
        <figure><img src="assets/guest_experience.svg" alt="Synthetic guest experience early-warning chart"></figure>
      </div>
    </section>

    <section class="section">
      <h2>SDC Action Table</h2>
      <table>
        <thead><tr><th>Management action</th><th>Evidence from analysis</th><th>How SDC could use it</th><th>Metric to monitor</th></tr></thead>
        <tbody>{action_rows}</tbody>
      </table>
    </section>

    <section class="section story sources">
      <h2>Data Sources</h2>
      <ul>{sources}</ul>
      <p><strong>Note:</strong> Singapore tourism demand and hotel benchmarks are official public datasets. Destination revenue, campaign ROAS, guest feedback and local-versus-tourist destination behavior are synthetic and used only to demonstrate analytical approach.</p>
    </section>
  </main>
</body>
</html>
"""
    (ROOT / "index.html").write_text(html, encoding="utf-8")


def write_readme(summary: dict, insights: list[str]) -> None:
    insight_md = "\n".join(f"- {item}" for item in insights)
    sources_md = "\n".join(f"- {meta['label']}: {meta['url']} (`{meta['id']}`)" for meta in DATASETS.values())
    readme = f"""# Destination Performance & Tourism Growth Dashboard

This portfolio piece combines official Singapore tourism data with synthetic destination-level operating data to simulate how a destination operator can monitor visitorship, revenue, guest experience and campaign performance.

## Direct Data Links

{sources_md}

## Analysis Flow

1. Pull official data.gov.sg datasets through the datastore API.
2. Normalize wide monthly tables into tidy monthly time series.
3. Analyze visitor recovery, source-country mix, seasonality and concentration risk.
4. Benchmark demand against hotel room revenue, occupancy and annual tourism receipts.
5. Generate a transparent synthetic destination layer for attraction revenue, F&B revenue, retail revenue, campaign ROAS, guest feedback and visitor segments.
6. Produce web-ready SVG visuals and a standalone `index.html` showcase.

## Key Findings

{insight_md}

## Transparency Note

Public datasets are used for Singapore-level demand and hotel benchmarking. Destination-level operating metrics are synthetic because internal destination data is not public.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw = save_raw()
    total, countries = prepare_arrivals(raw["visitor_arrivals_monthly"])
    hotel = prepare_hotel(raw["hotel_statistics_monthly"])
    receipts = prepare_receipts(raw["tourism_receipts_ytd_quarterly"])

    annual = annual_arrivals(total)
    top_markets = make_market_mix_chart(countries, ASSET_DIR / "source_market_mix.svg")
    hotel_alignment = make_hotel_alignment_chart(total, hotel, ASSET_DIR / "hotel_revenue_alignment.svg")
    opportunities = make_opportunity_matrix(countries, ASSET_DIR / "growth_opportunity_matrix.svg")
    make_recovery_chart(annual, ASSET_DIR / "arrivals_recovery.svg")

    dest = synthetic_destination_data(total, countries)
    make_destination_dashboard(dest, ASSET_DIR / "destination_dashboard.svg")
    make_experience_chart(dest, ASSET_DIR / "guest_experience.svg")

    total.to_csv(PROCESSED_DIR / "singapore_monthly_arrivals_total.csv", index=False)
    countries.to_csv(PROCESSED_DIR / "singapore_monthly_arrivals_by_market.csv", index=False)
    hotel.to_csv(PROCESSED_DIR / "singapore_monthly_hotel_statistics.csv", index=False)
    receipts.to_csv(PROCESSED_DIR / "singapore_annual_tourism_receipts.csv", index=False)
    dest.to_csv(PROCESSED_DIR / "synthetic_destination_monthly_kpis.csv", index=False)
    top_markets.to_csv(PROCESSED_DIR / "top_source_markets_2025.csv", index=False)
    opportunities.to_csv(PROCESSED_DIR / "source_market_opportunity_matrix.csv", index=False)
    hotel_alignment.to_csv(PROCESSED_DIR / "hotel_demand_alignment_index.csv", index=False)

    full_years = annual[annual["year"].between(2019, 2025)]
    arrivals_2019 = float(full_years.loc[full_years["year"].eq(2019), "arrivals"].iloc[0])
    arrivals_2025 = float(full_years.loc[full_years["year"].eq(2025), "arrivals"].iloc[0])
    recovery = arrivals_2025 / arrivals_2019 * 100
    ytd_2026 = total[total["date"].between("2026-01-01", "2026-03-01")]["arrivals"].sum()
    ytd_2019 = total[total["date"].between("2019-01-01", "2019-03-01")]["arrivals"].sum()
    ytd_recovery = ytd_2026 / ytd_2019 * 100
    top5_share = top_markets.head(5)["arrivals"].sum() / countries[countries["year"].eq(2025)].groupby("series")["arrivals"].sum().sum() * 100
    hotel_2025 = hotel[hotel["year"].eq(2025)]["Room Revenue"].sum()
    hotel_2019 = hotel[hotel["year"].eq(2019)]["Room Revenue"].sum()
    hotel_recovery = hotel_2025 / hotel_2019 * 100
    annual_receipts = receipts[
        receipts["component"].eq("Tourism Receipts") & receipts["quarter"].eq(4)
    ].sort_values("year")
    latest_annual_receipts = annual_receipts.tail(1).iloc[0]
    best_growth = opportunities[(opportunities["yoy_growth_2025"] > 0) & (opportunities["recovery_vs_2019"] < 100)].sort_values("yoy_growth_2025", ascending=False).head(3)
    growth_names = ", ".join(best_growth["series"].tolist()) if not best_growth.empty else "selected under-recovered markets"

    insights = [
        f"Singapore recorded {arrivals_2025/1_000_000:.1f}m international visitor arrivals in 2025, reaching {recovery:.0f}% of the 2019 baseline.",
        f"Jan-Mar 2026 arrivals reached {ytd_recovery:.0f}% of the same 2019 period, suggesting demand has broadly normalized but remains uneven by market.",
        f"The top five 2025 source countries account for {top5_share:.0f}% of arrivals, making source-country concentration an important management lens.",
        f"Hotel room revenue in 2025 reached {hotel_recovery:.0f}% of 2019, indicating yield recovery has outpaced pure visitor-volume recovery.",
        f"Opportunity screening highlights {growth_names} as markets worth deeper campaign and partnership review.",
        f"The synthetic destination layer shows how management could connect public demand signals to revenue per visitor, campaign ROAS and guest experience monitoring.",
    ]
    summary = {
        "kpis": {
            "2025 arrivals": f"{arrivals_2025/1_000_000:.1f}m",
            "Recovery vs 2019": f"{recovery:.0f}%",
            "Top 5 Countries Market Share": f"{top5_share:.0f}%",
            "Hotel revenue vs 2019": f"{hotel_recovery:.0f}%",
        },
        "latest_annual_receipts": {
            "year": int(latest_annual_receipts["year"]),
            "tourism_receipts_m": float(latest_annual_receipts["tourism_receipts_m"]),
        },
        "insights": insights,
        "sources": DATASETS,
    }
    (PROCESSED_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(summary, insights)
    write_readme(summary, insights)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
