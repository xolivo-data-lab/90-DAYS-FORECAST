import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = r"C:\Users\OlivoS01\OneDrive - TCL Technology Group Corporation\backup 2023\Borrar\Forecast folder"
OUTPUT_DIR = rf"{BASE_DIR}\output"

CONFIG = {
    "claims_file": rf"{BASE_DIR}\claims_file.xlsx",
    "rc_stock_file": rf"{BASE_DIR}\rc_stock_file.xlsx",
    "customer_forecast_file": rf"{BASE_DIR}\customer_forecast_file.xlsx",
    "output_dir": OUTPUT_DIR,

    "claims_date_col": "Finish Time",
    "claims_model_col": "Product model",
    "claims_part_col": "Material Code",

    "stock_part_col": "Part_Number",
    "stock_qty_col": "Quantity_On_Hand",

    "cust_model_col": "Model",
    "cust_month_col": "Forecast_Month",
    "cust_units_col": "Forecast_Units",
    "cust_return_rate_col": "Return_Rate_%",

    "lookback_months": 8,
    "safety_stock_factor": 1.5,
    "customer_weight": 0.45,
    "trend_factor": 0.60,
    "month_weights": [1.0, 0.7, 0.4],
}

# =========================================================
# HELPERS
# =========================================================
def normalize_text(s):
    if pd.isna(s):
        return ""
    return str(s).replace("\xa0", " ").strip()

def classify_priority(stock, fcst):
    if fcst > 0 and stock == 0:
        return "CRITICAL"
    elif stock < fcst * 0.5:
        return "HIGH"
    elif stock < fcst:
        return "MEDIUM"
    return "LOW"

def detect_separator(series):
    sample = " ".join(series.dropna().astype(str).head(50).tolist())
    for sep in [",", ";", "|"]:
        if sep in sample:
            return sep
    return ","

def calc_slope(values):
    y = np.array(values, dtype=float)
    if len(y) < 2:
        return 0.0
    if np.all(y == y[0]):
        return 0.0
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope)

def ensure_dir(path_str):
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    return p

def validate_input_files(cfg):
    print("\nValidating input files...")
    for key in ["claims_file", "rc_stock_file", "customer_forecast_file"]:
        file_path = Path(cfg[key])
        print(f"{key}: {file_path}")
        print("Exists:", file_path.exists())
        print("-" * 60)

# =========================================================
# LOAD CLAIMS
# =========================================================
def load_claims(cfg):
    print("\nLoading claims...")

    claims_path = Path(cfg["claims_file"])
    if not claims_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de claims:\n{claims_path}")

    df = pd.read_excel(claims_path)
    df.columns = [normalize_text(c) for c in df.columns]

    df[cfg["claims_date_col"]] = pd.to_datetime(df[cfg["claims_date_col"]], errors="coerce")
    df[cfg["claims_model_col"]] = df[cfg["claims_model_col"]].astype(str).map(normalize_text)
    df[cfg["claims_part_col"]] = df[cfg["claims_part_col"]].astype(str)

    df = df[df[cfg["claims_date_col"]].notna()].copy()

    sep = detect_separator(df[cfg["claims_part_col"]])
    df["parts_list"] = df[cfg["claims_part_col"]].str.split(sep)
    df = df.explode("parts_list")

    df["part_number"] = df["parts_list"].astype(str).map(normalize_text)
    df = df[~df["part_number"].isin(["", "nan", "None", "NONE"])].copy()

    df["model"] = df[cfg["claims_model_col"]]
    df["year_month"] = df[cfg["claims_date_col"]].dt.to_period("M")

    print("Claims rows after cleanup:", len(df))
    print("Unique models:", df["model"].nunique())
    print("Unique parts:", df["part_number"].nunique())

    return df[[cfg["claims_date_col"], "year_month", "model", "part_number"]].copy()

# =========================================================
# FILTER LAST MONTHS
# =========================================================
def filter_last_months(df, n):
    max_month = df["year_month"].max()
    min_month = max_month - (n - 1)
    out = df[df["year_month"] >= min_month].copy()

    print(f"\nUsing last {n} months: {min_month} to {max_month}")
    print("Rows in lookback window:", len(out))

    return out, min_month, max_month

# =========================================================
# MONTHLY USAGE
# =========================================================
def build_monthly_usage(claims_recent, min_month, max_month):
    print("\nBuilding monthly usage...")

    base = (
        claims_recent.groupby(["model", "part_number", "year_month"])
        .size()
        .reset_index(name="qty_used")
    )

    months = pd.period_range(min_month, max_month, freq="M")
    combos = base[["model", "part_number"]].drop_duplicates().copy()
    combos["key"] = 1

    mdf = pd.DataFrame({"year_month": months, "key": 1})

    full = combos.merge(mdf, on="key").drop(columns="key")
    full = full.merge(base, on=["model", "part_number", "year_month"], how="left")
    full["qty_used"] = full["qty_used"].fillna(0)

    print("Monthly usage rows:", len(full))
    return full

# =========================================================
# NEW PARTS
# =========================================================
def detect_new_parts(all_claims, recent_claims):
    print("\nDetecting new parts...")

    first_use = (
        all_claims.groupby("part_number")["year_month"]
        .min()
        .reset_index(name="first_used_month")
    )

    first_model = (
        all_claims.sort_values("year_month")
        .groupby("part_number")["model"]
        .first()
        .reset_index(name="first_model")
    )

    new_parts = first_use.merge(first_model, on="part_number", how="left")

    recent_min = recent_claims["year_month"].min()
    recent_max = recent_claims["year_month"].max()

    new_parts_recent = new_parts[
        (new_parts["first_used_month"] >= recent_min) &
        (new_parts["first_used_month"] <= recent_max)
    ].copy()

    print("New parts detected:", len(new_parts_recent))
    return new_parts_recent

# =========================================================
# RARE PARTS
# =========================================================
def detect_rare_parts(usage_full):
    print("\nDetecting rare parts...")
    rows = []

    for (model, part), grp in usage_full.groupby(["model", "part_number"]):
        grp = grp.sort_values("year_month")
        vals = grp["qty_used"].tolist()

        total_usage = sum(vals)
        months_active = sum(v > 0 for v in vals)
        avg_usage = np.mean(vals)
        std_usage = np.std(vals)
        last_month = vals[-1]
        slope = calc_slope(vals)

        flags = []

        if total_usage == 1:
            flags.append("USED_ONCE_ONLY")
        if months_active == 1 and total_usage > 1:
            flags.append("ONE_ACTIVE_MONTH_ONLY")
        if avg_usage > 0 and last_month >= avg_usage * 3 and last_month >= 3:
            flags.append("LAST_MONTH_SPIKE")
        if slope >= 1.0:
            flags.append("STRONG_INCREASING_TREND")
        if avg_usage > 0 and std_usage > avg_usage * 1.5:
            flags.append("HIGH_VOLATILITY")

        if flags:
            rows.append({
                "model": model,
                "part_number": part,
                "total_usage_8m": total_usage,
                "months_active": months_active,
                "avg_usage": round(avg_usage, 2),
                "std_usage": round(std_usage, 2),
                "last_month_usage": last_month,
                "trend_slope": round(slope, 2),
                "rare_flags": ", ".join(flags)
            })

    return pd.DataFrame(rows)

# =========================================================
# OBSOLETE / DECLINING
# =========================================================
def detect_obsolete_or_declining_parts(usage_full):
    print("\nDetecting obsolete or declining parts...")
    rows = []

    for (model, part), grp in usage_full.groupby(["model", "part_number"]):
        grp = grp.sort_values("year_month")
        vals = grp["qty_used"].tolist()

        total_usage = sum(vals)
        slope = calc_slope(vals)
        last_3_sum = sum(vals[-3:])
        last_month = vals[-1]
        prev_3_avg = np.mean(vals[:-3]) if len(vals) > 3 else np.mean(vals)

        flags = []

        if total_usage > 0 and last_3_sum == 0:
            flags.append("OBSOLETE_NO_USAGE_LAST_3M")
        if slope <= -0.5:
            flags.append("DECLINING_TREND")
        if prev_3_avg > 0 and last_month == 0 and slope < 0:
            flags.append("RECENT_DROP_TO_ZERO")

        if flags:
            rows.append({
                "model": model,
                "part_number": part,
                "total_usage_8m": total_usage,
                "last_month_usage": last_month,
                "last_3m_usage": last_3_sum,
                "trend_slope": round(slope, 2),
                "obsolete_declining_flags": ", ".join(flags)
            })

    return pd.DataFrame(rows)

# =========================================================
# OUTLIERS
# =========================================================
def detect_usage_outliers(usage_full):
    print("\nDetecting usage outliers...")
    rows = []

    for (model, part), grp in usage_full.groupby(["model", "part_number"]):
        grp = grp.sort_values("year_month")
        vals = grp["qty_used"].values.astype(float)

        avg = np.mean(vals)
        std = np.std(vals)

        if std == 0:
            continue

        for i, v in enumerate(vals):
            z = (v - avg) / std
            if abs(z) >= 3:
                rows.append({
                    "model": model,
                    "part_number": part,
                    "month": str(grp.iloc[i]["year_month"]),
                    "usage": v,
                    "avg_usage": round(avg, 2),
                    "std_usage": round(std, 2),
                    "z_score": round(z, 2),
                    "flag": "OUTLIER_USAGE"
                })

    return pd.DataFrame(rows)

# =========================================================
# LOAD STOCK
# =========================================================
def load_stock(cfg):
    print("\nLoading stock...")

    stock_path = Path(cfg["rc_stock_file"])
    if not stock_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de stock:\n{stock_path}")

    df = pd.read_excel(stock_path)
    df.columns = [normalize_text(c) for c in df.columns]

    df[cfg["stock_part_col"]] = df[cfg["stock_part_col"]].astype(str).map(normalize_text)
    df[cfg["stock_qty_col"]] = pd.to_numeric(df[cfg["stock_qty_col"]], errors="coerce").fillna(0)

    stock = (
        df.groupby(cfg["stock_part_col"])[cfg["stock_qty_col"]]
        .sum()
        .reset_index()
    )

    stock.columns = ["part_number", "total_stock"]

    print("Stock parts:", len(stock))
    return stock

# =========================================================
# CUSTOMER FORECAST
# =========================================================
def load_customer_forecast(cfg):
    print("\nLoading customer forecast...")

    cust_path = Path(cfg["customer_forecast_file"])
    if not cust_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de customer forecast:\n{cust_path}")

    df = pd.read_excel(cust_path)
    df.columns = [normalize_text(c) for c in df.columns]

    df[cfg["cust_model_col"]] = df[cfg["cust_model_col"]].astype(str).map(normalize_text)
    df[cfg["cust_month_col"]] = pd.to_datetime(df[cfg["cust_month_col"]], errors="coerce").dt.to_period("M")
    df[cfg["cust_units_col"]] = pd.to_numeric(df[cfg["cust_units_col"]], errors="coerce").fillna(0)
    df[cfg["cust_return_rate_col"]] = pd.to_numeric(df[cfg["cust_return_rate_col"]], errors="coerce").fillna(0)

    df = df[df[cfg["cust_month_col"]].notna()].copy()

    df["customer_effective_units"] = df[cfg["cust_units_col"]] * (
        df[cfg["cust_return_rate_col"]] / 100.0
    )

    out = df[[
        cfg["cust_model_col"],
        cfg["cust_month_col"],
        cfg["cust_units_col"],
        cfg["cust_return_rate_col"],
        "customer_effective_units"
    ]].copy()

    out.columns = [
        "model",
        "forecast_month",
        "forecast_units",
        "return_rate_pct",
        "customer_effective_units"
    ]

    print("Customer forecast rows:", len(out))
    return out

# =========================================================
# CUSTOMER RATIO
# =========================================================
def build_customer_ratio(usage_full, cust_df):
    print("\nBuilding customer ratio...")

    actual_model_month = (
        usage_full.groupby(["model", "year_month"])["qty_used"]
        .sum()
        .reset_index(name="actual_parts_usage_model")
    )

    actual_avg = (
        actual_model_month.groupby("model")["actual_parts_usage_model"]
        .mean()
        .reset_index(name="actual_avg_parts_usage_model")
    )

    cust_adj = cust_df.merge(actual_avg, on="model", how="left")
    cust_adj["actual_avg_parts_usage_model"] = cust_adj["actual_avg_parts_usage_model"].replace(0, np.nan)

    cust_adj["customer_ratio"] = (
        cust_adj["customer_effective_units"] /
        cust_adj["actual_avg_parts_usage_model"]
    )

    cust_adj["customer_ratio"] = (
        cust_adj["customer_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.50, upper=1.80)
    )

    return cust_adj

# =========================================================
# STATS
# =========================================================
def calculate_stats(usage_full):
    print("\nCalculating part stats...")

    stats = (
        usage_full.groupby(["model", "part_number"])["qty_used"]
        .agg(
            avg_usage="mean",
            std_usage="std",
            max_usage="max",
            min_usage="min",
            total_usage="sum",
            months_count="count"
        )
        .reset_index()
    )

    stats["std_usage"] = stats["std_usage"].fillna(0)

    slopes = []

    for (model, part), grp in usage_full.groupby(["model", "part_number"]):
        grp = grp.sort_values("year_month")
        slope = calc_slope(grp["qty_used"].tolist())
        last_month = grp["qty_used"].iloc[-1]
        slopes.append([model, part, slope, last_month])

    slopes_df = pd.DataFrame(
        slopes,
        columns=["model", "part_number", "trend_slope", "last_month_usage"]
    )

    stats = stats.merge(slopes_df, on=["model", "part_number"], how="left")

    stats["trend_direction"] = np.where(
        stats["trend_slope"] > 0.5, "increasing",
        np.where(stats["trend_slope"] < -0.5, "decreasing", "stable")
    )

    print("Stats rows:", len(stats))
    return stats

# =========================================================
# PPU
# =========================================================
def calculate_ppu(usage_full, cust_df):
    print("\nCalculating PPU...")

    usage_model = (
        usage_full.groupby("model")["qty_used"]
        .sum()
        .reset_index(name="parts_used")
    )

    units_model = (
        cust_df.groupby("model")["forecast_units"]
        .sum()
        .reset_index(name="units_forecast")
    )

    ppu_df = usage_model.merge(units_model, on="model", how="left")
    ppu_df["units_forecast"] = ppu_df["units_forecast"].fillna(0)

    ppu_df["PPU"] = np.where(
        ppu_df["units_forecast"] > 0,
        ppu_df["parts_used"] / ppu_df["units_forecast"],
        0
    )

    ppu_df["PPU"] = ppu_df["PPU"].replace([np.inf, -np.inf], np.nan).fillna(0)
    ppu_df["PPU"] = ppu_df["PPU"].round(4)

    print("PPU rows:", len(ppu_df))
    return ppu_df

# =========================================================
# ABC
# =========================================================
def classify_abc(stats_df):
    print("\nClassifying ABC...")

    df = stats_df.copy()
    df = df.sort_values("total_usage", ascending=False)

    total_usage = df["total_usage"].sum()

    if total_usage == 0:
        df["usage_cumsum"] = 0
        df["usage_pct"] = 0
        df["ABC_class"] = "C"
        return df

    df["usage_cumsum"] = df["total_usage"].cumsum()
    df["usage_pct"] = df["usage_cumsum"] / total_usage

    conditions = [
        df["usage_pct"] <= 0.80,
        (df["usage_pct"] > 0.80) & (df["usage_pct"] <= 0.95),
        df["usage_pct"] > 0.95
    ]

    df["ABC_class"] = np.select(conditions, ["A", "B", "C"], default="C")
    return df


# =========================================================
# IMPROVED FORECAST - CONSERVATIVE VERSION
# Corrige inflación por PPU
# =========================================================
def generate_forecast(stats, stock_df, cust_adj, cfg, usage_full, ppu_df,
                      rare_parts, obsolete_declining_parts, new_parts):

    print("\nGenerating conservative behavior-based forecast...")

    today_month = pd.Timestamp.today().to_period("M")
    forecast_months = [today_month + 1, today_month + 2, today_month + 3]

    rows = []

    rare_keys = set()
    obsolete_keys = set()
    new_keys = set()

    if rare_parts is not None and not rare_parts.empty:
        rare_keys = set(zip(rare_parts["model"], rare_parts["part_number"]))

    if obsolete_declining_parts is not None and not obsolete_declining_parts.empty:
        obsolete_keys = set(zip(obsolete_declining_parts["model"], obsolete_declining_parts["part_number"]))

    if new_parts is not None and not new_parts.empty:
        new_keys = set(new_parts["part_number"])

    for _, r in stats.iterrows():

        model = r["model"]
        part = r["part_number"]

        avg_usage = float(r["avg_usage"])
        std_usage = float(r["std_usage"])
        slope = float(r["trend_slope"])
        last_month_usage = float(r["last_month_usage"])
        total_usage = float(r["total_usage"])
        abc_class = r.get("ABC_class", "C")

        hist = usage_full[
            (usage_full["model"] == model) &
            (usage_full["part_number"] == part)
        ].sort_values("year_month")

        vals = hist["qty_used"].astype(float).tolist()

        if len(vals) == 0:
            vals = [0]

        weights = np.arange(1, len(vals) + 1)
        weighted_avg = np.average(vals, weights=weights)

        rolling_3m = np.mean(vals[-3:]) if len(vals) >= 3 else np.mean(vals)

        behavior_base = (
            0.50 * weighted_avg +
            0.30 * rolling_3m +
            0.20 * last_month_usage
        )

        # Control de spike
        if std_usage > 0 and last_month_usage > avg_usage + (3 * std_usage):
            behavior_base = avg_usage + std_usage

        # Share real de la parte dentro del modelo
        model_total_usage = stats[stats["model"] == model]["total_usage"].sum()

        if model_total_usage > 0:
            part_share = total_usage / model_total_usage
        else:
            part_share = 0

        for i, f_month in enumerate(forecast_months):

            month_offset = i + 1

            temp = cust_adj[
                (cust_adj["model"] == model) &
                (cust_adj["forecast_month"] == f_month)
            ]

            if not temp.empty:
                customer_ratio = float(temp["customer_ratio"].iloc[0])
                forecast_units = float(temp["forecast_units"].iloc[0])
                return_rate = float(temp["return_rate_pct"].iloc[0])
            else:
                customer_ratio = 1.0
                forecast_units = np.nan
                return_rate = np.nan

            # =====================================================
            # CAMBIO IMPORTANTE:
            # Ya NO aplicamos PPU directo a cada parte.
            # Ahora el customer forecast solo ajusta suavemente.
            # =====================================================

            trend_adj = slope * month_offset * 0.30  # antes era 0.60, ahora más conservador

            demand_base = max(behavior_base + trend_adj, 0)

            # Si el forecast del cliente baja, no dejamos que suba artificialmente
            if customer_ratio < 1:
                customer_adjustment = 1 - ((1 - customer_ratio) * 0.50)
            else:
                customer_adjustment = 1 + ((customer_ratio - 1) * 0.20)

            demand_after_customer = demand_base * customer_adjustment

            # Safety stock más controlado
            if abc_class == "A":
                safety_factor = 0.80
            elif abc_class == "B":
                safety_factor = 0.60
            else:
                safety_factor = 0.40

            safety_stock = std_usage * safety_factor

            final_fcst = demand_after_customer + safety_stock

            # Reglas de negocio
            if (model, part) in obsolete_keys:
                final_fcst *= 0.20

            if (model, part) in rare_keys:
                final_fcst *= 0.40

            if part in new_keys:
                final_fcst = max(final_fcst * 1.15, 1)

            if len(vals) >= 3 and sum(vals[-3:]) == 0 and abc_class != "A":
                final_fcst *= 0.20

            if slope >= 1:
                final_fcst *= 1.10

            # =====================================================
            # CAP DE SEGURIDAD
            # Evita que una parte con bajo consumo explote.
            # =====================================================
            historical_cap = max(
                last_month_usage * 2.0,
                rolling_3m * 2.0,
                avg_usage * 2.5,
                1
            )

            final_fcst = min(final_fcst, historical_cap)

            final_fcst = max(final_fcst, 0)

            rows.append({
                "model": model,
                "part_number": part,
                "forecast_month": f_month,
                "month_offset": month_offset,

                "avg_usage_8m": round(avg_usage, 2),
                "weighted_avg_usage": round(weighted_avg, 2),
                "rolling_3m_usage": round(rolling_3m, 2),
                "last_month_usage": round(last_month_usage, 2),
                "total_usage_8m": round(total_usage, 2),

                "std_usage_8m": round(std_usage, 2),
                "trend_slope": round(slope, 2),
                "trend_direction": r["trend_direction"],
                "ABC_class": abc_class,

                "part_share_in_model": round(part_share, 4),
                "behavior_base": round(behavior_base, 2),
                "forecast_units": forecast_units,
                "return_rate_pct": return_rate,
                "customer_ratio": round(customer_ratio, 2),
                "customer_adjustment": round(customer_adjustment, 2),

                "trend_adjustment": round(trend_adj, 2),
                "safety_stock": round(safety_stock, 2),
                "historical_cap": round(historical_cap, 2),

                "rare_part_flag": (model, part) in rare_keys,
                "obsolete_declining_flag": (model, part) in obsolete_keys,
                "new_part_flag": part in new_keys,

                "forecast_demand": int(np.ceil(final_fcst))
            })

    fcst = pd.DataFrame(rows)

    fcst = fcst.merge(stock_df, on="part_number", how="left")
    fcst["total_stock"] = fcst["total_stock"].fillna(0)

    fcst["net_requirement"] = np.maximum(
        fcst["forecast_demand"] - fcst["total_stock"],
        0
    )

    month1 = fcst[fcst["month_offset"] == 1].copy()

    month1["priority"] = month1.apply(
        lambda x: classify_priority(x["total_stock"], x["forecast_demand"]),
        axis=1
    )

    fcst = fcst.merge(
        month1[["model", "part_number", "priority"]],
        on=["model", "part_number"],
        how="left"
    )

    print("Conservative forecast rows:", len(fcst))
    return fcst
# =========================================================
# EXPORT
# =========================================================
def export_all(output_dir, claims_recent, usage_full, new_parts, rare_parts,
               obsolete_declining_parts, outliers_df, ppu_df, stats_df,
               critical_parts, forecast_df):

    print("\nExporting results...")

    out_dir = ensure_dir(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"spare_parts_forecast_improved_{ts}.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:

        claims_recent.to_excel(writer, sheet_name="Claims_Last_8M", index=False)
        usage_full.to_excel(writer, sheet_name="Monthly_Usage_8M", index=False)
        new_parts.to_excel(writer, sheet_name="New_Parts_First_Use", index=False)
        rare_parts.to_excel(writer, sheet_name="Rare_Parts", index=False)
        obsolete_declining_parts.to_excel(writer, sheet_name="Obsolete_Declining", index=False)
        outliers_df.to_excel(writer, sheet_name="Usage_Outliers", index=False)
        ppu_df.to_excel(writer, sheet_name="PPU_By_Model", index=False)
        stats_df.to_excel(writer, sheet_name="Part_Stats_ABC", index=False)
        critical_parts.to_excel(writer, sheet_name="Critical_Parts", index=False)
        forecast_df.to_excel(writer, sheet_name="Full_Forecast_3M", index=False)

        monthly_summary = (
            forecast_df.groupby("forecast_month")
            .agg(
                forecast_demand=("forecast_demand", "sum"),
                total_stock=("total_stock", "sum"),
                net_requirement=("net_requirement", "sum"),
                unique_parts=("part_number", "nunique")
            )
            .reset_index()
        )

        monthly_summary.to_excel(writer, sheet_name="Monthly_Summary", index=False)

        critical_high = forecast_df[
            (forecast_df["month_offset"] == 1) &
            (forecast_df["priority"].isin(["CRITICAL", "HIGH"]))
        ].copy()

        critical_high.to_excel(writer, sheet_name="Critical_High_M1", index=False)

        top_m1 = forecast_df[forecast_df["month_offset"] == 1].copy()
        top_m1 = top_m1.sort_values("forecast_demand", ascending=False).head(100)

        top_m1.to_excel(writer, sheet_name="Top100_M1", index=False)

    print("Excel generated successfully")
    print("Saved in:", out_file)

    return out_file
# =========================================================
# CRITICAL PARTS
# =========================================================
def detect_critical_parts(forecast_df):
    print("\nDetecting critical parts...")

    df = forecast_df[forecast_df["month_offset"] == 1].copy()
    flags = []

    for _, row in df.iterrows():
        part_flags = []

        if row["forecast_demand"] > 0 and row["total_stock"] == 0:
            part_flags.append("NO_STOCK_WITH_DEMAND")

        if row["net_requirement"] > 0:
            part_flags.append("SHORTAGE")

        if row["trend_direction"] == "increasing" and row["net_requirement"] > 0:
            part_flags.append("INCREASING_AND_SHORT")

        if row["forecast_demand"] >= 20:
            part_flags.append("HIGH_FORECAST")

        if row["ABC_class"] == "A":
            part_flags.append("ABC_A")

        if part_flags:
            flags.append({
                "model": row["model"],
                "part_number": row["part_number"],
                "forecast_month": row["forecast_month"],
                "forecast_demand": row["forecast_demand"],
                "total_stock": row["total_stock"],
                "net_requirement": row["net_requirement"],
                "trend_direction": row["trend_direction"],
                "ABC_class": row["ABC_class"],
                "priority": row["priority"],
                "critical_flags": ", ".join(part_flags)
            })

    critical_df = pd.DataFrame(flags)

    if not critical_df.empty:
        priority_order = {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3
        }

        critical_df["priority_order"] = (
            critical_df["priority"]
            .map(priority_order)
            .fillna(9)
        )

        critical_df = (
            critical_df
            .sort_values(
                ["priority_order", "net_requirement", "forecast_demand"],
                ascending=[True, False, False]
            )
            .drop(columns="priority_order")
        )

    print("Critical parts detected:", len(critical_df))
    return critical_df

# =========================================================
# MAIN
# =========================================================
def main(cfg):

    print("=" * 80)
    print("SPARE PARTS FORECASTING TOOL - IMPROVED VERSION")
    print("=" * 80)

    validate_input_files(cfg)

    claims_all = load_claims(cfg)

    claims_recent, min_month, max_month = filter_last_months(
        claims_all,
        cfg["lookback_months"]
    )

    usage_full = build_monthly_usage(claims_recent, min_month, max_month)

    new_parts = detect_new_parts(claims_all, claims_recent)
    rare_parts = detect_rare_parts(usage_full)
    obsolete_declining_parts = detect_obsolete_or_declining_parts(usage_full)
    outliers_df = detect_usage_outliers(usage_full)

    stock_df = load_stock(cfg)
    cust_df = load_customer_forecast(cfg)
    cust_adj = build_customer_ratio(usage_full, cust_df)

    stats = calculate_stats(usage_full)
    stats = classify_abc(stats)

    ppu_df = calculate_ppu(usage_full, cust_df)

    forecast_df = generate_forecast(
        stats=stats,
        stock_df=stock_df,
        cust_adj=cust_adj,
        cfg=cfg,
        usage_full=usage_full,
        ppu_df=ppu_df,
        rare_parts=rare_parts,
        obsolete_declining_parts=obsolete_declining_parts,
        new_parts=new_parts
    )

    critical_parts = detect_critical_parts(forecast_df)

    output_file = export_all(
        cfg["output_dir"],
        claims_recent,
        usage_full,
        new_parts,
        rare_parts,
        obsolete_declining_parts,
        outliers_df,
        ppu_df,
        stats,
        critical_parts,
        forecast_df
    )

    print("\nSUMMARY")
    print("-" * 80)
    print("Recent claims rows:", len(claims_recent))
    print("Unique parts:", claims_recent["part_number"].nunique())
    print("Unique models:", claims_recent["model"].nunique())
    print("New parts:", len(new_parts))
    print("Rare parts:", len(rare_parts))
    print("Obsolete or declining parts:", len(obsolete_declining_parts))
    print("Usage outliers:", len(outliers_df))
    print("PPU models:", len(ppu_df))
    print("Critical parts:", len(critical_parts))
    print("\nFinal file:")
    print(output_file)

    return (
        claims_recent,
        usage_full,
        new_parts,
        rare_parts,
        obsolete_declining_parts,
        outliers_df,
        ppu_df,
        stats,
        critical_parts,
        forecast_df,
        output_file
    )

# =========================================================
# RUN
# =========================================================
claims_recent, usage_full, new_parts_df, rare_parts_df, obsolete_declining_df, outliers_df, ppu_df, stats_df, critical_parts_df, forecast_df, output_file = main(CONFIG)