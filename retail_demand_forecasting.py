"""
Adaptive Retail Demand Forecasting & Inventory Planning Framework
=================================================================
NIT Calicut – B.Tech Production Engineering Project, May 2026
Authors : Kanhaiya Singh, Niraj Kumar, Pavan S, Rohit Akhilesh
Guides  : Dr. Sajan T John, Dr. Sanghamitra Das

Dataset : M5 Walmart Retail Dataset (FOODS_1 department, CA_1 store)
          https://www.kaggle.com/competitions/m5-forecasting-accuracy

Framework steps
---------------
1.  Data loading & weekly aggregation
2.  Demand structure analysis (STL, ACF, ADF, ADI-CV²)
3.  Adaptive model routing
4.  Rolling forecast evaluation (Naive, MA, Holt, Holt-Winters,
    ARIMA, Croston, SBA, TSB, Random Forest, LightGBM)
5.  Forecast reliability assessment (MASE thresholds)
6.  Inventory planning (safety stock, reorder point, ABC-XYZ)
"""

# ─────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

# SKUs selected from FOODS_1 department, CA_1 store
TARGET_SKUS = [
    "FOODS_1_001_CA_1",
    "FOODS_1_004_CA_1",
    "FOODS_1_005_CA_1",
    "FOODS_1_008_CA_1",
]

# Rolling forecast test window (last N weeks held out)
TEST_WEEKS = 52

# Lead time assumption (weeks) for inventory planning
LEAD_TIME = 1

# STL seasonal period (weekly data → annual cycle = 52 weeks)
STL_PERIOD = 52

# ADI-CV² classification thresholds (Syntetos & Boylan, 2005)
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

# MASE reliability thresholds
MASE_HIGH = 1.0
MASE_MODERATE = 1.5

# Service level Z-scores mapped to forecast reliability
SERVICE_LEVEL_MAP = {
    "High":     {"service_level": 0.90, "z": 1.28},
    "Moderate": {"service_level": 0.95, "z": 1.65},
    "Low":      {"service_level": 0.98, "z": 2.05},
}

# ABC classification thresholds (cumulative demand share)
ABC_A_THRESHOLD = 0.70
ABC_B_THRESHOLD = 0.90

# XYZ classification thresholds (CV of demand)
XYZ_X_THRESHOLD = 0.30
XYZ_Y_THRESHOLD = 0.60


# ─────────────────────────────────────────────
# 2. Data Loading & Weekly Aggregation
# ─────────────────────────────────────────────

def load_m5_data(sales_path: str, calendar_path: str) -> dict[str, pd.Series]:
    """
    Load M5 sales and calendar files, filter to TARGET_SKUS,
    and return a dict of {sku_id: weekly_demand_series}.

    Parameters
    ----------
    sales_path    : path to sales_train_evaluation.csv
    calendar_path : path to calendar.csv
    """
    print("Loading M5 dataset …")
    sales_df    = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)

    # Filter to our 4 SKUs in CA_1 / FOODS_1
    mask = (
        sales_df["item_id"].isin([s.rsplit("_CA_1", 1)[0] + "" for s in TARGET_SKUS])
        | sales_df["id"].isin(TARGET_SKUS)
    )
    sales_df = sales_df[sales_df["id"].isin(TARGET_SKUS)].copy()

    # Build date index from calendar
    day_cols   = [c for c in sales_df.columns if c.startswith("d_")]
    date_index = (
        calendar_df.set_index("d")
        .loc[day_cols, "date"]
        .pipe(pd.to_datetime)
    )

    weekly_series = {}
    for sku in TARGET_SKUS:
        row      = sales_df[sales_df["id"] == sku][day_cols].iloc[0]
        ts_daily = pd.Series(row.values.astype(float), index=date_index, name=sku)
        # Aggregate to weekly sums (Monday anchor)
        ts_weekly = ts_daily.resample("W-MON").sum()
        weekly_series[sku] = ts_weekly
        print(f"  {sku}: {len(ts_weekly)} weekly observations")

    return weekly_series


# ─────────────────────────────────────────────
# 3. Demand Structure Analysis
# ─────────────────────────────────────────────

def compute_adi_cv2(series: pd.Series) -> tuple[float, float, str]:
    """
    Compute ADI and CV² for a demand series and return demand type.

    Returns
    -------
    (adi, cv2, demand_type)
    demand_type ∈ {"Smooth", "Erratic", "Intermittent", "Lumpy"}
    """
    nonzero = series[series > 0]
    n_total  = len(series)
    n_nonzero = len(nonzero)

    if n_nonzero == 0:
        return np.inf, np.inf, "Lumpy"

    adi = n_total / n_nonzero
    mu  = nonzero.mean()
    cv2 = (nonzero.std() / mu) ** 2 if mu > 0 else 0.0

    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        demand_type = "Smooth"
    elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        demand_type = "Erratic"
    elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        demand_type = "Intermittent"
    else:
        demand_type = "Lumpy"

    return round(adi, 3), round(cv2, 3), demand_type


def stl_strength(series: pd.Series, period: int = STL_PERIOD) -> tuple[float, float]:
    """
    Decompose series with STL and compute trend + seasonal strength.

    Strength formula (Wang et al., 2006):
        F_trend    = max(0, 1 - Var(R) / Var(T + R))
        F_seasonal = max(0, 1 - Var(R) / Var(S + R))
    """
    # STL requires at least 2 full periods; pad if needed
    if len(series) < 2 * period:
        period = max(4, len(series) // 2)

    stl    = STL(series, period=period, robust=True)
    result = stl.fit()

    T, S, R = result.trend, result.seasonal, result.resid

    var_r   = np.var(R)
    f_trend = max(0.0, 1.0 - var_r / np.var(T + R)) if np.var(T + R) > 0 else 0.0
    f_seas  = max(0.0, 1.0 - var_r / np.var(S + R)) if np.var(S + R) > 0 else 0.0

    return round(f_trend, 3), round(f_seas, 3)


def interpret_strength(value: float) -> str:
    if value < 0.30:
        return "Weak"
    elif value <= 0.60:
        return "Moderate"
    else:
        return "Strong"


def run_adf_test(series: pd.Series) -> tuple[float, str]:
    """Augmented Dickey-Fuller stationarity test. p < 0.05 → Stationary."""
    result  = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    status  = "Stationary" if p_value < 0.05 else "Non-Stationary"
    return round(p_value, 4), status


def max_significant_acf(series: pd.Series, nlags: int = 20) -> float:
    """Return the maximum ACF value outside the 95% confidence band."""
    n        = len(series)
    ci_bound = 1.96 / np.sqrt(n)
    acf_vals = acf(series.dropna(), nlags=nlags, fft=True)
    # Skip lag 0 (always 1.0)
    outside  = [abs(v) for v in acf_vals[1:] if abs(v) > ci_bound]
    return round(max(outside), 3) if outside else 0.0


def interpret_acf(max_acf: float) -> str:
    if max_acf < 0.3:
        return "Weak"
    elif max_acf < 0.6:
        return "Moderate"
    else:
        return "Strong"


def analyse_all_skus(weekly_series: dict) -> pd.DataFrame:
    """Run full demand analysis for all SKUs and return summary DataFrame."""
    rows = []
    for sku, ts in weekly_series.items():
        adi, cv2, demand_type = compute_adi_cv2(ts)
        f_trend, f_seas       = stl_strength(ts)
        p_val, stationarity   = run_adf_test(ts)
        max_acf               = max_significant_acf(ts)

        rows.append({
            "SKU":              sku,
            "ADI":              adi,
            "CV²":              cv2,
            "Demand Type":      demand_type,
            "Trend Strength":   f_trend,
            "Trend Type":       interpret_strength(f_trend),
            "Seasonal Strength":f_seas,
            "Seasonal Type":    interpret_strength(f_seas),
            "Max ACF":          max_acf,
            "Dependency":       interpret_acf(max_acf),
            "ADF p-value":      p_val,
            "Stationarity":     stationarity,
        })

    df = pd.DataFrame(rows)
    print("\n── Demand Structure Summary ──")
    print(df[["SKU", "Demand Type", "Trend Type", "Seasonal Type",
              "Dependency", "Stationarity"]].to_string(index=False))
    return df


# ─────────────────────────────────────────────
# 4. Adaptive Model Routing
# ─────────────────────────────────────────────

def select_candidate_models(demand_row: pd.Series) -> list[str]:
    """
    Return a list of candidate forecasting model names
    based on ADI-CV² type, dependency, and stationarity.
    """
    d_type       = demand_row["Demand Type"]
    dependency   = demand_row["Dependency"]
    stationarity = demand_row["Stationarity"]

    candidates = ["Naive"]  # Naive is always included as baseline

    if d_type in ("Smooth", "Erratic"):
        candidates += ["MA", "Holt", "HoltWinters", "RF", "LightGBM"]
        if dependency in ("Moderate", "Strong"):
            candidates.append("ARIMA")

    if d_type == "Lumpy":
        candidates += ["SBA", "TSB", "RF", "LightGBM"]

    if d_type == "Intermittent":
        candidates += ["Croston", "SBA", "TSB", "RF", "LightGBM"]

    if stationarity == "Non-Stationary" and "Holt" not in candidates:
        candidates += ["Holt", "HoltWinters"]

    return list(dict.fromkeys(candidates))  # preserve order, remove dupes


# ─────────────────────────────────────────────
# 5. Forecasting Models
# ─────────────────────────────────────────────

def naive_forecast(train: pd.Series) -> float:
    """Last observed value."""
    return float(train.iloc[-1])


def ma_forecast(train: pd.Series, window: int = 4) -> float:
    """Simple moving average over last `window` periods."""
    return float(train.iloc[-window:].mean())


def holt_forecast(train: pd.Series) -> float:
    """Holt's linear trend method (double exponential smoothing)."""
    model = ExponentialSmoothing(train, trend="add", seasonal=None)
    fit   = model.fit(optimized=True, use_brute=False)
    return float(fit.forecast(1).iloc[0])


def holtwinters_forecast(train: pd.Series) -> float:
    """Holt-Winters additive seasonal method."""
    period = min(STL_PERIOD, len(train) // 2)
    try:
        model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                     seasonal_periods=period)
        fit   = model.fit(optimized=True, use_brute=False)
        return float(fit.forecast(1).iloc[0])
    except Exception:
        # Fall back to Holt if seasonal fit fails
        return holt_forecast(train)


def arima_forecast(train: pd.Series) -> float:
    """ARIMA(1,1,1) — auto-differencing + AR(1) + MA(1)."""
    try:
        model = ARIMA(train, order=(1, 1, 1))
        fit   = model.fit()
        return float(fit.forecast(1).iloc[0])
    except Exception:
        return naive_forecast(train)


def croston_forecast(train: pd.Series, alpha: float = 0.1) -> float:
    """
    Croston's method for intermittent demand.
    Separately smooths demand size (z) and inter-arrival interval (p).
    Forecast = z / p
    """
    z, p = 1.0, 1.0
    for val in train:
        if val > 0:
            z = alpha * val + (1 - alpha) * z
            p = alpha * 1   + (1 - alpha) * p
        else:
            p = alpha * (p + 1) + (1 - alpha) * p  # increment interval
    return max(0.0, z / p)


def sba_forecast(train: pd.Series, alpha: float = 0.1) -> float:
    """
    Syntetos-Boylan Approximation — bias-corrected Croston.
    Forecast = (1 - alpha/2) * z / p
    """
    z, p = 1.0, 1.0
    for val in train:
        if val > 0:
            z = alpha * val + (1 - alpha) * z
            p = alpha * 1   + (1 - alpha) * p
        else:
            p = alpha * (p + 1) + (1 - alpha) * p
    return max(0.0, (1 - alpha / 2) * z / p)


def tsb_forecast(train: pd.Series, alpha: float = 0.1, beta: float = 0.1) -> float:
    """
    Teunter-Syntetos-Babai (TSB) method.
    Smooths demand probability (pt) and mean demand size (zt).
    Forecast = pt * zt
    """
    pt = 0.5
    zt = train[train > 0].mean() if (train > 0).any() else 1.0

    for val in train:
        if val > 0:
            pt = alpha * 1   + (1 - alpha) * pt
            zt = beta  * val + (1 - beta)  * zt
        else:
            pt = alpha * 0   + (1 - alpha) * pt

    return max(0.0, pt * zt)


def _make_lag_features(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised learning features from a time series.
    Features: lag_1, lag_2, lag_4, rolling_mean_4, rolling_std_4
    """
    df = pd.DataFrame({"y": series})
    df["lag_1"]        = df["y"].shift(1)
    df["lag_2"]        = df["y"].shift(2)
    df["lag_4"]        = df["y"].shift(4)
    df["rolling_mean"] = df["y"].shift(1).rolling(4).mean()
    df["rolling_std"]  = df["y"].shift(1).rolling(4).std()
    df.dropna(inplace=True)

    X = df[["lag_1", "lag_2", "lag_4", "rolling_mean", "rolling_std"]].values
    y = df["y"].values
    return X, y


def rf_forecast(train: pd.Series) -> float:
    """Random Forest one-step-ahead forecast using lag features."""
    if len(train) < 8:
        return naive_forecast(train)
    X, y = _make_lag_features(train)
    if len(X) < 5:
        return naive_forecast(train)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Build feature vector for the next step
    lag1  = float(train.iloc[-1])
    lag2  = float(train.iloc[-2])
    lag4  = float(train.iloc[-4])
    rmean = float(train.iloc[-4:].mean())
    rstd  = float(train.iloc[-4:].std())
    X_new = np.array([[lag1, lag2, lag4, rmean, rstd]])
    return max(0.0, float(model.predict(X_new)[0]))


def lgbm_forecast(train: pd.Series) -> float:
    """LightGBM one-step-ahead forecast using lag features."""
    if len(train) < 8:
        return naive_forecast(train)
    X, y = _make_lag_features(train)
    if len(X) < 5:
        return naive_forecast(train)

    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1,
                               max_depth=4, random_state=42, verbose=-1)
    model.fit(X, y)

    lag1  = float(train.iloc[-1])
    lag2  = float(train.iloc[-2])
    lag4  = float(train.iloc[-4])
    rmean = float(train.iloc[-4:].mean())
    rstd  = float(train.iloc[-4:].std())
    X_new = np.array([[lag1, lag2, lag4, rmean, rstd]])
    return max(0.0, float(model.predict(X_new)[0]))


# Dispatcher: model name → function
MODEL_FUNCTIONS = {
    "Naive":       naive_forecast,
    "MA":          ma_forecast,
    "Holt":        holt_forecast,
    "HoltWinters": holtwinters_forecast,
    "ARIMA":       arima_forecast,
    "Croston":     croston_forecast,
    "SBA":         sba_forecast,
    "TSB":         tsb_forecast,
    "RF":          rf_forecast,
    "LightGBM":    lgbm_forecast,
}


# ─────────────────────────────────────────────
# 6. Evaluation Metrics
# ─────────────────────────────────────────────

def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mase(actual: np.ndarray, predicted: np.ndarray,
                 train: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error.
    Scale = MAE of naïve in-sample one-step forecast (seasonal naive with h=1).
    """
    naive_errors = np.abs(np.diff(train))
    scale = naive_errors.mean()
    if scale == 0:
        return np.inf
    return float(np.mean(np.abs(actual - predicted)) / scale)


# ─────────────────────────────────────────────
# 7. Rolling Forecast Evaluation
# ─────────────────────────────────────────────

def rolling_evaluate(series: pd.Series, model_name: str,
                     test_size: int = TEST_WEEKS) -> dict:
    """
    Expanding-window rolling forecast evaluation.
    Trains on all history before t, forecasts t, advances by 1.
    Returns dict with MAE, RMSE, MASE.
    """
    n        = len(series)
    train_end = n - test_size
    values   = series.values

    if train_end < 10:
        return {"model": model_name, "MAE": np.inf,
                "RMSE": np.inf, "MASE": np.inf}

    forecast_fn = MODEL_FUNCTIONS[model_name]
    actuals     = []
    forecasts   = []

    for t in range(train_end, n):
        train = pd.Series(values[:t])
        try:
            pred = forecast_fn(train)
        except Exception:
            pred = naive_forecast(train)
        pred = max(0.0, pred)  # demand cannot be negative
        actuals.append(values[t])
        forecasts.append(pred)

    actual_arr = np.array(actuals)
    pred_arr   = np.array(forecasts)
    train_arr  = values[:train_end]

    return {
        "model": model_name,
        "MAE":   round(compute_mae(actual_arr, pred_arr), 3),
        "RMSE":  round(compute_rmse(actual_arr, pred_arr), 3),
        "MASE":  round(compute_mase(actual_arr, pred_arr, train_arr), 3),
    }


def evaluate_sku(sku: str, series: pd.Series, candidates: list[str]) -> pd.DataFrame:
    """Evaluate all candidate models for one SKU and return results DataFrame."""
    print(f"\n  Evaluating {sku} ({len(candidates)} models) …")
    results = []
    for model in candidates:
        res = rolling_evaluate(series, model)
        results.append(res)
        print(f"    {model:12s}  MASE={res['MASE']:.3f}  "
              f"MAE={res['MAE']:.3f}  RMSE={res['RMSE']:.3f}")

    df = pd.DataFrame(results).sort_values("MASE").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 8. Forecast Reliability
# ─────────────────────────────────────────────

def assess_reliability(mase: float) -> str:
    if mase < MASE_HIGH:
        return "High"
    elif mase < MASE_MODERATE:
        return "Moderate"
    else:
        return "Low"


# ─────────────────────────────────────────────
# 9. Inventory Planning
# ─────────────────────────────────────────────

def compute_safety_stock(series: pd.Series, reliability: str,
                         lead_time: int = LEAD_TIME) -> float:
    """
    Safety Stock = Z × σ_d × √L
    Z and service level are selected based on forecast reliability.
    """
    z    = SERVICE_LEVEL_MAP[reliability]["z"]
    sigma = series.std()
    ss   = z * sigma * np.sqrt(lead_time)
    return round(ss, 1)


def compute_reorder_point(series: pd.Series, safety_stock: float,
                          lead_time: int = LEAD_TIME) -> float:
    """
    ROP = (Average Demand × Lead Time) + Safety Stock
    """
    avg_demand = series.mean()
    rop = avg_demand * lead_time + safety_stock
    return round(rop, 1)


def classify_abc(sku_demands: dict[str, float]) -> dict[str, str]:
    """
    ABC classification by cumulative demand contribution.
    A: top 70%, B: next 20%, C: remaining 10%
    """
    total     = sum(sku_demands.values())
    sorted_skus = sorted(sku_demands.items(), key=lambda x: x[1], reverse=True)
    cumulative  = 0.0
    result      = {}

    for sku, demand in sorted_skus:
        cumulative += demand / total
        if cumulative <= ABC_A_THRESHOLD:
            result[sku] = "A"
        elif cumulative <= ABC_B_THRESHOLD:
            result[sku] = "B"
        else:
            result[sku] = "C"

    return result


def classify_xyz(series: pd.Series) -> str:
    """
    XYZ classification by coefficient of variation (CV).
    X: CV ≤ 0.30 (stable), Y: 0.30 < CV ≤ 0.60 (moderate), Z: CV > 0.60
    """
    mu = series.mean()
    if mu == 0:
        return "Z"
    cv = series.std() / mu
    if cv <= XYZ_X_THRESHOLD:
        return "X"
    elif cv <= XYZ_Y_THRESHOLD:
        return "Y"
    else:
        return "Z"


INVENTORY_STRATEGY = {
    "AX": "Aggressive stocking and premium allocation",
    "AY": "Controlled monitoring and balanced replenishment",
    "AZ": "Higher safety stock and close monitoring",
    "BX": "Standard replenishment planning",
    "BY": "Balanced inventory control",
    "BZ": "Cautious stocking strategy",
    "CX": "Limited replenishment priority",
    "CY": "Conservative inventory monitoring",
    "CZ": "Minimal stocking priority",
}


def build_inventory_plan(weekly_series: dict[str, pd.Series],
                         best_results: dict[str, dict]) -> pd.DataFrame:
    """Build full inventory plan for all SKUs."""
    # ABC classification using mean weekly demand as proxy for demand volume
    sku_mean_demand = {sku: ts.mean() for sku, ts in weekly_series.items()}
    abc_classes     = classify_abc(sku_mean_demand)

    rows = []
    for sku, ts in weekly_series.items():
        res         = best_results[sku]
        reliability = assess_reliability(res["MASE"])
        ss          = compute_safety_stock(ts, reliability)
        rop         = compute_reorder_point(ts, ss)
        xyz         = classify_xyz(ts)
        abc         = abc_classes[sku]
        category    = abc + xyz
        strategy    = INVENTORY_STRATEGY.get(category, "Standard replenishment")

        rows.append({
            "SKU":             sku,
            "Best Model":      res["model"],
            "MASE":            res["MASE"],
            "Reliability":     reliability,
            "Safety Stock":    ss,
            "Reorder Point":   rop,
            "ABC":             abc,
            "XYZ":             xyz,
            "Category":        category,
            "Strategy":        strategy,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 10. Visualisation Helpers
# ─────────────────────────────────────────────

def plot_demand_overview(weekly_series: dict[str, pd.Series],
                         save_path: str = "demand_overview.png") -> None:
    """Plot weekly demand for all 4 SKUs in a 2×2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, (sku, ts) in enumerate(weekly_series.items()):
        axes[i].plot(ts.values, color="#1A73E8", linewidth=0.9)
        axes[i].set_title(sku, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Week Number")
        axes[i].set_ylabel("Weekly Sales")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Weekly Demand Visualisation — M5 FOODS_1 / CA_1", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_stl_decomposition(sku: str, series: pd.Series,
                            save_path: str = None) -> None:
    """Plot STL decomposition for a single SKU."""
    period = min(STL_PERIOD, len(series) // 2)
    stl    = STL(series, period=period, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    components = [series.values, stl.trend, stl.seasonal, stl.resid]
    labels     = ["Original Demand", "Trend Component",
                  "Seasonal Component", "Residual Component"]

    for ax, comp, label in zip(axes, components, labels):
        ax.plot(comp, linewidth=0.9, color="#1A73E8")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[3].scatter(range(len(stl.resid)), stl.resid,
                    s=8, color="#1A73E8", alpha=0.6)
    axes[0].set_title(f"STL Decomposition — {sku}", fontweight="bold")
    axes[-1].set_xlabel("Week Number")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────
# 11. Main Pipeline
# ─────────────────────────────────────────────

def run_pipeline(sales_path: str, calendar_path: str) -> None:
    """
    End-to-end pipeline execution.

    Parameters
    ----------
    sales_path    : path to M5 sales_train_evaluation.csv
    calendar_path : path to M5 calendar.csv
    """

    # ── Step 1: Load data ──────────────────────────────────────────────
    weekly_series = load_m5_data(sales_path, calendar_path)

    # ── Step 2: Demand structure analysis ─────────────────────────────
    print("\n── Step 2: Demand Structure Analysis ──")
    analysis_df = analyse_all_skus(weekly_series)

    # ── Step 3: Adaptive model routing ────────────────────────────────
    print("\n── Step 3: Adaptive Model Routing ──")
    candidate_map = {}
    for _, row in analysis_df.iterrows():
        candidates = select_candidate_models(row)
        candidate_map[row["SKU"]] = candidates
        print(f"  {row['SKU']} ({row['Demand Type']}): {candidates}")

    # ── Step 4: Rolling forecast evaluation ───────────────────────────
    print("\n── Step 4: Rolling Forecast Evaluation ──")
    all_results  = {}
    best_results = {}

    for sku in TARGET_SKUS:
        results_df         = evaluate_sku(sku, weekly_series[sku],
                                          candidate_map[sku])
        all_results[sku]   = results_df
        best_row           = results_df.iloc[0]
        best_results[sku]  = {
            "model": best_row["model"],
            "MASE":  best_row["MASE"],
            "MAE":   best_row["MAE"],
            "RMSE":  best_row["RMSE"],
        }

    print("\n── Best Model Per SKU ──")
    summary_rows = []
    for sku, res in best_results.items():
        reliability = assess_reliability(res["MASE"])
        summary_rows.append({
            "SKU": sku, "Best Model": res["model"],
            "RMSE": res["RMSE"], "MAE": res["MAE"],
            "MASE": res["MASE"], "Reliability": reliability,
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # ── Step 5: Inventory planning ────────────────────────────────────
    print("\n── Step 5: Inventory Planning ──")
    inventory_df = build_inventory_plan(weekly_series, best_results)
    print(inventory_df[["SKU", "Safety Stock", "Reorder Point",
                          "Category", "Strategy"]].to_string(index=False))

    # ── Step 6: Visualisations ────────────────────────────────────────
    print("\n── Step 6: Visualisations ──")
    plot_demand_overview(weekly_series, save_path="demand_overview.png")

    for sku, ts in weekly_series.items():
        fname = f"stl_{sku}.png"
        plot_stl_decomposition(sku, ts, save_path=fname)

    # ── Save outputs ──────────────────────────────────────────────────
    analysis_df.to_csv("demand_analysis_results.csv", index=False)
    summary_df.to_csv("forecast_results.csv", index=False)
    inventory_df.to_csv("inventory_plan.csv", index=False)
    print("\nOutputs saved: demand_analysis_results.csv, "
          "forecast_results.csv, inventory_plan.csv")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        run_pipeline(
            sales_path    = sys.argv[1],
            calendar_path = sys.argv[2],
        )
    else:
        print(
            "Usage:\n"
            "  python retail_demand_forecasting.py "
            "<sales_train_evaluation.csv> <calendar.csv>\n\n"
            "Download the M5 dataset from:\n"
            "  https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"
        )
