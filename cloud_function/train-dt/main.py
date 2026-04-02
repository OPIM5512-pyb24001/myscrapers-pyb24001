# Random Forest: train on all data before latest local day; hold out latest local day
# HTTP entrypoint: train_dt_http
# Keeps the same output structure so your existing GitHub sync workflow still works:
# structured/preds/<YYYYMMDDHH>/preds.csv

import os, io, json, logging, traceback
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "structured/preds")   # keep this for existing sync YAML
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")


def _clean_numeric(s: pd.Series) -> pd.Series:
    # Strip $, commas, spaces; keep digits and dot
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = [
        "make", "model", "transmission", "fuel_type", "drive_type",
        "title_status", "condition", "color", "city", "state"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip().str.lower()
    return df


def _clean_make(s: pd.Series) -> pd.Series:
    return s.replace({
        "chevy": "chevrolet",
        "hyundia": "hyundai",
        "vw": "volkswagen"
    })


def _clean_drive_type(s: pd.Series) -> pd.Series:
    drive_map = {
        "awd": "awd",
        "4wd": "awd",
        "fwd": "fwd",
        "rwd": "rwd"
    }
    cleaned = s.map(drive_map)
    return cleaned.fillna("unknown")


def _clean_condition(s: pd.Series) -> pd.Series:
    def clean_one(x: str) -> str:
        if "new" in x:
            return "like new"
        elif "excellent" in x:
            return "excellent"
        elif "good" in x or "great" in x or "nice" in x:
            return "good"
        elif "fair" in x:
            return "fair"
        else:
            return "other"
    return s.apply(clean_one)


def _clean_color(s: pd.Series) -> pd.Series:
    def clean_one(x: str) -> str:
        if "black" in x:
            return "black"
        elif "white" in x:
            return "white"
        elif "gray" in x or "grey" in x or "silver" in x:
            return "gray"
        elif "blue" in x:
            return "blue"
        elif "red" in x:
            return "red"
        elif "green" in x:
            return "green"
        else:
            return "other"
    return s.apply(clean_one)


def run_once(
    dry_run: bool = False,
    n_estimators: int = 200,
    max_depth: int = 20,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {
        "scraped_at", "price", "make", "model", "year", "mileage",
        "transmission", "fuel_type", "drive_type", "title_status",
        "condition", "color", "city", "state"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numerics ---
    orig_rows = len(df)
    df["price_num"] = _clean_numeric(df["price"])
    df["year_num"] = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    # --- Standardize categorical values like in notebook ---
    df = _standardize_text_columns(df)
    df["make"] = _clean_make(df["make"])
    df["drive_type"] = _clean_drive_type(df["drive_type"])
    df["condition"] = _clean_condition(df["condition"])
    df["color"] = _clean_color(df["color"])

    # --- Feature engineering ---
    current_year = pd.Timestamp.now(tz=TIMEZONE).year
    df["car_age"] = current_year - df["year_num"]
    df["log_mileage"] = np.log1p(df["mileage_num"])

    # --- Basic row filtering like notebook ---
    df = df[
        df["price_num"].notna() &
        df["year_num"].notna() &
        df["mileage_num"].notna() &
        (df["price_num"] >= 500) &
        (df["price_num"] <= 100000) &
        (df["mileage_num"] >= 0) &
        (df["mileage_num"] <= 500000)
    ].copy()

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | valid after cleaning=%d", orig_rows, valid_price_rows)

    counts = df["date_local"].value_counts().sort_index()
    logging.info("Recent date counts (local): %s", json.dumps({str(k): int(v) for k, v in counts.tail(8).items()}))

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {
            "status": "noop",
            "reason": "need at least two distinct local dates after cleaning",
            "dates": [str(d) for d in unique_dates]
        }

    latest_local_day = unique_dates[-1]
    train_df = df[df["date_local"] < latest_local_day].copy()
    holdout_df = df[df["date_local"] == latest_local_day].copy()

    logging.info("Train rows: %d", len(train_df))
    logging.info("Holdout rows for latest local day (%s): %d", latest_local_day, len(holdout_df))

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    # --- Final model features ---
    cat_cols = [
        "make", "model", "transmission", "fuel_type", "drive_type",
        "title_status", "condition", "color", "city", "state"
    ]
    num_cols = ["car_age", "log_mileage"]
    feats = cat_cols + num_cols
    target = "price_num"

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])

    X_train = train_df[feats]
    y_train = train_df[target]
    pipe.fit(X_train, y_train)

    # ---- Predict/evaluate on holdout ----
    mae_today = None
    preds_df = pd.DataFrame()

    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat = pipe.predict(X_h)

        cols = ["post_id", "scraped_at", "make", "model", "year", "mileage", "price"]
        preds_df = holdout_df[cols].copy()
        preds_df["actual_price"] = holdout_df["price_num"]
        preds_df["pred_price"] = np.round(y_hat, 2)

        y_true = holdout_df["price_num"]
        mask = y_true.notna()
        if mask.any():
            mae_today = float(mean_absolute_error(y_true[mask], y_hat[mask]))

    # --- Output path: hourly folder structure for sync workflow ---
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    out_key = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}/preds.csv"

    if not dry_run and len(preds_df) > 0:
        _write_csv_to_gcs(client, GCS_BUCKET, out_key, preds_df)
        logging.info("Wrote predictions to gs://%s/%s (%d rows)", GCS_BUCKET, out_key, len(preds_df))
    else:
        logging.info("Dry run or no holdout rows; skip write. Would write to gs://%s/%s", GCS_BUCKET, out_key)

    return {
        "status": "ok",
        "latest_local_day": str(latest_local_day),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_price_rows": valid_price_rows,
        "mae_today": mae_today,
        "output_key": out_key,
        "dry_run": dry_run,
        "timezone": TIMEZONE,
        "model": "random_forest",
        "features": feats,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
    }


def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False)),
            n_estimators=int(body.get("n_estimators", 200)),
            max_depth=int(body.get("max_depth", 20)),
            min_samples_split=int(body.get("min_samples_split", 2)),
            min_samples_leaf=int(body.get("min_samples_leaf", 1)),
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
