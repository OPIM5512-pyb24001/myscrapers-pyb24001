import os
import io
import json
import logging
import traceback
import numpy as np
import pandas as pd
from google.cloud import storage

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------- CONFIG ----------------
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "structured/preds")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")

logging.basicConfig(level=logging.INFO)


# ---------------- GCS HELPERS ----------------
def read_csv_gcs(client, bucket, key):
    blob = client.bucket(bucket).blob(key)
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def write_csv_gcs(client, bucket, key, df):
    blob = client.bucket(bucket).blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")


# ---------------- MAIN PIPELINE ----------------
def run_once():
    client = storage.Client(project=PROJECT_ID)

    df = read_csv_gcs(client, GCS_BUCKET, DATA_KEY)

    def clean_num(s):
        return pd.to_numeric(
            s.astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce"
        )

    df["price_num"] = clean_num(df["price"])
    df["year_num"] = clean_num(df["year"])
    df["mileage_num"] = clean_num(df["mileage"])

    current_year = pd.Timestamp.now().year
    df["car_age"] = current_year - df["year_num"]
    df["log_mileage"] = np.log1p(df["mileage_num"])

    df = df[
        df["price_num"].notna() &
        df["year_num"].notna() &
        df["mileage_num"].notna()
    ].copy()

    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["date"] = df["scraped_at"].dt.date

    latest_date = sorted(df["date"].dropna().unique())[-1]

    train_df = df[df["date"] < latest_date]
    test_df = df[df["date"] == latest_date]

    logging.info(f"Train rows: {len(train_df)}")
    logging.info(f"Test rows: {len(test_df)}")

    cat_cols = [
        "make",
        "model",
        "transmission",
        "fuel_type",
        "drive_type",
        "title_status",
        "condition",
        "color",
        "city",
        "state",
    ]

    num_cols = ["car_age", "log_mileage"]
    features = cat_cols + num_cols

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])

    X_train = train_df[features]
    y_train = train_df["price_num"]

    pipe.fit(X_train, y_train)

    X_test = test_df[features]
    preds = pipe.predict(X_test)

    test_df["pred_price"] = preds

    if len(test_df) > 0:
        mae = mean_absolute_error(test_df["price_num"], preds)
        logging.info(f"MAE: {mae}")

    # --- Output path: UNIQUE TIMESTAMP folder structure ---
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    run_id = now_utc.strftime("%Y%m%dT%H%M%SZ")
    output_key = f"{OUTPUT_PREFIX}/{run_id}/preds.csv"

    output_df = test_df[[
        "post_id",
        "scraped_at",
        "make",
        "model",
        "year",
        "mileage",
        "price",
        "price_num",
        "pred_price"
    ]]

    write_csv_gcs(client, GCS_BUCKET, output_key, output_df)

    logging.info(f"Wrote: gs://{GCS_BUCKET}/{output_key}")

    return {"status": "ok", "rows": len(output_df), "output_key": output_key}


# ---------------- ENTRYPOINT ----------------
def train_dt_http(request):
    try:
        result = run_once()
        return (json.dumps(result), 200)
    except Exception as e:
        logging.error(traceback.format_exc())
        return (json.dumps({"error": str(e)}), 500)
