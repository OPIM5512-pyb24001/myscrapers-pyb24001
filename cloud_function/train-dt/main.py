import os
import io
import json
import logging
import traceback
import numpy as np
import pandas as pd
from google.cloud import storage
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

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


def upload_file_gcs(client, bucket, local_path, gcs_key, content_type=None):
    blob = client.bucket(bucket).blob(gcs_key)
    if content_type:
        blob.upload_from_filename(local_path, content_type=content_type)
    else:
        blob.upload_from_filename(local_path)


# ---------------- MAIN PIPELINE ----------------
def run_once():
    client = storage.Client(project=PROJECT_ID)

    # Read cumulative LLM dataset from GCS
    df = read_csv_gcs(client, GCS_BUCKET, DATA_KEY)

    # ---------------- Clean numeric columns ----------------
    def clean_num(s):
        return pd.to_numeric(
            s.astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce"
        )

    df["price_num"] = clean_num(df["price"])
    df["year_num"] = clean_num(df["year"])
    df["mileage_num"] = clean_num(df["mileage"])

    # ---------------- Feature engineering ----------------
    current_year = pd.Timestamp.now().year
    df["car_age"] = current_year - df["year_num"]
    df["log_mileage"] = np.log1p(df["mileage_num"])

    # ---------------- Filter usable rows ----------------
    df = df[
        df["price_num"].notna() &
        df["year_num"].notna() &
        df["mileage_num"].notna()
    ].copy()

    # ---------------- Time split ----------------
    # Train on past data and predict newest day's listings
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["date"] = df["scraped_at"].dt.date

    latest_date = sorted(df["date"].dropna().unique())[-1]

    train_df = df[df["date"] < latest_date].copy()
    test_df = df[df["date"] == latest_date].copy()

    logging.info(f"Train rows: {len(train_df)}")
    logging.info(f"Test rows: {len(test_df)}")

    # ---------------- Features ----------------
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

    # ---------------- Preprocessing ----------------
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    # ---------------- Final tuned model ----------------
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

    # ---------------- Train ----------------
    X_train = train_df[features]
    y_train = train_df["price_num"]
    pipe.fit(X_train, y_train)

    # ---------------- Predict ----------------
    X_test = test_df[features]
    preds = pipe.predict(X_test)
    test_df["pred_price"] = preds

    # ---------------- Evaluate ----------------
    mae = None
    if len(test_df) > 0:
        mae = mean_absolute_error(test_df["price_num"], preds)
        logging.info(f"MAE: {mae}")

    # ---------------- Unique run_id ----------------
    # This ensures every run gets its own folder
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    run_id = now_utc.strftime("%Y%m%dT%H%M%SZ")

    # ---------------- Save predictions ----------------
    preds_key = f"{OUTPUT_PREFIX}/{run_id}/preds.csv"

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
    ]].copy()

    write_csv_gcs(client, GCS_BUCKET, preds_key, output_df)
    logging.info(f"Wrote predictions: gs://{GCS_BUCKET}/{preds_key}")

    # ---------------- Permutation importance ----------------
    perm = permutation_importance(
        pipe,
        X_test,
        test_df["price_num"],
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    feature_names = pipe.named_steps["pre"].get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)

    importance_key = f"{OUTPUT_PREFIX}/{run_id}/importance.csv"
    write_csv_gcs(client, GCS_BUCKET, importance_key, importance_df)
    logging.info(f"Wrote importance: gs://{GCS_BUCKET}/{importance_key}")

    # ---------------- PDP for top 3 features ----------------
    top_features = importance_df["feature"].head(3).tolist()
    logging.info(f"Top 3 PDP features: {top_features}")

    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        pipe,
        X_test,
        features=top_features,
        ax=ax
    )

    pdp_file = "/tmp/pdp.png"
    plt.tight_layout()
    plt.savefig(pdp_file)
    plt.close()

    pdp_key = f"{OUTPUT_PREFIX}/{run_id}/pdp.png"
    upload_file_gcs(client, GCS_BUCKET, pdp_file, pdp_key, content_type="image/png")
    logging.info(f"Wrote PDP: gs://{GCS_BUCKET}/{pdp_key}")

    return {
        "status": "ok",
        "rows": len(output_df),
        "mae": None if mae is None else float(mae),
        "run_id": run_id,
        "preds_key": preds_key,
        "importance_key": importance_key,
        "pdp_key": pdp_key
    }


# ---------------- ENTRYPOINT ----------------
def train_dt_http(request):
    try:
        result = run_once()
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error(traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
