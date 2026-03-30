# main.py
# CUMULATIVE materialize-llm (keeps growing dataset)

import csv
import json
import os
import re
from typing import Dict, Iterable, List, Tuple

from flask import Request, jsonify
from google.cloud import storage

BUCKET_NAME = os.getenv("GCS_BUCKET")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured")
MAX_FILES = int(os.getenv("MAX_FILES", "30"))

storage_client = storage.Client()

RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage",
    "transmission", "fuel_type", "drive_type", "title_status",
    "condition", "color", "city", "state", "zip_code",
    "source_txt", "llm_provider", "llm_model", "llm_ts"
]

# -------------------- GET RECENT FILES --------------------
def _list_recent_llm_files(bucket, prefix, max_files):
    b = storage_client.bucket(bucket)
    files = []

    for blob in b.list_blobs(prefix=f"{prefix}/"):
        name = blob.name

        if "/jsonl_llm/" not in name or not name.endswith(".jsonl"):
            continue

        parts = name.split("/")
        run_part = next((p for p in parts if p.startswith("run_id=")), None)

        if not run_part:
            continue

        run_id = run_part.split("run_id=", 1)[1]

        if RUN_ID_ISO_RE.match(run_id) or RUN_ID_PLAIN_RE.match(run_id):
            files.append((run_id, name))

    files.sort(reverse=True)
    return files[:max_files]


# -------------------- READ JSON --------------------
def _read_jsonl_files(bucket, files):
    b = storage_client.bucket(bucket)

    for run_id, blob_name in files:
        blob = b.blob(blob_name)
        data = blob.download_as_text()

        for line in data.splitlines():
            try:
                rec = json.loads(line)
                rec.setdefault("run_id", run_id)
                yield rec
            except:
                continue


# -------------------- LOAD EXISTING CSV --------------------
def _load_existing_csv(bucket, key):
    b = storage_client.bucket(bucket)
    blob = b.blob(key)

    if not blob.exists():
        return {}

    data = blob.download_as_text().splitlines()
    reader = csv.DictReader(data)

    existing = {}
    for row in reader:
        existing[row["post_id"]] = row

    return existing


# -------------------- WRITE CSV --------------------
def _write_csv(records, dest_key):
    b = storage_client.bucket(BUCKET_NAME)
    blob = b.blob(dest_key)

    with blob.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        count = 0
        for rec in records:
            row = {c: rec.get(c, None) for c in CSV_COLUMNS}
            writer.writerow(row)
            count += 1

    return count


# -------------------- MAIN --------------------
def materialize_http(request: Request):

    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "Missing bucket"}), 500

        output_path = f"{STRUCTURED_PREFIX}/datasets/listings_master_llm.csv"

        # 1. Load existing dataset
        existing_data = _load_existing_csv(BUCKET_NAME, output_path)

        # 2. Get latest 30 files
        recent_files = _list_recent_llm_files(
            BUCKET_NAME,
            STRUCTURED_PREFIX,
            MAX_FILES
        )

        # 3. Add new data
        for rec in _read_jsonl_files(BUCKET_NAME, recent_files):
            pid = rec.get("post_id")
            if not pid:
                continue

            existing_data[pid] = rec  # update or insert

        # 4. Write back cumulative dataset
        rows = _write_csv(existing_data.values(), output_path)

        return jsonify({
            "ok": True,
            "total_rows": rows,
            "files_used": len(recent_files),
            "output": f"gs://{BUCKET_NAME}/{output_path}"
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
