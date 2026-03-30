# main.py
# Build CSV from MOST RECENT 30 LLM jsonl files (fast + scalable)

import csv
import json
import os
import re
from typing import Dict, Iterable, List, Tuple

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME = os.getenv("GCS_BUCKET")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured")
MAX_FILES = int(os.getenv("MAX_FILES", "30"))

storage_client = storage.Client()

RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

# ✅ UPDATED COLUMNS (A08 READY)
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage",
    "transmission", "fuel_type", "drive_type", "title_status",
    "condition", "color", "city", "state",
    "source_txt", "llm_provider", "llm_model", "llm_ts"
]


# -------------------- GET RECENT FILES --------------------
def _list_recent_llm_files(bucket: str, prefix: str, max_files: int) -> List[Tuple[str, str]]:
    """
    Returns latest MAX_FILES jsonl_llm files across all runs.
    """
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

        if not (RUN_ID_ISO_RE.match(run_id) or RUN_ID_PLAIN_RE.match(run_id)):
            continue

        # store (run_id, blob_name)
        files.append((run_id, name))

    # 🔥 SORT BY RUN_ID (latest first)
    files.sort(reverse=True)

    return files[:max_files]


# -------------------- READ JSON --------------------
def _read_jsonl_files(bucket: str, files: List[Tuple[str, str]]):
    b = storage_client.bucket(bucket)

    for run_id, blob_name in files:
        blob = b.blob(blob_name)
        data = blob.download_as_text()

        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                rec.setdefault("run_id", run_id)
                yield rec
            except Exception:
                continue


# -------------------- WRITE CSV --------------------
def _write_csv(records: Iterable[Dict], dest_key: str):
    b = storage_client.bucket(BUCKET_NAME)
    blob = b.blob(dest_key)

    with blob.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        count = 0
        for rec in records:
            row = {c: rec.get(c, None) for c in CSV_COLUMNS}
            writer.writerow(row)
            count += 1

    return count


# -------------------- MAIN FUNCTION --------------------
def materialize_http(request: Request):

    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "Missing GCS_BUCKET"}), 500

        # 🔥 GET LATEST 30 FILES (NOT ALL)
        recent_files = _list_recent_llm_files(
            BUCKET_NAME,
            STRUCTURED_PREFIX,
            MAX_FILES
        )

        if not recent_files:
            return jsonify({"ok": False, "error": "No LLM files found"}), 200

        # 🔥 DEDUP BY post_id
        latest_by_post: Dict[str, Dict] = {}

        for rec in _read_jsonl_files(BUCKET_NAME, recent_files):
            pid = rec.get("post_id")

            if not pid:
                continue

            latest_by_post[pid] = rec

        # 🔥 WRITE FINAL CSV
        output_path = f"{STRUCTURED_PREFIX}/datasets/listings_master_llm.csv"

        rows = _write_csv(latest_by_post.values(), output_path)

        return jsonify({
            "ok": True,
            "files_used": len(recent_files),
            "rows_written": rows,
            "output": f"gs://{BUCKET_NAME}/{output_path}"
        }), 200

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500
