from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT = DATA_DIR / "products_details.sqlite"
MAX_SIDE = 420
JPEG_Q = 72
TEXT_MAX = 120
_LONG = frozenset(
    {"about product", "product specification", "technical details", "description"}
)

ProgressCallback = Callable[[int, str | None], None]


def _jpeg_bytes(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if not isinstance(v, dict):
        return None
    b = v.get("bytes")
    if not isinstance(b, (bytes, bytearray)) or not b:
        return None
    try:
        im = Image.open(BytesIO(bytes(b))).convert("RGB")
        w, h = im.size
        m = max(w, h)
        if m > MAX_SIDE:
            s = MAX_SIDE / m
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_Q, optimize=True)
        return buf.getvalue()
    except Exception:
        return bytes(b)


def _trunc(v, n: int):
    if n <= 0:
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v)
    return s if len(s) <= n else s[:n] + "…"


def download_dataset(progress_callback: ProgressCallback | None = None) -> Path:
    """
    Download HF dataset, build SQLite at DATA_DIR / products_details.sqlite.
    Writes to a .partial file first, then replaces OUT atomically.

    progress_callback(progress_0_100, message_or_none) — may be called from worker thread.
    """
    def cb(p: int, msg: str | None = None) -> None:
        if progress_callback:
            progress_callback(max(0, min(100, int(p))), msg)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    partial = OUT.with_suffix(".sqlite.partial")
    if partial.is_file():
        try:
            partial.unlink()
        except OSError:
            pass

    try:
        cb(2, "Preparing data directory…")
        cb(5, "Loading dataset from Hugging Face (may take several minutes)…")
        ds = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
        cb(22, "Converting dataset to dataframe…")
        df = ds.to_pandas()
        nrows = len(df)
        cb(28, f"Loaded {nrows:,} rows — preparing columns…")

        icol = next((c for c in df.columns if str(c).strip().lower() == "image"), None)
        if icol:
            cb(35, "Compressing images…")
            df[icol] = df[icol].map(_jpeg_bytes)
            cb(52, "Image column processed.")

        if TEXT_MAX > 0:
            cb(55, "Truncating long text fields…")
            for c in df.columns:
                if str(c).strip().lower() in _LONG:
                    df[c] = df[c].map(lambda x, n=TEXT_MAX: _trunc(x, n))
            cb(62, "Text fields truncated.")

        cb(68, "Writing SQLite (this may take a minute)…")
        conn = sqlite3.connect(str(partial))
        try:
            df.to_sql("products", conn, index=False, if_exists="replace")
        finally:
            conn.close()

        cb(92, "Replacing database file…")
        os.replace(str(partial), str(OUT))
        cb(100, "Dataset ready.")
        return OUT
    except Exception:
        if partial.is_file():
            try:
                partial.unlink()
            except OSError:
                pass
        raise


if __name__ == "__main__":
    def _cli_cb(p, m=None):
        print(f"[{p:3d}%]", m or "", flush=True)

    download_dataset(progress_callback=_cli_cb)
    print("Wrote:", OUT)
