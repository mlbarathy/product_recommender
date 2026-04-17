import ast
import base64
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from recommender import ALGORITHM_INFO, ProductRecommender, product_list_from_sqlite

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
PRODUCTS_SQLITE = DATA_DIR / "products_details.sqlite"
DOWNLOAD_THREAD_NAME = "dataset-download"

MAX_CELL_CHARS = 400
DEFAULT_PER_PAGE = 25
MAX_PER_PAGE = 100
MAX_IMAGE_EMBED_BYTES = 400_000

app = FastAPI(
    title="ProductMatch API",
    description="Product catalog + recommendations using five text embeddings (BoW, TF-IDF, Word2Vec, GloVe, FastText) and cosine similarity — same logic as CineMatch, product text instead of movie text.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender: ProductRecommender | None = None
_recommender_init_lock = threading.Lock()

_download_lock = threading.Lock()
_download_state: dict[str, Any] = {
    "status": "idle",
    "progress": 0,
    "message": "",
    "error": None,
}


def _download_status_snapshot() -> dict[str, Any]:
    with _download_lock:
        return {
            "status": str(_download_state["status"]),
            "progress": int(_download_state["progress"]),
            "message": str(_download_state.get("message") or ""),
            "error": _download_state.get("error"),
        }


def _download_state_update(**kwargs: Any) -> None:
    with _download_lock:
        _download_state.update(kwargs)


def _dataset_download_worker_alive() -> bool:
    return any(t.name == DOWNLOAD_THREAD_NAME and t.is_alive() for t in threading.enumerate())


def _normalize_download_state_if_dataset_missing() -> None:
    """
    If the primary SQLite file is missing:
    - Reset completed/failed so a new download can start.
    - Reset orphaned 'downloading' (no worker thread), e.g. after manual file delete,
      server reload, or a crashed worker — otherwise POST /download-dataset returns 409 forever.
    Leave 'downloading' unchanged only while the dataset-download thread is still alive.
    """
    if PRODUCTS_SQLITE.is_file():
        return
    with _download_lock:
        st = _download_state["status"]
        if st in ("completed", "failed"):
            _download_state.update(
                status="idle",
                progress=0,
                message="",
                error=None,
            )
            return
        if st == "downloading" and not _dataset_download_worker_alive():
            _download_state.update(
                status="idle",
                progress=0,
                message="",
                error=None,
            )


def _reload_recommender() -> None:
    global recommender
    with _recommender_init_lock:
        if PRODUCTS_SQLITE.is_file():
            try:
                recommender = ProductRecommender(str(PRODUCTS_SQLITE))
            except Exception as exc:  # pragma: no cover
                print("Recommender reload failed:", exc)
                recommender = None
        else:
            recommender = None


def _download_worker() -> None:
    try:
        from file_downloader import download_dataset

        def on_progress(p: int, msg: str | None = None) -> None:
            _download_state_update(progress=p, message=msg or "")

        download_dataset(progress_callback=on_progress)
        _reload_recommender()
        _download_state_update(
            status="completed",
            progress=100,
            message="Dataset ready for use.",
            error=None,
        )
    except Exception as exc:  # pragma: no cover - network / disk
        _download_state_update(
            status="failed",
            message="Download failed.",
            error=str(exc),
        )


@app.on_event("startup")
def _startup():
    global recommender
    if PRODUCTS_SQLITE.is_file():
        try:
            with _recommender_init_lock:
                recommender = ProductRecommender(str(PRODUCTS_SQLITE))
        except Exception as exc:  # pragma: no cover
            print("Startup recommender load failed; will lazy-load on first /recommend or /compare:", exc)
            recommender = None
    else:
        print("No products_details.sqlite — recommendation endpoints disabled until data exists.")


def _mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def format_image_bytes(data: bytes) -> str:
    if not data:
        return ""
    if len(data) > MAX_IMAGE_EMBED_BYTES:
        return f"[Image too large for preview: {len(data) // 1024} KB]"
    return f"data:{_mime(data)};base64,{base64.standard_b64encode(data).decode('ascii')}"


def format_image_cell(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip()
    if not s:
        return ""
    try:
        d = ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError):
        return s[:MAX_CELL_CHARS] + ("…" if len(s) > MAX_CELL_CHARS else "")
    if not isinstance(d, dict):
        return s[:MAX_CELL_CHARS] + ("…" if len(s) > MAX_CELL_CHARS else "")
    blob = d.get("bytes")
    if isinstance(blob, (bytes, bytearray)) and len(blob) > 0:
        return format_image_bytes(bytes(blob))
    return str(d.get("path") or "") or "[No image]"


def format_image_value(v) -> str:
    if isinstance(v, (bytes, bytearray, memoryview)):
        return format_image_bytes(bytes(v))
    return format_image_cell(str(v))


def truncate_record(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out[k] = ""
        elif k.strip().lower() == "image":
            out[k] = format_image_value(v)
        else:
            s = str(v)
            out[k] = s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS] + "…"
    return out


def list_data_files() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in DATA_DIR.iterdir()
        if p.is_file()
        and (
            p.suffix.lower() in (".sqlite", ".db")
            or p.name.lower().endswith(".csv.gz")
            or (p.suffix.lower() == ".csv" and not p.name.lower().endswith(".gz"))
        )
    )


def read_sqlite_page(path: Path, page: int, per_page: int) -> tuple[list[str], list[dict], bool]:
    page = max(1, page)
    per_page = min(max(1, per_page), MAX_PER_PAGE)
    offset = (page - 1) * per_page
    conn = sqlite3.connect(path)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(products)").fetchall()]
        if not cols:
            raise HTTPException(500, "SQLite file has no 'products' table")
        df = pd.read_sql(
            "SELECT * FROM products LIMIT ? OFFSET ?",
            conn,
            params=[per_page + 1, offset],
        )
    finally:
        conn.close()
    has_more = len(df) > per_page
    if has_more:
        df = df.iloc[:per_page]
    rows = [truncate_record(rec) for rec in df.to_dict(orient="records")]
    return cols, rows, has_more


def read_csv_page(path: Path, page: int, per_page: int) -> tuple[list[str], list[dict], bool]:
    page = max(1, page)
    per_page = min(max(1, per_page), MAX_PER_PAGE)
    columns = list(pd.read_csv(path, nrows=0, compression="infer").columns)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page + 1
    rows_out: list[dict] = []
    current = 0
    read_kw = {
        "chunksize": 512,
        "usecols": columns,
        "dtype": str,
        "low_memory": False,
        "compression": "infer",
        "on_bad_lines": "skip",
    }
    for chunk in pd.read_csv(path, **read_kw):
        chunk = chunk.fillna("")
        n = len(chunk)
        chunk_end = current + n
        if chunk_end <= start_idx:
            current = chunk_end
            continue
        if current >= end_idx:
            break
        lo, hi = max(0, start_idx - current), min(n, end_idx - current)
        if lo < hi:
            for rec in chunk.iloc[lo:hi].to_dict(orient="records"):
                rows_out.append(truncate_record(rec))
                if len(rows_out) >= per_page + 1:
                    break
        current = chunk_end
        if len(rows_out) >= per_page + 1:
            break
    has_more = len(rows_out) > per_page
    if has_more:
        rows_out = rows_out[:per_page]
    return columns, rows_out, has_more


def _require_recommender() -> ProductRecommender:
    """
    Return the in-memory recommender, loading it on first use if the SQLite file exists
    but startup or reload did not populate `recommender` (failed load, new file on disk, etc.).
    """
    global recommender
    if recommender is not None:
        return recommender
    if not PRODUCTS_SQLITE.is_file():
        raise HTTPException(
            503,
            detail="Recommendation engine not loaded. Add data/products_details.sqlite and restart.",
        )
    with _recommender_init_lock:
        if recommender is not None:
            return recommender
        try:
            print("Loading ProductRecommender (lazy init)…")
            recommender = ProductRecommender(str(PRODUCTS_SQLITE))
            return recommender
        except Exception as exc:
            print("ProductRecommender lazy load failed:", exc)
            raise HTTPException(
                503,
                detail=f"Could not load recommendation engine: {exc}",
            ) from exc


class RecommendRequest(BaseModel):
    product_id: int
    method: str = "tfidf"
    top_n: int = 10


class CompareRequest(BaseModel):
    product_id: int
    top_n: int = 8


@app.get("/health")
def health():
    n_files = len(list_data_files())
    n_products = len(recommender.df) if recommender is not None else 0
    return {
        "status": "ok",
        "data_files": n_files,
        "products_loaded": n_products,
        "recommender_ready": recommender is not None,
        "dataset_available": PRODUCTS_SQLITE.is_file(),
    }


@app.get("/check-dataset")
def check_dataset():
    """UI: detect whether the primary SQLite dataset exists and list catalog files."""
    _normalize_download_state_if_dataset_missing()
    available = PRODUCTS_SQLITE.is_file()
    files = list_data_files()
    size_bytes: int | None = None
    if available:
        try:
            size_bytes = PRODUCTS_SQLITE.stat().st_size
        except OSError:
            size_bytes = None
    return {
        "dataset_available": available,
        "files": files,
        "dataset_path": str(PRODUCTS_SQLITE),
        "dataset_size_bytes": size_bytes,
    }


@app.get("/download-status")
def download_status():
    """Polling: status idle | downloading | completed | failed; progress 0–100."""
    _normalize_download_state_if_dataset_missing()
    return _download_status_snapshot()


@app.post("/download-dataset")
def start_dataset_download():
    """
    Start background download into ./data/ using file_downloader.download_dataset.
    Poll GET /download-status until status is completed or failed.
    """
    if PRODUCTS_SQLITE.is_file():
        raise HTTPException(
            400,
            detail="Dataset already present. Delete data/products_details.sqlite first if you need a full re-download.",
        )
    _normalize_download_state_if_dataset_missing()
    with _download_lock:
        if _download_state["status"] == "downloading":
            raise HTTPException(409, detail="Download already in progress.")
        _download_state.update(
            {
                "status": "downloading",
                "progress": 0,
                "message": "Starting…",
                "error": None,
            },
        )
    threading.Thread(target=_download_worker, name=DOWNLOAD_THREAD_NAME, daemon=True).start()
    return {"started": True}


@app.get("/files")
def api_files():
    return {"data_dir": str(DATA_DIR), "files": list_data_files()}


@app.get("/products")
def list_products():
    """
    Product dropdown for the Recommend tab. Uses the full recommender when ready;
    otherwise reads SQLite only so the list appears while models initialize.
    """
    if recommender is not None:
        return {"products": recommender.get_products(), "models_ready": True}
    if PRODUCTS_SQLITE.is_file():
        try:
            return {
                "products": product_list_from_sqlite(str(PRODUCTS_SQLITE)),
                "models_ready": False,
            }
        except Exception as exc:
            raise HTTPException(
                503,
                detail=f"Could not read product catalog from SQLite: {exc}",
            ) from exc
    raise HTTPException(
        503,
        detail="Recommendation engine not loaded. Add data/products_details.sqlite and restart.",
    )


@app.get("/algorithms")
def list_algorithms():
    return ALGORITHM_INFO


@app.post("/recommend")
def recommend(req: RecommendRequest):
    valid = ["bow", "tfidf", "word2vec", "glove", "fasttext"]
    if req.method not in valid:
        raise HTTPException(400, detail=f"method must be one of {valid}")
    r = _require_recommender()
    n = len(r.df)
    if req.product_id < 0 or req.product_id >= n:
        raise HTTPException(404, detail=f"product_id {req.product_id} out of range (0–{n - 1})")
    results = r.recommend(req.product_id, method=req.method, top_n=req.top_n)
    prods = r.get_products()
    query_name = next((p["name"] for p in prods if p["id"] == req.product_id), str(req.product_id))
    return {
        "query": query_name,
        "product_id": req.product_id,
        "method": req.method,
        "algorithm": ALGORITHM_INFO[req.method]["name"],
        "recommendations": results,
    }


@app.post("/compare")
def compare(req: CompareRequest):
    r = _require_recommender()
    n = len(r.df)
    if req.product_id < 0 or req.product_id >= n:
        raise HTTPException(404, detail=f"product_id {req.product_id} out of range (0–{n - 1})")
    all_results = r.compare_all(req.product_id, top_n=req.top_n)
    prods = r.get_products()
    query_name = next((p["name"] for p in prods if p["id"] == req.product_id), str(req.product_id))
    return {"query": query_name, "product_id": req.product_id, "results": all_results, "algorithms": ALGORITHM_INFO}


@app.get("/table/{filename}")
def api_table(
    filename: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(DEFAULT_PER_PAGE, ge=1, le=MAX_PER_PAGE),
):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    fn = filename.lower()
    path = DATA_DIR / filename
    if not path.is_file():
        raise HTTPException(404, f"File not found: {filename}")
    try:
        if fn.endswith(".sqlite") or fn.endswith(".db"):
            columns, rows, has_more = read_sqlite_page(path, page, per_page)
        elif fn.endswith(".csv.gz") or (fn.endswith(".csv") and not fn.endswith(".gz")):
            columns, rows, has_more = read_csv_page(path, page, per_page)
        else:
            raise HTTPException(400, "Use .sqlite, .db, .csv, or .csv.gz")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to read data: {e}") from e
    return {
        "filename": filename,
        "page": page,
        "per_page": per_page,
        "columns": columns,
        "rows": rows,
        "has_more": has_more,
        "max_cell_chars": MAX_CELL_CHARS,
    }


@app.get("/")
def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(500, "Missing static/index.html")
    return FileResponse(index)
