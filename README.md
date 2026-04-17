# Product recommender (ProductMatch)

Single **FastAPI** app: one process serves **`static/index.html`** and JSON. Same **five embedding methods + cosine similarity** as [CineMatch](../movie_recommender/README.md) (BoW, TF-IDF, Word2Vec, GloVe, FastText), but **product text** (description + category + brand/model fields) instead of movie text. **No separate frontend framework** — one UI file + Python only.

---

## Layout

| File | Role |
|------|------|
| `app.py` | FastAPI: catalog routes + recommendation routes; loads `ProductRecommender` at startup from `data/products_details.sqlite`. |
| `recommender.py` | `_prepare_text()`, `_product_to_dict()`, builds five similarity matrices, `recommend()` / `compare_all()`. |
| `static/index.html` | **Catalog** tab (paged table) + **Recommend** tab (product picker, single-method or compare-all-five). |
| `data/products_details.sqlite` | Table `products` — build with `file_downloader.py`. |
| `file_downloader.py` | HF dataset → SQLite with compressed image blobs. |

---

## Text fields (vs movies)

- **`_prepare_text()`** — Concatenates **description** (`description` / `About Product`), **category** (`Category`), and **brand proxy** (`Brand` / `Manufacturer` / `Model Number` / `Is Amazon Seller` — first match wins).
- **`_product_to_dict()`** — Returns `name`, `description`, `category`, `price`, `image` (data URL when small enough), `uniq_id`, `similarity`.

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | `status`, `data_files`, `products_loaded`, `recommender_ready` |
| `GET` | `/files` | Files under `data/` for the catalog table |
| `GET` | `/table/{filename}` | Paginated rows (catalog) |
| `GET` | `/products` | `{ "products": [ { "id", "name", "uniq_id" }, ... ] }` for the Recommend tab |
| `GET` | `/algorithms` | Metadata dict per method (same keys as CineMatch) |
| `POST` | `/recommend` | Body: `{ "product_id": int, "method": "tfidf", "top_n": 10 }` |
| `POST` | `/compare` | Body: `{ "product_id": int, "top_n": 8 }` — all five methods |

Swagger: `http://localhost:8000/docs`

---

## Local setup

```bash
cd product_recommender
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python file_downloader.py   # if you need a fresh products_details.sqlite
uvicorn app:app --reload --port 8080
```

Open **http://localhost:8080** — use **Catalog** or **Recommend**.

`curl http://localhost:8080/health`

---

## Fast dev (skip large Gensim downloads)

Same idea as CineMatch’s `CINEMATCH_LIGHT`:

```bash
PRODUCTMATCH_LIGHT=1 uvicorn app:app --reload --port 8080
```

(`CINEMATCH_LIGHT=1` is also accepted.) Word2Vec / GloVe / FastText slots use the **TF-IDF** similarity matrix so startup stays quick.

---

## Attribution

Dataset: [philschmid/amazon-product-descriptions-vlm](https://huggingface.co/datasets/philschmid/amazon-product-descriptions-vlm).
