"""
Product text embeddings — same five methods as CineMatch (BoW, TF-IDF, Word2Vec, GloVe, FastText)
and cosine similarity; input text is built from product fields instead of movie metadata.
"""

import base64
import os
import re
import sqlite3

import certifi
import gensim.downloader as api
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

MAX_CARD_IMAGE_BYTES = 120_000

ALGORITHM_INFO = {
    "bow": {
        "name": "Bag of Words",
        "year": "1954",
        "color": "#ef4444",
        "tagline": "Counts word occurrences — order doesn't matter",
        "pro": "Simple, fast, interpretable",
        "con": "Ignores word order and semantics",
        "formula": "V[word] = count of word in document",
    },
    "tfidf": {
        "name": "TF-IDF",
        "year": "1972",
        "color": "#f59e0b",
        "tagline": "Rare words matter more than common ones",
        "pro": "Weights distinctive words higher",
        "con": "Still bag-of-words; no deep semantics",
        "formula": "TF-IDF = TF(t,d) × log(N / df(t))",
    },
    "word2vec": {
        "name": "Word2Vec (Google News)",
        "year": "2013",
        "color": "#10b981",
        "tagline": "Pre-trained 300d semantic vectors",
        "pro": "Rich real-world word meanings",
        "con": "OOV words get zero vector",
        "formula": "P(center | context) → vectors",
    },
    "glove": {
        "name": "GloVe",
        "year": "2014",
        "color": "#6366f1",
        "tagline": "Global co-occurrence + local context",
        "pro": "Stable pre-trained vectors",
        "con": "One vector per word — polysemy",
        "formula": "u·v ≈ log P(word_i | word_j)",
    },
    "fasttext": {
        "name": "FastText (Wiki News)",
        "year": "2016",
        "color": "#ec4899",
        "tagline": "Subword n-grams — handles OOV / typos",
        "pro": "Works on rare product tokens",
        "con": "Subword noise on very short text",
        "formula": "word_vector = avg of n-gram vectors",
    },
}


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    norm = {re.sub(r"[\s_]+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        k = re.sub(r"[\s_]+", "", cand.lower())
        if k in norm:
            return norm[k]
    for cand in candidates:
        cl = cand.lower().strip()
        for c in df.columns:
            if c.lower().strip() == cl:
                return c
    return None


def _identity_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Product label + optional stable id column for dropdowns / display."""
    name_col = _find_col(
        df,
        "Product Name",
        "product_name",
        "name",
        "Product Title",
        "product title",
        "title",
        "product_title",
    )
    id_col = _find_col(
        df,
        "Uniq Id",
        "Unique_Id",
        "uniq id",
        "ASIN",
        "asin",
        "SKU",
        "sku",
    )
    return name_col, id_col


def _rows_to_product_options(df: pd.DataFrame, name_col: str, id_col: str | None) -> list[dict]:
    out: list[dict] = []
    for i in range(len(df)):
        name = str(df.iloc[i][name_col])
        if name.lower() in ("nan", "none", ""):
            name = f"Product #{i}"
        uid = ""
        if id_col:
            v = df.iloc[i][id_col]
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                uid = str(v)
        out.append({"id": i, "name": name, "uniq_id": uid or str(i)})
    return out


def product_list_from_sqlite(sqlite_path: str) -> list[dict]:
    """
    Lightweight product list for the UI while ProductRecommender is still
    initializing (model load can take minutes on first run).
    Row order matches ProductRecommender (ORDER BY rowid).
    """
    conn = sqlite3.connect(sqlite_path)
    df = pd.read_sql("SELECT * FROM products ORDER BY rowid", conn)
    conn.close()
    name_col, id_col = _identity_columns(df)
    if not name_col:
        raise ValueError(
            "SQLite table 'products' must include a product name column (e.g. Product Name).",
        )
    return _rows_to_product_options(df, name_col, id_col)


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


def _image_to_data_url(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        data = bytes(raw)
    else:
        return ""
    if not data:
        return ""
    if len(data) > MAX_CARD_IMAGE_BYTES:
        return ""
    return f"data:{_mime(data)};base64,{base64.standard_b64encode(data).decode('ascii')}"


class ProductRecommender:
    def __init__(self, sqlite_path: str):
        print("Loading products from SQLite…")
        conn = sqlite3.connect(sqlite_path)
        self.df = pd.read_sql("SELECT * FROM products ORDER BY rowid", conn)
        conn.close()
        print("Rows:", len(self.df))

        self._name_col, self._id_col = _identity_columns(self.df)
        self._desc_col = _find_col(self.df, "description", "About Product", "about_product")
        self._cat_col = _find_col(self.df, "Category", "category")
        self._brand_col = _find_col(
            self.df, "Brand", "Manufacturer", "Model Number", "model_number", "Is Amazon Seller"
        )
        self._price_col = _find_col(self.df, "Selling Price", "selling price", "price")
        self._img_col = _find_col(self.df, "image", "Image")
        self._about_col = _find_col(self.df, "About Product", "about_product")

        if not self._name_col:
            raise ValueError("SQLite table must include a product name column (e.g. Product Name).")

        self._prepare_text()
        self._build_all_models()
        print("Product recommender ready.")

    def _clean(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _prepare_text(self):
        """Combine product description + category + brand (proxy columns if Brand missing)."""
        n = len(self.df)
        empty = pd.Series([""] * n, index=self.df.index)
        desc = empty.copy()
        if self._desc_col:
            desc = desc + " " + self.df[self._desc_col].fillna("").astype(str)
        if self._about_col and self._about_col != self._desc_col:
            desc = desc + " " + self.df[self._about_col].fillna("").astype(str)
        cat = self.df[self._cat_col].fillna("").astype(str) if self._cat_col else empty
        brand = self.df[self._brand_col].fillna("").astype(str) if self._brand_col else empty
        self.df["combined"] = (desc + " " + cat + " " + brand).str.replace(r"\s+", " ", regex=True).str.strip()
        self.df["text_clean"] = self.df["combined"].apply(self._clean)
        self.corpus = self.df["text_clean"].tolist()
        self.tokenized = [doc.split() for doc in self.corpus]

    def _build_all_models(self):
        self._build_bow()
        self._build_tfidf()
        light = os.environ.get("PRODUCTMATCH_LIGHT", os.environ.get("CINEMATCH_LIGHT", "")).strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if light:
            print("  PRODUCTMATCH_LIGHT=1 — word2vec / glove / fasttext use TF-IDF similarity.")
            self.sim_w2v = self.sim_tfidf.copy()
            self.sim_glove = self.sim_tfidf.copy()
            self.sim_fasttext = self.sim_tfidf.copy()
            return
        self._build_word2vec()
        self._build_glove()
        self._build_fasttext()

    def _build_bow(self):
        print("  [1/5] Bag-of-Words…")
        vec = CountVectorizer(max_features=8000, stop_words="english")
        matrix = vec.fit_transform(self.corpus)
        self.sim_bow = cosine_similarity(matrix)

    def _build_tfidf(self):
        print("  [2/5] TF-IDF…")
        vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")
        matrix = vec.fit_transform(self.corpus)
        self.sim_tfidf = cosine_similarity(matrix)

    def _avg_vectors(self, model_wv, size: int) -> np.ndarray:
        vecs = []
        for tokens in self.tokenized:
            token_vecs = [model_wv[w] for w in tokens if w in model_wv]
            vecs.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(size))
        return np.array(vecs)

    def _build_word2vec(self):
        print("  [3/5] Word2Vec (Google News)…")
        try:
            wv = api.load("word2vec-google-news-300")
            matrix = self._avg_vectors(wv, 300)
            self.sim_w2v = cosine_similarity(matrix)
        except Exception as e:
            print(f"  Word2Vec fallback: {e}")
            self.sim_w2v = self.sim_tfidf.copy()

    def _build_glove(self):
        print("  [4/5] GloVe…")
        try:
            glove = api.load("glove-wiki-gigaword-50")
            matrix = self._avg_vectors(glove, 50)
            self.sim_glove = cosine_similarity(matrix)
        except Exception as e:
            print(f"  GloVe fallback: {e}")
            self.sim_glove = self.sim_tfidf.copy()

    def _build_fasttext(self):
        print("  [5/5] FastText…")
        try:
            wv = api.load("fasttext-wiki-news-subwords-300")
            matrix = self._avg_vectors(wv, 300)
            self.sim_fasttext = cosine_similarity(matrix)
        except Exception as e:
            print(f"  FastText fallback: {e}")
            self.sim_fasttext = self.sim_tfidf.copy()

    _SIM_MAP = {
        "bow": lambda self: self.sim_bow,
        "tfidf": lambda self: self.sim_tfidf,
        "word2vec": lambda self: self.sim_w2v,
        "glove": lambda self: self.sim_glove,
        "fasttext": lambda self: self.sim_fasttext,
    }

    def _product_to_dict(self, row: pd.Series, similarity: float) -> dict:
        name = str(row[self._name_col])
        desc = ""
        if self._desc_col:
            desc = str(row.get(self._desc_col, "") or "")
        if not desc.strip() and self._about_col:
            desc = str(row.get(self._about_col, "") or "")
        cat = str(row.get(self._cat_col, "") or "") if self._cat_col else ""
        price = str(row.get(self._price_col, "") or "") if self._price_col else ""
        img = ""
        if self._img_col:
            img = _image_to_data_url(row.get(self._img_col))
        uid = str(row.get(self._id_col, "") or "") if self._id_col else ""
        return {
            "name": name,
            "description": desc[:800] + ("…" if len(desc) > 800 else ""),
            "category": cat,
            "price": price,
            "image": img,
            "uniq_id": uid,
            "similarity": round(float(similarity), 4),
        }

    def recommend(self, product_id: int, method: str = "tfidf", top_n: int = 10) -> list:
        if method not in self._SIM_MAP:
            raise ValueError(f"Unknown method '{method}'")
        n = len(self.df)
        if product_id < 0 or product_id >= n:
            return []
        sim_matrix = self._SIM_MAP[method](self)
        scores = sim_matrix[product_id]
        ranked = np.argsort(-scores)
        top_indices = [i for i in ranked if i != product_id][:top_n]
        return [self._product_to_dict(self.df.iloc[i], scores[i]) for i in top_indices]

    def compare_all(self, product_id: int, top_n: int = 8) -> dict:
        return {method: self.recommend(product_id, method=method, top_n=top_n) for method in self._SIM_MAP}

    def get_products(self) -> list[dict]:
        return _rows_to_product_options(self.df, self._name_col, self._id_col)

    def algorithm_info(self) -> dict:
        return ALGORITHM_INFO
