"""
data_ingestion.py — Data Ingestion & Cleaning Pipeline
=======================================================
Phase 1 of the RAG-powered Hybrid Book Recommendation Engine.

Loads the Kaggle "Top 500 Amazon Bestselling Books" CSV, applies
standardized cleaning transformations, and exports a clean artifact
ready for downstream vectorization and reranking.

"""

import re
import pandas as pd
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
RAW_CSV = Path(__file__).parent / "Amazon_BestSelling_Books_500.csv"
CLEAN_CSV = Path(__file__).parent / "cleaned_bestsellers.csv"

# Columns that must be strictly numeric after cleaning
NUMERIC_COLS = ["price_usd", "rating", "reviews", "weeks_on_list", "amazon_bsr"]

# Fill strategies for missing values
FILL_ZERO_COLS = ["reviews", "weeks_on_list"]
FILL_UNKNOWN_COLS = ["sub_genre", "publisher"]

# String columns to strip whitespace from
TEXT_COLS = ["title", "author", "category", "sub_genre", "format", "publisher", "isbn"]


# ── Helpers ──────────────────────────────────────────────────────────────────
def to_snake_case(col_name: str) -> str:
    """Convert a column name to clean snake_case.

    Examples:
        'Price (USD)'    → 'price_usd'
        'Weeks on List'  → 'weeks_on_list'
        'Amazon BSR'     → 'amazon_bsr'
        'Sub-Genre'      → 'sub_genre'
        'Year Published' → 'year_published'
    """
    col = col_name.strip()
    # Replace hyphens and slashes with spaces (for tokenization)
    col = col.replace("-", " ").replace("/", " ")
    # Remove parentheses and other special chars but keep alphanumeric & spaces
    col = re.sub(r"[^a-zA-Z0-9\s]", "", col)
    # Collapse multiple spaces → single, then replace spaces with underscores
    col = re.sub(r"\s+", "_", col.strip())
    return col.lower()


def strip_non_numeric(value) -> str:
    """Remove $, commas, and other non-numeric chars (except . and -) from a value."""
    if pd.isna(value):
        return value
    return re.sub(r"[^\d.\-]", "", str(value))


# ── Main Pipeline ────────────────────────────────────────────────────────────
def run_pipeline() -> pd.DataFrame:
    """Execute the full ingestion → clean → export pipeline."""

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"[LOAD]  Loading raw CSV: {RAW_CSV.name}")
    df = pd.read_csv(RAW_CSV)
    print(f"    Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print(f"    Original columns: {df.columns.tolist()}\n")

    # ── 2. Standardize Column Names ──────────────────────────────────────────
    original_cols = df.columns.tolist()
    df.columns = [to_snake_case(c) for c in df.columns]
    renamed = {o: n for o, n in zip(original_cols, df.columns) if o != n}
    print(f"[RENAME]  Column Renaming ({len(renamed)} changed):")
    for old, new in renamed.items():
        print(f"    '{old}' -> '{new}'")
    print()

    # ── 3. Strip Non-Numeric Characters & Type Cast ──────────────────────────
    print("[CAST]  Numeric Coercion:")
    for col in NUMERIC_COLS:
        if col not in df.columns:
            print(f"    [WARN] Column '{col}' not found -- skipping.")
            continue

        before_dtype = df[col].dtype
        # Strip currency symbols, commas, etc.
        df[col] = df[col].apply(strip_non_numeric)
        # Coerce to numeric; invalid entries become NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_dtype = df[col].dtype
        nulls_after = df[col].isna().sum()
        print(f"    {col}: {before_dtype} -> {after_dtype}  (NaNs introduced: {nulls_after})")

    # Downcast integer-safe columns
    for col in ["reviews", "weeks_on_list", "amazon_bsr", "year_published"]:
        if col in df.columns and df[col].notna().all():
            df[col] = df[col].astype(int)
    print()

    # ── 4. Handle Missing Values ─────────────────────────────────────────────
    print("[FILL]  Missing Value Imputation:")
    for col in FILL_ZERO_COLS:
        if col in df.columns:
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(0).astype(int)
            print(f"    {col}: filled {n_filled} nulls with 0")

    for col in FILL_UNKNOWN_COLS:
        if col in df.columns:
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna("Unknown")
            print(f"    {col}: filled {n_filled} nulls with 'Unknown'")

    # Report remaining nulls across the entire frame
    remaining = df.isna().sum()
    remaining = remaining[remaining > 0]
    if remaining.empty:
        print("    [OK] No remaining nulls in any column.")
    else:
        print(f"    [WARN] Remaining nulls:\n{remaining.to_string()}")
    print()

    # ── 5. Text Normalization ────────────────────────────────────────────────
    print("[TEXT]  Text Normalization:")
    for col in TEXT_COLS:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].str.strip()
            print(f"    {col}: stripped leading/trailing whitespace")
    print()

    # ── 6. Drop Low-Value Columns ────────────────────────────────────────────
    # 'amazon_url' is a constant "View" link — no analytical value
    if "amazon_url" in df.columns:
        unique_vals = df["amazon_url"].nunique()
        if unique_vals <= 1:
            df.drop(columns=["amazon_url"], inplace=True)
            print(f"[DROP]  Dropped 'amazon_url' (constant value, {unique_vals} unique).\n")

    # ── 7. Export ─────────────────────────────────────────────────────────────
    df.to_csv(CLEAN_CSV, index=False)
    print(f"[EXPORT]  Cleaned CSV exported -> {CLEAN_CSV.name}")
    print(f"    Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"    Final columns: {df.columns.tolist()}")
    print(f"    Final dtypes:\n{df.dtypes.to_string()}")

    return df


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Book Recommendation Engine -- Data Ingestion Pipeline (Phase 1)")
    print("=" * 70, "\n")

    df = run_pipeline()

    print("\n" + "=" * 70)
    print("  Pipeline Complete [DONE]")
    print("=" * 70)
