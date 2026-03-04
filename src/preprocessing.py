from pathlib import Path
import pandas as pd
import re

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "Alakita-Grocery.csv"
OUT_DIR = BASE_DIR / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(x):
    """
    Normalize text values to ensure consistent formatting.
    """
    if pd.isna(x):
        return ""

    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)

    return x


# ============================================================
# MULTI-SELECT SPLITTING FUNCTION
# ============================================================

def split_multiselect(value: str):
    """
    Split Google Forms multi-select responses correctly.
    """

    value = clean_text(value)

    if not value:
        return []

    # Handle labels like "Dairy: Milk, yogurt"
    if ":" in value:
        parts = re.split(r",\s(?=[A-Z][A-Za-z &/()-]+:)", value)
        return [p.strip() for p in parts if p.strip()]

    return [p.strip() for p in value.split(",") if p.strip()]


# ============================================================
# ONE HOT ENCODING FUNCTION
# ============================================================

def one_hot_from_multiselect(series: pd.Series, prefix: str = "") -> pd.DataFrame:
    """
    Convert multi-select column to one-hot encoded dataset.
    """

    all_lists = series.apply(split_multiselect)

    unique_items = sorted({item for sublist in all_lists for item in sublist})

    out = pd.DataFrame(0, index=series.index, columns=unique_items, dtype=int)

    for idx, items in all_lists.items():
        for it in items:
            out.loc[idx, it] = 1

    if prefix:
        out = out.rename(columns={c: f"{prefix}{c}" for c in out.columns})

    return out


# ============================================================
# SIMPLIFY BASKET COLUMN NAMES
# ============================================================

def simplify_basket_column_names(df_basket: pd.DataFrame) -> pd.DataFrame:
    """
    Shorten long product category column names.
    """

    new_cols = []

    for c in df_basket.columns:
        c2 = c.replace("BUY_", "")
        c2 = c2.split(":")[0].strip()
        c2 = re.sub(r"\s+", "", c2)
        c2 = c2.replace("&", "And")

        new_cols.append(c2)

    df_basket = df_basket.copy()
    df_basket.columns = new_cols

    return df_basket


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(RAW_PATH)

    # --------------------------------------------------------
    # Remove duplicate responses
    # --------------------------------------------------------

    df = df.drop_duplicates()

    # --------------------------------------------------------
    # Remove unnecessary columns
    # --------------------------------------------------------

    drop_keywords = [
        "timestamp",
        "consent",
        "privacy notice",
        "by clicking",
        "additional comments",
        "comments",
        "layout"
    ]

    cols_to_drop = [c for c in df.columns if any(k in c.lower() for k in drop_keywords)]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    print("Dropped columns:", cols_to_drop)

    # --------------------------------------------------------
    # Clean all text columns
    # --------------------------------------------------------

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].apply(clean_text)

    # --------------------------------------------------------
    # Save cleaned survey dataset
    # --------------------------------------------------------

    df.to_csv(OUT_DIR / "cleaned_survey.csv", index=False)

    # --------------------------------------------------------
    # Create basket dataset for ARM
    # --------------------------------------------------------

    products_col = "Which among the following products and goods do you usually buy? (Select all that apply)"

    if products_col not in df.columns:
        raise ValueError("Products multi-select column not found in dataset.")

    basket_products = one_hot_from_multiselect(df[products_col], prefix="BUY_")

    # --------------------------------------------------------
    # Remove extremely rare categories
    # --------------------------------------------------------

    min_count = 3

    keep_cols = basket_products.columns[basket_products.sum(axis=0) >= min_count]
    basket_products = basket_products[keep_cols]

    # --------------------------------------------------------
    # Simplify column names
    # --------------------------------------------------------

    basket_products = simplify_basket_column_names(basket_products)

    # --------------------------------------------------------
    # Save basket dataset
    # --------------------------------------------------------

    basket_products.to_csv(OUT_DIR / "basket_products.csv", index=False)

    print("Saved:")
    print(" - data/processed/cleaned_survey.csv")
    print(" - data/processed/basket_products.csv")


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()