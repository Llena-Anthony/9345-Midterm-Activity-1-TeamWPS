from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "Alakita-Grocery.csv"
OUT_DIR = BASE_DIR / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(x):
    """Normalize text cells safely."""
    if pd.isna(x):
        return ""
    x = str(x).strip()
    # collapse multiple spaces
    x = re.sub(r"\s+", " ", x)
    return x

def split_multiselect(value: str):
    """
    Split Google Forms multi-select responses.
    Handles options that contain commas inside descriptions by splitting only at
    boundaries that look like a new option label: ', ' followed by 'Something:'
    """
    value = clean_text(value)
    if not value:
        return []

    # If it has colon-based labels (e.g., "Dairy: ...", "Fresh Produce: ..."),
    # split ONLY where a new label begins.
    if ":" in value:
        parts = re.split(r",\s(?=[A-Z][A-Za-z &/()-]+:)", value)
        return [p.strip() for p in parts if p.strip()]

    # Otherwise, normal comma separation works (e.g., payment methods, stores)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts

def one_hot_from_multiselect(series: pd.Series, prefix: str = "") -> pd.DataFrame:
    """Convert a multi-select column to one-hot encoded DataFrame."""
    all_lists = series.apply(split_multiselect)
    unique_items = sorted({item for sublist in all_lists for item in sublist})

    out = pd.DataFrame(0, index=series.index, columns=unique_items, dtype=int)
    for idx, items in all_lists.items():
        for it in items:
            out.loc[idx, it] = 1

    # Optional prefixing (useful if you combine multiple multiselect columns)
    if prefix:
        out = out.rename(columns={c: f"{prefix}{c}" for c in out.columns})

    return out

def main():
    df = pd.read_csv(RAW_PATH)

    # 1) Basic cleanup
    df = df.drop_duplicates()

    # Parse timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Remove consent + optional comments (not useful for ARM; comments can be separate qual analysis)
    consent_col = 'By clicking "I agree" below, you confirm that you have read and understood the above terms and voluntarily consent to participate in this survey.'
    if consent_col in df.columns:
        df = df.drop(columns=[consent_col])

    comments_col = "Additional comments, suggestions, or other experiences related to how a grocery store should be planned and laid out. (Optional)"
    if comments_col in df.columns:
        # keep it if you want, but don't mix it into modeling
        # df_comments = df[[comments_col]].copy()
        pass

    # Clean all text columns
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].apply(clean_text)

    # Save cleaned survey table (for descriptive analytics)
    df.to_csv(OUT_DIR / "cleaned_survey.csv", index=False)

    # 2) Build ARM dataset (basket / one-hot)
    products_col = "Which among the following products and goods do you usually buy? (Select all that apply)"
    if products_col not in df.columns:
        raise ValueError("Products multi-select column not found. Check column name in CSV.")

    basket_products = one_hot_from_multiselect(df[products_col], prefix="BUY_")

    # Optional: drop extremely rare items (e.g., chosen by < 3 people)
    min_count = 3
    keep_cols = basket_products.columns[basket_products.sum(axis=0) >= min_count]
    basket_products = basket_products[keep_cols]

    basket_products.to_csv(OUT_DIR / "basket_products.csv", index=False)

    print("Saved:")
    print(" - data/processed/cleaned_survey.csv")
    print(" - data/processed/basket_products.csv")

if __name__ == "__main__":
    main()