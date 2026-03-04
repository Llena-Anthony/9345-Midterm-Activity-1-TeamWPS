from pathlib import Path
import pandas as pd
import re

# ============================================================
# PATH CONFIGURATION
# ============================================================
# Get the project root directory regardless of where the script
# is executed from.
BASE_DIR = Path(__file__).resolve().parent.parent

# Location of the raw dataset
RAW_PATH = BASE_DIR / "data" / "raw" / "Alakita-Grocery.csv"

# Output directory for processed datasets
OUT_DIR = BASE_DIR / "data" / "processed"

# Ensure the processed directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================
def clean_text(x):
    """
    Normalize text values to ensure consistent formatting.

    Operations:
    - Convert NaN values to empty string
    - Remove leading and trailing spaces
    - Replace multiple spaces with a single space

    This helps avoid duplicate categories caused by inconsistent spacing.
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
    Split Google Forms multi-select responses into individual items.

    Some options contain commas inside the description
    (example: 'Pantry Staples: Canned goods, grains, pasta').

    To avoid incorrect splitting, we only split when a new option label
    appears (identified by the pattern 'Something:').
    """

    value = clean_text(value)

    if not value:
        return []

    # If the response contains colon labels, split only at label boundaries
    if ":" in value:
        parts = re.split(r",\s(?=[A-Z][A-Za-z &/()-]+:)", value)
        return [p.strip() for p in parts if p.strip()]

    # Otherwise use normal comma splitting
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts


# ============================================================
# ONE-HOT ENCODING FUNCTION
# ============================================================
def one_hot_from_multiselect(series: pd.Series, prefix: str = "") -> pd.DataFrame:
    """
    Convert a multi-select column into a one-hot encoded dataset.

    Each unique option becomes a column.
    Each row represents a respondent/transaction.

    Example:
        Bakery  Dairy  Snacks
          1       0       1

    1 = category selected
    0 = category not selected

    This format is required for Association Rule Mining algorithms.
    """

    # Convert each row into a list of selected options
    all_lists = series.apply(split_multiselect)

    # Extract all unique product categories
    unique_items = sorted({item for sublist in all_lists for item in sublist})

    # Create empty binary matrix
    out = pd.DataFrame(0, index=series.index, columns=unique_items, dtype=int)

    # Mark selected categories with 1
    for idx, items in all_lists.items():
        for it in items:
            out.loc[idx, it] = 1

    # Optionally add prefix to column names
    if prefix:
        out = out.rename(columns={c: f"{prefix}{c}" for c in out.columns})

    return out


# ============================================================
# COLUMN NAME SIMPLIFICATION
# ============================================================
def simplify_basket_column_names(df_basket: pd.DataFrame) -> pd.DataFrame:
    """
    Shorten product category column names.

    The original survey labels are long and contain descriptions.
    For example:
        BUY_Beverages: Soft drinks, juices, water...

    This function keeps only the main category label:
        Beverages
    """

    new_cols = []

    for c in df_basket.columns:
        c2 = c.replace("BUY_", "")      # remove prefix
        c2 = c2.split(":")[0].strip()   # keep label before description
        c2 = re.sub(r"\s+", "", c2)     # remove spaces
        c2 = c2.replace("&", "And")

        new_cols.append(c2)

    df_basket = df_basket.copy()
    df_basket.columns = new_cols

    return df_basket


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================
def main():
    """
    Execute the full preprocessing pipeline.

    Output files:
        cleaned_survey.csv   -> used for descriptive analysis
        basket_products.csv  -> used for association rule mining
    """

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    df = pd.read_csv(RAW_PATH)

    # --------------------------------------------------------
    # Remove duplicate survey responses
    # --------------------------------------------------------
    df = df.drop_duplicates()

    # --------------------------------------------------------
    # Convert Timestamp column to datetime format
    # --------------------------------------------------------
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # --------------------------------------------------------
    # Remove consent confirmation column (administrative field)
    # --------------------------------------------------------
    consent_col = 'By clicking "I agree" below, you confirm that you have read and understood the above terms and voluntarily consent to participate in this survey.'

    if consent_col in df.columns:
        df = df.drop(columns=[consent_col])

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
    # Create ARM basket dataset from product selections
    # --------------------------------------------------------
    products_col = "Which among the following products and goods do you usually buy? (Select all that apply)"

    if products_col not in df.columns:
        raise ValueError("Products multi-select column not found in dataset.")

    basket_products = one_hot_from_multiselect(df[products_col], prefix="BUY_")

    # --------------------------------------------------------
    # Remove extremely rare product categories
    # (helps produce stronger association rules)
    # --------------------------------------------------------
    min_count = 3

    keep_cols = basket_products.columns[basket_products.sum(axis=0) >= min_count]
    basket_products = basket_products[keep_cols]

    # --------------------------------------------------------
    # Simplify column names for readability
    # --------------------------------------------------------
    basket_products = simplify_basket_column_names(basket_products)

    # --------------------------------------------------------
    # Save basket dataset
    # --------------------------------------------------------
    basket_products.to_csv(OUT_DIR / "basket_products.csv", index=False)

    print("Saved:")
    print(" - data/processed/cleaned_survey.csv")
    print(" - data/processed/basket_products.csv")


# Run script
if __name__ == "__main__":
    main()