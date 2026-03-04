from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

BASKET_PATH = BASE_DIR / "data" / "processed" / "basket_products.csv"
OUT_DIR = BASE_DIR / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PARAMETERS (EDIT THESE FOR YOUR REPORT)
# ============================================================
MIN_SUPPORT = 0.10        # minsup (e.g., 0.10 means 10% of transactions)
MIN_CONFIDENCE = 0.60     # minconf (e.g., 0.60 means 60% confidence)
MAX_LEN = 3               # max size of itemsets (controls output size)
TOP_K = 20                # number of top rules to print


# ============================================================
# LOAD + VALIDATE BASKET DATA
# ============================================================
def load_basket() -> pd.DataFrame:
    """
    Loads basket dataset where:
      rows = transactions/respondents
      columns = items/categories
      values = 0/1 (not purchased/purchased)

    This is the standard required format for Apriori.
    """
    basket = pd.read_csv(BASKET_PATH)

    # Validate: no missing values
    if basket.isna().any().any():
        raise ValueError("basket_products.csv contains missing values. Fix preprocessing first.")

    # Validate: must be binary (0/1)
    unique_vals = set(pd.unique(basket.values.ravel()))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Basket must contain only 0/1 values. Found: {sorted(unique_vals)}")

    # Convert to boolean for mlxtend Apriori (recommended)
    return basket.astype(bool)


# ============================================================
# STEP 1: APRIORI (FREQUENT ITEMSETS)
# ============================================================
def generate_frequent_itemsets(basket: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 (Apriori): Find frequent itemsets whose support >= minsup.
    """
    itemsets = apriori(
        basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN
    )

    if itemsets.empty:
        raise ValueError("No frequent itemsets found. Lower MIN_SUPPORT or increase MAX_LEN.")

    # Sort by support descending (most frequent first)
    itemsets = itemsets.sort_values("support", ascending=False).reset_index(drop=True)
    return itemsets


# ============================================================
# STEP 2: RULE GENERATION (SUPPORT, CONFIDENCE, LIFT, CONVICTION)
# ============================================================
def generate_rules(itemsets: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Generate association rules from frequent itemsets, filtering by minconf.

    Metrics produced by mlxtend include:
    - support
    - confidence
    - lift
    - conviction
    - antecedent support
    - consequent support
    """
    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    if rules.empty:
        raise ValueError("No rules found. Lower MIN_CONFIDENCE or MIN_SUPPORT.")

    # Convert frozensets into readable strings
    rules = rules.copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

    # Keep only the main columns commonly discussed in reports
    keep_cols = [
        "antecedents",
        "consequents",
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "conviction"
    ]
    rules = rules[keep_cols]

    # Sort by strongest associations first
    rules = rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False
    ).reset_index(drop=True)

    return rules


# ============================================================
# SAVE OUTPUTS
# ============================================================
def save_outputs(itemsets: pd.DataFrame, rules: pd.DataFrame) -> None:
    """
    Save itemsets and rules to CSV for report tables.
    """
    itemsets_path = OUT_DIR / "frequent_itemsets.csv"
    rules_path = OUT_DIR / "association_rules.csv"

    itemsets.to_csv(itemsets_path, index=False)
    rules.to_csv(rules_path, index=False)

    print("\nSaved outputs:")
    print(f" - {itemsets_path}")
    print(f" - {rules_path}")


# ============================================================
# PRINT SUMMARY (TOP RULES)
# ============================================================
def print_top_rules(rules: pd.DataFrame) -> None:
    print("\nTOP RULES (Sorted by Lift, then Confidence):")
    print(rules.head(TOP_K).to_string(index=False))


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading basket dataset...")
    basket = load_basket()
    print(f"Transactions: {basket.shape[0]} | Items: {basket.shape[1]}")

    print("\nRunning Apriori (Frequent Itemsets)...")
    itemsets = generate_frequent_itemsets(basket)
    print(f"Frequent itemsets found: {len(itemsets)}")

    print("\nGenerating Association Rules...")
    rules = generate_rules(itemsets)
    print(f"Rules found: {len(rules)}")

    save_outputs(itemsets, rules)
    print_top_rules(rules)

    print("\nDone.")


if __name__ == "__main__":
    main()