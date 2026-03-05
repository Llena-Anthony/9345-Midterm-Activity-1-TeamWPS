from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

# PATH CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent

SURVEY_PATH = BASE_DIR / "data" / "processed" / "cleaned_survey.csv"
BASKET_PATH = BASE_DIR / "data" / "processed" / "basket_products.csv"
RULES_PATH = BASE_DIR / "outputs" / "tables" / "association_rules.csv"

FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# LOAD DATA
survey_df = pd.read_csv(SURVEY_PATH)
basket_df = pd.read_csv(BASKET_PATH)

rules_df = None
if RULES_PATH.exists():
    rules_df = pd.read_csv(RULES_PATH)


# HELPERS
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def split_multiselect(value: str):
    """
    Split multi-select survey answers safely.
    - If colon-labeled options exist, split only at new label boundaries.
    - Otherwise, split by commas.
    """
    value = clean_text(value)
    if not value:
        return []
    if ":" in value:
        parts = re.split(r",\s(?=[A-Z][A-Za-z &/()-]+:)", value)
        return [p.strip() for p in parts if p.strip()]
    return [p.strip() for p in value.split(",") if p.strip()]


def multiselect_counts(series: pd.Series) -> pd.Series:
    all_items = []
    for v in series:
        all_items.extend(split_multiselect(v))
    if not all_items:
        return pd.Series(dtype=int)
    return pd.Series(all_items).value_counts()


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, top_n: int = None):

    s = series.copy()
    if top_n is not None:
        s = s.head(top_n)

    # Bigger figure for long labels
    plt.figure(figsize=(12, 6))
    ax = s.plot(kind="bar")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Rotate labels for readability
    plt.xticks(rotation=45, ha="right")

    # Save with tight bounding box so labels are not cut off
    plt.savefig(FIG_DIR / filename, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()


def save_pie(series: pd.Series, title: str, filename: str):
    plt.figure(figsize=(7, 6))
    series.plot(kind="pie", autopct="%1.1f%%")
    plt.title(title)
    plt.ylabel("")
    plt.savefig(FIG_DIR / filename, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()


# DESCRIPTIVE VISUALIZATIONS
def plot_descriptive():
    """
    Generates and saves descriptive analytics charts.
    """
    # Single-column analyses
    save_bar(
        survey_df["Age"].value_counts(),
        "Age Distribution of Respondents",
        "Age Group",
        "Number of Respondents",
        "01_age_distribution.png"
    )

    save_pie(
        survey_df["Gender"].value_counts(),
        "Gender Distribution of Respondents",
        "02_gender_distribution.png"
    )

    save_bar(
        survey_df["Current Occupation"].value_counts(),
        "Occupation Distribution of Respondents",
        "Occupation",
        "Number of Respondents",
        "03_occupation_distribution.png"
    )

    save_bar(
        survey_df["Monthly income salary or allowance"].value_counts(),
        "Monthly Income / Allowance Distribution",
        "Income Range",
        "Number of Respondents",
        "04_income_distribution.png"
    )

    save_bar(
        survey_df["How often do you shop for groceries in a month?"].value_counts(),
        "Shopping Frequency per Month",
        "Frequency",
        "Number of Respondents",
        "05_shopping_frequency.png"
    )

    save_bar(
        survey_df["How much do you spend on groceries every month?"].value_counts(),
        "Monthly Grocery Spending",
        "Spending Range",
        "Number of Respondents",
        "06_monthly_spending.png"
    )

    save_bar(
        survey_df["Among the following when do you usually take a trip to the grocery store?"].value_counts(),
        "Usual Time of Grocery Trip",
        "Time of Day",
        "Number of Respondents",
        "07_trip_time.png"
    )

    save_bar(
        survey_df["What would be the duration that you typically spend in a grocery store?"].value_counts(),
        "Typical Duration of Grocery Visit",
        "Duration",
        "Number of Respondents",
        "08_trip_duration.png"
    )

    # Multi-select analyses (count selections)
    store_counts = multiselect_counts(
        survey_df["What grocery stores do you usually buy your necessities from? (Select all that apply)"]
    )
    if not store_counts.empty:
        save_bar(
            store_counts,
            "Most Common Grocery Stores Visited",
            "Grocery Store",
            "Number of Mentions",
            "09_grocery_stores_top10.png",
            top_n=10
        )

    factor_counts = multiselect_counts(
        survey_df["What primary factors do you find important in a  grocery store? (Select all that apply)"]
    )
    if not factor_counts.empty:
        save_bar(
            factor_counts,
            "Primary Factors Considered in a Grocery Store",
            "Factor",
            "Number of Mentions",
            "10_primary_factors_top10.png",
            top_n=10
        )

    payment_counts = multiselect_counts(
        survey_df["What payment methods do you use in paying for groceries? (Select all that apply)"]
    )
    if not payment_counts.empty:
        save_bar(
            payment_counts,
            "Payment Methods Used by Respondents",
            "Payment Method",
            "Number of Mentions",
            "11_payment_methods.png"
        )

    # Product category frequency (from basket dataset)
    product_freq = basket_df.sum().sort_values(ascending=False)
    save_bar(
        product_freq,
        "Product Categories Purchased (Frequency)",
        "Product Category",
        "Number of Respondents",
        "12_product_categories_frequency.png"
    )


# ARM VISUALIZATIONS
def plot_arm():
    """
    Generates and saves ARM-related charts based on association_rules.csv.
    """
    if rules_df is None or rules_df.empty:
        print("No association rules file found. Run association_rule_mining.py first.")
        return

    # Top rules by lift (bar chart)
    top_lift = rules_df.sort_values("lift", ascending=False).head(10)
    labels = top_lift["antecedents"] + " → " + top_lift["consequents"]
    lift_vals = top_lift["lift"]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(lift_vals)), lift_vals)
    plt.xticks(range(len(lift_vals)), labels, rotation=45, ha="right")
    plt.title("Top 10 Association Rules by Lift")
    plt.xlabel("Rule (Antecedent → Consequent)")
    plt.ylabel("Lift")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ARM_top10_rules_by_lift.png", dpi=300, facecolor="white")
    plt.close()

    # Scatter plot: Support vs Confidence (size by lift)
    plt.figure(figsize=(7, 5))
    plt.scatter(rules_df["support"], rules_df["confidence"], s=(rules_df["lift"] * 30))
    plt.title("Association Rules: Support vs Confidence (size ~ Lift)")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ARM_support_vs_confidence.png", dpi=300, facecolor="white")
    plt.close()


# MAIN
def main():
    print("Generating descriptive visualizations...")
    plot_descriptive()
    print(f"Saved descriptive figures to: {FIG_DIR}")

    print("\nGenerating ARM visualizations...")
    plot_arm()
    print(f"Saved ARM figures to: {FIG_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()