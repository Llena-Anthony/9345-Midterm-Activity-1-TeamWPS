from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

# PATH CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_survey.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = BASE_DIR / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# COLUMN NAMES
COL_AGE = "Age"
COL_GENDER = "Gender"
COL_OCC = "Current Occupation"
COL_INCOME = "Monthly income salary or allowance"
COL_FREQ = "How often do you shop for groceries in a month?"
COL_SPEND = "How much do you spend on groceries every month?"
COL_STORES = "What grocery stores do you usually buy your necessities from? (Select all that apply)"
COL_FACTORS = "What primary factors do you find important in a  grocery store? (Select all that apply)"
COL_PAYMENT = "What payment methods do you use in paying for groceries? (Select all that apply)"
COL_TRIPTIME = "Among the following when do you usually take a trip to the grocery store?"
COL_DURATION = "What would be the duration that you typically spend in a grocery store?"
COL_PRODUCTS = "Which among the following products and goods do you usually buy? (Select all that apply)"

INCOME_ORDER = [
    "Below ₱10,957",
    "₱10,957 to ₱21,914",
    "₱21,914 to ₱43,828",
    "₱43,828 to ₱76,669",
    "₱76,669 to ₱131,484",
    "₱131,484 to ₱219,140"
]

SPENDING_ORDER = [
    "Less than ₱5,000",
    "₱5,000 - ₱10,000",
    "₱10,001 - ₱15,000",
    "₱15,001 - ₱20,000",
    "More than ₱20,000"
]

OCC_ORDER = [
    "Student",
    "Employed (full-time)",
    "Employed (part-time)",
    "Self-employed",
    "Freelancer",
    "Unemployed",
    "Retired"
]
# HELPER FUNCTIONS
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def split_multiselect(value: str):
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
    return pd.Series(all_items).value_counts()


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str,
             top_n: int = None, horizontal: bool = False):
    s = series.copy()
    if top_n is not None:
        s = s.head(top_n)

    plt.figure(figsize=(10, 6))

    if horizontal:
        s.sort_values().plot(kind="barh")
    else:
        s.plot(kind="bar")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_pie(series: pd.Series, title: str, filename: str):
    plt.figure(figsize=(7, 7))
    series.plot(kind="pie", autopct="%1.1f%%")
    plt.title(title)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_crosstab_bar(ct: pd.DataFrame, title: str, xlabel: str, ylabel: str, filename: str,
                      stacked: bool = False):
    plt.figure(figsize=(11, 6))
    ct.plot(kind="bar", stacked=stacked, ax=plt.gca())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def save_crosstab_tables(ct: pd.DataFrame, prefix: str):
    """Save count and row-percentage tables."""
    ct.to_csv(TABLE_DIR / f"{prefix}_counts.csv")

    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.round(2)
    ct_pct.to_csv(TABLE_DIR / f"{prefix}_row_percent.csv")

    return ct_pct


def save_100pct_stacked_bar(ct_pct: pd.DataFrame, title: str, xlabel: str, ylabel: str, filename: str):
    """Save a 100% stacked bar chart using row percentages."""
    plt.figure(figsize=(11, 6))
    ct_pct.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Monthly Grocery Spending", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def reorder_if_present(series_or_index, preferred_order):
    """Keep only labels that exist, while following a preferred order."""
    present = list(series_or_index)
    ordered = [x for x in preferred_order if x in present]
    remaining = [x for x in present if x not in ordered]
    return ordered + remaining


# ANALYSES

def analysis_01_age():
    res = df[COL_AGE].value_counts()
    print("\n[1] AGE DISTRIBUTION")
    print(res)
    save_bar(res, "Age Distribution of Respondents", "Age Group", "Number of Respondents",
             "01_age_distribution.png")


def analysis_02_gender():
    res = df[COL_GENDER].value_counts()
    print("\n[2] GENDER DISTRIBUTION")
    print(res)
    save_pie(res, "Gender Distribution", "02_gender_distribution.png")


def analysis_03_occupation():
    res = df[COL_OCC].value_counts()
    print("\n[3] OCCUPATION DISTRIBUTION")
    print(res)
    save_bar(res, "Occupation of Respondents", "Occupation", "Count",
             "03_occupation_distribution.png")


def analysis_04_income():
    res = df[COL_INCOME].value_counts()
    print("\n[4] INCOME DISTRIBUTION")
    print(res)
    save_bar(res, "Monthly Income / Allowance Distribution", "Income Range", "Number of Respondents",
             "04_income_distribution.png")


def analysis_05_frequency():
    res = df[COL_FREQ].value_counts()
    print("\n[5] SHOPPING FREQUENCY")
    print(res)
    save_bar(res, "Shopping Frequency per Month", "Frequency", "Number of Respondents",
             "05_shopping_frequency.png")


def analysis_06_spending():
    res = df[COL_SPEND].value_counts()
    print("\n[6] MONTHLY GROCERY SPENDING")
    print(res)
    save_bar(res, "Monthly Grocery Spending", "Spending Range", "Number of Respondents",
             "06_monthly_spending.png")


def analysis_07_stores():
    res = multiselect_counts(df[COL_STORES])
    print("\n[7] GROCERY STORES FREQUENCY (MULTI-SELECT)")
    print(res)
    save_bar(res, "Most Common Grocery Stores Visited", "Number of Mentions", "Grocery Store",
             "07_grocery_stores_top10.png", top_n=10, horizontal=True)


def analysis_08_products():
    res = multiselect_counts(df[COL_PRODUCTS])
    print("\n[8] PRODUCT CATEGORY PURCHASE FREQUENCY (MULTI-SELECT)")
    print(res)
    save_bar(res, "Most Purchased Product Categories", "Number of Mentions", "Product Category",
             "08_product_categories_frequency.png", top_n=15, horizontal=True)


def analysis_09_frequency_vs_spending():
    ct = pd.crosstab(df[COL_FREQ], df[COL_SPEND])
    print("\n[9] SHOPPING FREQUENCY VS MONTHLY GROCERY SPENDING")
    print(ct)
    save_crosstab_bar(
        ct,
        "Shopping Frequency vs Monthly Grocery Spending",
        "Shopping Frequency",
        "Number of Respondents",
        "09_frequency_vs_spending.png",
        stacked=False
    )

def analysis_10_income_vs_spending():
        income = df[COL_INCOME].apply(clean_text)
        spend = df[COL_SPEND].apply(clean_text)

        ct = pd.crosstab(income, spend)

        row_order = reorder_if_present(ct.index, INCOME_ORDER)
        col_order = reorder_if_present(ct.columns, SPENDING_ORDER)
        ct = ct.reindex(index=row_order, columns=col_order, fill_value=0)

        print("\n[10] INCOME LEVEL VS MONTHLY GROCERY SPENDING")
        print(ct)

        save_crosstab_bar(
            ct,
            "Income Level vs Monthly Grocery Spending",
            "Monthly Income / Allowance",
            "Number of Respondents",
            "10_income_vs_spending_counts.png",
            stacked=False
        )

        ct_pct = save_crosstab_tables(ct, "10_income_vs_spending")
        save_100pct_stacked_bar(
            ct_pct,
            "Income Level vs Monthly Grocery Spending (Row %)",
            "Monthly Income / Allowance",
            "Percentage of Respondents",
            "10_income_vs_spending_100pct.png"
        )

def analysis_11_occupation_vs_spending():
        occ = df[COL_OCC].apply(clean_text)
        spend = df[COL_SPEND].apply(clean_text)

        ct = pd.crosstab(occ, spend)

        row_order = reorder_if_present(ct.index, OCC_ORDER)
        col_order = reorder_if_present(ct.columns, SPENDING_ORDER)
        ct = ct.reindex(index=row_order, columns=col_order, fill_value=0)

        print("\n[11] OCCUPATION VS MONTHLY GROCERY SPENDING")
        print(ct)

        save_crosstab_bar(
            ct,
            "Occupation vs Monthly Grocery Spending",
            "Occupation",
            "Number of Respondents",
            "11_occupation_vs_spending_counts.png",
            stacked=False
        )

        ct_pct = save_crosstab_tables(ct, "11_occupation_vs_spending")
        save_100pct_stacked_bar(
            ct_pct,
            "Occupation vs Monthly Grocery Spending (Row %)",
            "Occupation",
            "Percentage of Respondents",
            "11_occupation_vs_spending_100pct.png"
        )


def extra_primary_factors():
    res = multiselect_counts(df[COL_FACTORS])
    print("\n[EXTRA] PRIMARY FACTORS IN A GROCERY STORE")
    print(res)
    save_bar(res, "Primary Factors Considered in a Grocery Store", "Number of Mentions", "Factor",
             "10_primary_factors_top10.png", top_n=10, horizontal=True)


def extra_payment_methods():
    res = multiselect_counts(df[COL_PAYMENT])
    print("\n[EXTRA] PAYMENT METHODS")
    print(res)
    save_bar(res, "Payment Methods Used", "Number of Mentions", "Payment Method",
             "11_payment_methods.png", horizontal=True)


def extra_trip_time():
    res = df[COL_TRIPTIME].value_counts()
    print("\n[EXTRA] TIME OF TRIP TO GROCERY")
    print(res)
    save_bar(res, "Usual Time of Grocery Trips", "Time of Day", "Number of Respondents",
             "12_trip_time.png")


def main():
    analysis_01_age()
    analysis_02_gender()
    analysis_03_occupation()
    analysis_04_income()
    analysis_05_frequency()
    analysis_06_spending()
    analysis_07_stores()
    analysis_08_products()
    analysis_09_frequency_vs_spending()
    analysis_10_income_vs_spending()
    analysis_11_occupation_vs_spending()


    # optional extras
    extra_primary_factors()
    extra_payment_methods()
    extra_trip_time()


if __name__ == "__main__":
    main()