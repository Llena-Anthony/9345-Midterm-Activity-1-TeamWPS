from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_survey.csv"

FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = BASE_DIR / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

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

COL_MOST_TIME = "Which section of a grocery store do you spend the most time during a typical visit? (Select all that apply)"
COL_LEAST_TIME = "Which section of a grocery store do you spend the least time during a typical visit? (Select all that apply)"
COL_FORGOTTEN = "Which products do you usually forget when you go grocery shopping? (Select all that apply)"

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


def clean_text(x):
    """Standardizes text values for consistent comparison."""
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def split_multiselect(value: str):
    """Splits multi-select survey responses into individual selections."""
    value = clean_text(value)
    if not value:
        return []

    if ":" in value:
        parts = re.split(r",\s(?=[A-Z][A-Za-z &/()-]+:)", value)
        return [p.strip() for p in parts if p.strip()]

    return [p.strip() for p in value.split(",") if p.strip()]


def multiselect_counts(series: pd.Series) -> pd.Series:
    """Counts the frequency of options in a multi-select column."""
    all_items = []
    for v in series.dropna():
        all_items.extend(split_multiselect(v))

    if not all_items:
        return pd.Series(dtype=int)

    return pd.Series(all_items).value_counts()


def save_bar(series, title, xlabel, ylabel, filename, top_n=None, horizontal=False):
    """Saves a bar chart from a pandas Series."""
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

    if horizontal:
        plt.yticks(rotation=0)
    else:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_pie(series, title, filename):
    """Saves a pie chart."""
    plt.figure(figsize=(7, 7))
    series.plot(kind="pie", autopct="%1.1f%%")
    plt.title(title)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_crosstab_bar(ct, title, xlabel, ylabel, filename, stacked=False):
    """Creates bar charts from cross-tabulated data."""
    plt.figure(figsize=(11, 6))
    ct.plot(kind="bar", stacked=stacked, ax=plt.gca())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_crosstab_tables(ct, prefix):
    """Exports cross-tabulation tables in counts and row percentages."""
    ct.to_csv(TABLE_DIR / f"{prefix}_counts.csv")

    row_sums = ct.sum(axis=1).replace(0, pd.NA)
    ct_pct = ct.div(row_sums, axis=0) * 100
    ct_pct = ct_pct.fillna(0).round(2)
    ct_pct.to_csv(TABLE_DIR / f"{prefix}_row_percent.csv")

    return ct_pct


def save_100pct_stacked_bar(ct_pct, title, xlabel, ylabel, filename):
    """Creates a 100% stacked bar chart using row percentages."""
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


def reorder_if_present(labels, preferred_order):
    """Reorders labels using a preferred order when available."""
    present = list(labels)
    ordered = [x for x in preferred_order if x in present]
    remaining = [x for x in present if x not in ordered]
    return ordered + remaining


def analysis_01_age():
    """Generates the age distribution chart."""
    res = df[COL_AGE].value_counts()
    print("\nAGE DISTRIBUTION")
    print(res)

    save_bar(
        res,
        "Age Distribution of Respondents",
        "Age Group",
        "Number of Respondents",
        "01_age_distribution.png"
    )


def analysis_02_gender():
    """Generates the gender distribution chart."""
    res = df[COL_GENDER].value_counts()
    print("\nGENDER DISTRIBUTION")
    print(res)

    save_pie(
        res,
        "Gender Distribution",
        "02_gender_distribution.png"
    )


def analysis_03_occupation():
    """Generates the occupation distribution chart."""
    res = df[COL_OCC].value_counts()
    print("\nOCCUPATION DISTRIBUTION")
    print(res)

    ordered_index = reorder_if_present(res.index, OCC_ORDER)
    res = res.reindex(ordered_index)

    save_bar(
        res,
        "Occupation of Respondents",
        "Occupation",
        "Count",
        "03_occupation_distribution.png"
    )


def analysis_04_income():
    """Generates the income distribution chart."""
    res = df[COL_INCOME].value_counts()
    print("\nINCOME DISTRIBUTION")
    print(res)

    ordered_index = reorder_if_present(res.index, INCOME_ORDER)
    res = res.reindex(ordered_index)

    save_bar(
        res,
        "Monthly Income / Allowance Distribution",
        "Income Range",
        "Number of Respondents",
        "04_income_distribution.png"
    )


def analysis_05_frequency():
    """Generates the shopping frequency chart."""
    res = df[COL_FREQ].value_counts()
    print("\nSHOPPING FREQUENCY")
    print(res)

    save_bar(
        res,
        "Shopping Frequency per Month",
        "Frequency",
        "Number of Respondents",
        "05_shopping_frequency.png"
    )


def analysis_06_spending():
    """Generates the monthly grocery spending chart."""
    res = df[COL_SPEND].value_counts()
    print("\nMONTHLY GROCERY SPENDING")
    print(res)

    ordered_index = reorder_if_present(res.index, SPENDING_ORDER)
    res = res.reindex(ordered_index)

    save_bar(
        res,
        "Monthly Grocery Spending",
        "Spending Range",
        "Number of Respondents",
        "06_monthly_spending.png"
    )


def analysis_07_stores():
    """Generates the top grocery stores frequency chart."""
    res = multiselect_counts(df[COL_STORES])
    print("\nGROCERY STORES FREQUENCY")
    print(res)

    save_bar(
        res,
        "Most Common Grocery Stores Visited",
        "Number of Mentions",
        "Grocery Store",
        "07_grocery_stores_top10.png",
        top_n=10,
        horizontal=True
    )


def analysis_08_products():
    """Generates the product category purchase frequency chart."""
    res = multiselect_counts(df[COL_PRODUCTS])
    print("\nPRODUCT CATEGORY FREQUENCY")
    print(res)

    save_bar(
        res,
        "Most Purchased Product Categories",
        "Number of Mentions",
        "Product Category",
        "08_product_categories_frequency.png",
        top_n=15,
        horizontal=True
    )


def analysis_09_frequency_vs_spending():
    """Generates a cross-tab chart for shopping frequency versus monthly spending."""
    ct = pd.crosstab(df[COL_FREQ], df[COL_SPEND])

    col_order = reorder_if_present(ct.columns, SPENDING_ORDER)
    ct = ct.reindex(columns=col_order, fill_value=0)

    print("\nSHOPPING FREQUENCY VS SPENDING")
    print(ct)

    save_crosstab_bar(
        ct,
        "Shopping Frequency vs Monthly Grocery Spending",
        "Shopping Frequency",
        "Number of Respondents",
        "09_frequency_vs_spending.png"
    )


def analysis_10_income_vs_spending():
    """Generates count and row-percentage charts for income versus grocery spending."""
    income = df[COL_INCOME].apply(clean_text)
    spend = df[COL_SPEND].apply(clean_text)

    ct = pd.crosstab(income, spend)

    row_order = reorder_if_present(ct.index, INCOME_ORDER)
    col_order = reorder_if_present(ct.columns, SPENDING_ORDER)
    ct = ct.reindex(index=row_order, columns=col_order, fill_value=0)

    print("\nINCOME VS SPENDING")
    print(ct)

    save_crosstab_bar(
        ct,
        "Income Level vs Monthly Grocery Spending",
        "Monthly Income / Allowance",
        "Number of Respondents",
        "10_income_vs_spending_counts.png"
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
    """Generates count and row-percentage charts for occupation versus grocery spending."""
    occ = df[COL_OCC].apply(clean_text)
    spend = df[COL_SPEND].apply(clean_text)

    ct = pd.crosstab(occ, spend)

    row_order = reorder_if_present(ct.index, OCC_ORDER)
    col_order = reorder_if_present(ct.columns, SPENDING_ORDER)
    ct = ct.reindex(index=row_order, columns=col_order, fill_value=0)

    print("\nOCCUPATION VS SPENDING")
    print(ct)

    save_crosstab_bar(
        ct,
        "Occupation vs Monthly Grocery Spending",
        "Occupation",
        "Number of Respondents",
        "11_occupation_vs_spending_counts.png"
    )

    ct_pct = save_crosstab_tables(ct, "11_occupation_vs_spending")

    save_100pct_stacked_bar(
        ct_pct,
        "Occupation vs Monthly Grocery Spending (Row %)",
        "Occupation",
        "Percentage of Respondents",
        "11_occupation_vs_spending_100pct.png"
    )


def analysis_12_most_time_spent():
    """Generates the chart for sections where customers spend the most time."""
    res = multiselect_counts(df[COL_MOST_TIME])
    print("\nMOST TIME SPENT SECTION")
    print(res)

    save_bar(
        res,
        "Sections Where Customers Spend the Most Time",
        "Number of Mentions",
        "Store Section",
        "12_most_time_spent_section.png",
        horizontal=True
    )


def analysis_13_least_time_spent():
    """Generates the chart for sections where customers spend the least time."""
    res = multiselect_counts(df[COL_LEAST_TIME])
    print("\nLEAST TIME SPENT SECTION")
    print(res)

    save_bar(
        res,
        "Sections Where Customers Spend the Least Time",
        "Number of Mentions",
        "Store Section",
        "13_least_time_spent_section.png",
        horizontal=True
    )


def analysis_14_forgotten_products():
    """Generates the chart for products customers usually forget to buy."""
    res = multiselect_counts(df[COL_FORGOTTEN])
    print("\nFORGOTTEN PRODUCTS")
    print(res)

    save_bar(
        res,
        "Products Customers Usually Forget to Buy",
        "Number of Mentions",
        "Product Category",
        "14_forgotten_products.png",
        horizontal=True
    )


def extra_primary_factors():
    """Generates the chart for primary factors considered in a grocery store."""
    res = multiselect_counts(df[COL_FACTORS])
    print("\nPRIMARY FACTORS")
    print(res)

    save_bar(
        res,
        "Primary Factors Considered in a Grocery Store",
        "Number of Mentions",
        "Factor",
        "15_primary_factors.png",
        horizontal=True
    )


def extra_payment_methods():
    """Generates the chart for payment methods used by respondents."""
    res = multiselect_counts(df[COL_PAYMENT])
    print("\nPAYMENT METHODS")
    print(res)

    save_bar(
        res,
        "Payment Methods Used",
        "Number of Mentions",
        "Payment Method",
        "16_payment_methods.png",
        horizontal=True
    )


def extra_trip_time():
    """Generates the chart for usual grocery trip time."""
    res = df[COL_TRIPTIME].value_counts()
    print("\nTIME OF TRIP TO GROCERY")
    print(res)

    save_bar(
        res,
        "Usual Time of Grocery Trips",
        "Time of Day",
        "Number of Respondents",
        "17_trip_time.png"
    )


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
    analysis_12_most_time_spent()
    analysis_13_least_time_spent()
    analysis_14_forgotten_products()

    extra_primary_factors()
    extra_payment_methods()
    extra_trip_time()


if __name__ == "__main__":
    main()