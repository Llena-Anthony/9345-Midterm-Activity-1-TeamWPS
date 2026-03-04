from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_survey.csv"

df = pd.read_csv(DATA_PATH)

# ============================================================
# COLUMN NAMES (YOUR EXACT HEADERS)
# ============================================================
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

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x

def split_multiselect(value: str):
    """
    Safe split for Google Forms multi-select fields.
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
    """
    Count frequency of each selection in a multi-select column.
    """
    all_items = []
    for v in series:
        all_items.extend(split_multiselect(v))
    return pd.Series(all_items).value_counts()

def show_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, top_n: int = None):
    """
    Display a bar chart for a series (value_counts output).
    """
    s = series.copy()
    if top_n is not None:
        s = s.head(top_n)

    s.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# ============================================================
# DESCRIPTIVE ANALYSES (8)
# ============================================================

# 1) Age distribution
def analysis_01_age():
    res = df[COL_AGE].value_counts()
    print("\n[1] AGE DISTRIBUTION")
    print(res)
    show_bar(res, "Age Distribution of Respondents", "Age Group", "Number of Respondents")

# 2) Gender distribution
def analysis_02_gender():
    res = df[COL_GENDER].value_counts()
    print("\n[2] GENDER DISTRIBUTION")
    print(res)
    res.plot(kind="pie", autopct="%1.1f%%", title="Gender Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# 3) Occupation distribution
def analysis_03_occupation():
    res = df[COL_OCC].value_counts()
    print("\n[3] OCCUPATION DISTRIBUTION")
    print(res)
    show_bar(res, "Occupation of Respondents", "Occupation", "Count")

# 4) Monthly income distribution
def analysis_04_income():
    res = df[COL_INCOME].value_counts()
    print("\n[4] INCOME DISTRIBUTION")
    print(res)
    show_bar(res, "Monthly Income / Allowance Distribution", "Income Range", "Number of Respondents")

# 5) Shopping frequency distribution
def analysis_05_frequency():
    res = df[COL_FREQ].value_counts()
    print("\n[5] SHOPPING FREQUENCY")
    print(res)
    show_bar(res, "Shopping Frequency per Month", "Frequency", "Number of Respondents")

# 6) Monthly grocery spending distribution
def analysis_06_spending():
    res = df[COL_SPEND].value_counts()
    print("\n[6] MONTHLY GROCERY SPENDING")
    print(res)
    show_bar(res, "Monthly Grocery Spending", "Spending Range", "Number of Respondents")

# 7) Most common grocery stores visited (multi-select)
def analysis_07_stores():
    res = multiselect_counts(df[COL_STORES])
    print("\n[7] GROCERY STORES FREQUENCY (MULTI-SELECT)")
    print(res)
    show_bar(res, "Most Common Grocery Stores Visited", "Grocery Store", "Number of Mentions", top_n=10)

# 8) Product categories purchased (multi-select; supports ARM context)
def analysis_08_products():
    res = multiselect_counts(df[COL_PRODUCTS])
    print("\n[8] PRODUCT CATEGORY PURCHASE FREQUENCY (MULTI-SELECT)")
    print(res)
    show_bar(res, "Most Purchased Product Categories", "Product Category", "Number of Mentions", top_n=15)


# ============================================================
# OPTIONAL EXTRA (GOOD ADD-ONS IF YOU WANT 10+ ANALYSES)
# ============================================================
def extra_payment_methods():
    res = multiselect_counts(df[COL_PAYMENT])
    print("\n[EXTRA] PAYMENT METHODS (MULTI-SELECT)")
    print(res)
    show_bar(res, "Payment Methods Used", "Payment Method", "Number of Mentions")

def extra_primary_factors():
    res = multiselect_counts(df[COL_FACTORS])
    print("\n[EXTRA] PRIMARY FACTORS IN A GROCERY STORE (MULTI-SELECT)")
    print(res)
    show_bar(res, "Primary Factors Considered in a Grocery Store", "Factor", "Number of Mentions", top_n=10)

def extra_trip_time():
    res = df[COL_TRIPTIME].value_counts()
    print("\n[EXTRA] TIME OF TRIP TO GROCERY")
    print(res)
    show_bar(res, "Usual Time of Grocery Trips", "Time of Day", "Number of Respondents")


# ============================================================
# MAIN
# ============================================================
def main():
    analysis_01_age()
    analysis_02_gender()
    analysis_03_occupation()
    analysis_04_income()
    analysis_05_frequency()
    analysis_06_spending()
    analysis_07_stores()
    analysis_08_products()

    # Optional extras (uncomment if needed)
    # extra_payment_methods()
    # extra_primary_factors()
    # extra_trip_time()

if __name__ == "__main__":
    main()