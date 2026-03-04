"""
Descriptive Analytics Script

Purpose:
Generate descriptive statistics and visualizations to understand
respondent demographics, shopping behavior, and purchasing patterns.

Input:
data/processed/cleaned_survey.csv
data/processed/basket_products.csv

Output:
Basic statistics and charts describing the dataset.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SURVEY_PATH = BASE_DIR / "data" / "processed" / "cleaned_survey.csv"
BASKET_PATH = BASE_DIR / "data" / "processed" / "basket_products.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

survey_df = pd.read_csv(SURVEY_PATH)
basket_df = pd.read_csv(BASKET_PATH)


# --------------------------------------------------
# 1. AGE DISTRIBUTION
# --------------------------------------------------

def analyze_age_distribution():
    age_counts = survey_df["Age"].value_counts()

    print("\nAge Distribution")
    print(age_counts)

    age_counts.plot(kind="bar", title="Age Distribution of Respondents")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Respondents")
    plt.show()


# --------------------------------------------------
# 2. GENDER DISTRIBUTION
# --------------------------------------------------

def analyze_gender_distribution():
    gender_counts = survey_df["Gender"].value_counts()

    print("\nGender Distribution")
    print(gender_counts)

    gender_counts.plot(kind="pie", autopct="%1.1f%%", title="Gender Distribution")
    plt.ylabel("")
    plt.show()


# --------------------------------------------------
# 3. OCCUPATION DISTRIBUTION
# --------------------------------------------------

def analyze_occupation_distribution():
    occupation_counts = survey_df["Current Occupation"].value_counts()

    print("\nOccupation Distribution")
    print(occupation_counts)

    occupation_counts.plot(kind="bar", title="Occupation of Respondents")
    plt.xlabel("Occupation")
    plt.ylabel("Count")
    plt.show()


# --------------------------------------------------
# 4. INCOME DISTRIBUTION
# --------------------------------------------------

def analyze_income_distribution():
    income_counts = survey_df["Monthly income salary or allowance"].value_counts()

    print("\nIncome Distribution")
    print(income_counts)

    income_counts.plot(kind="bar", title="Monthly Income Distribution")
    plt.xlabel("Income Range")
    plt.ylabel("Count")
    plt.show()


# --------------------------------------------------
# 5. SHOPPING FREQUENCY
# --------------------------------------------------

def analyze_shopping_frequency():
    freq_counts = survey_df["How often do you shop"].value_counts()

    print("\nShopping Frequency")
    print(freq_counts)

    freq_counts.plot(kind="bar", title="Shopping Frequency")
    plt.xlabel("Frequency")
    plt.ylabel("Count")
    plt.show()


# --------------------------------------------------
# 6. PAYMENT METHOD
# --------------------------------------------------

def analyze_payment_method():
    payment_counts = survey_df["What payment method"].value_counts()

    print("\nPayment Method")
    print(payment_counts)

    payment_counts.plot(kind="bar", title="Preferred Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Count")
    plt.show()


# --------------------------------------------------
# 7. SHOPPING TIME
# --------------------------------------------------

def analyze_shopping_time():
    time_counts = survey_df["When do you shop"].value_counts()

    print("\nShopping Time")
    print(time_counts)

    time_counts.plot(kind="bar", title="Preferred Shopping Time")
    plt.xlabel("Time of Day")
    plt.ylabel("Count")
    plt.show()


# --------------------------------------------------
# 8. PRODUCT CATEGORY FREQUENCY
# --------------------------------------------------

def analyze_product_categories():
    product_counts = basket_df.sum().sort_values(ascending=False)

    print("\nProduct Category Frequency")
    print(product_counts)

    product_counts.plot(kind="bar", title="Most Purchased Product Categories")
    plt.xlabel("Product Category")
    plt.ylabel("Number of Customers")
    plt.show()


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    analyze_age_distribution()
    analyze_gender_distribution()
    analyze_occupation_distribution()
    analyze_income_distribution()
    analyze_shopping_frequency()
    analyze_payment_method()
    analyze_shopping_time()
    analyze_product_categories()


if __name__ == "__main__":
    main()
