"""
Screen Time and App Usage Dataset
Data Integrity Audit
"""

import numpy as np
import pandas as pd


# =========================================================
# CSV FILE
# =========================================================

CSV_FILE = "screen_time_app_usage_dataset.csv"


# =========================================================
# COLUMN NAMES
# =========================================================
# Mapped to the actual dataset header:
# user_id,date,app_name,category,screen_time_min,launches,interactions,
# is_productive,youtube_views,youtube_likes,youtube_comments,
# extra_col_11..extra_col_23

USER_ID = "user_id"
APP = "app_name"
CATEGORY = "category"
SCREEN_TIME = "screen_time_min"
APP_OPENS = "launches"
NOTIFICATIONS = "interactions"
# No Data_Usage_MB column exists in this dataset — data-usage metrics removed.

HIGH_SCREEN_TIME_MINUTES = 240.0


# =========================================================
# 1. SHAPE
# =========================================================

def shape(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }


# =========================================================
# 2. UNIQUENESS
# =========================================================

def uniqueness(df: pd.DataFrame) -> dict[str, int]:
    return {
        "unique_users": int(df[USER_ID].nunique()),
        "unique_apps": int(df[APP].nunique()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# =========================================================
# 3. MISSING VALUES
# =========================================================

def missingness(df: pd.DataFrame) -> dict[str, int]:
    counts = df.isna().sum()

    return {
        column: int(count)
        for column, count in counts.items()
        if count > 0
    }


# =========================================================
# 4. CARDINALITIES
# =========================================================

def cardinalities(df: pd.DataFrame) -> dict[str, int]:
    columns = [USER_ID, APP, CATEGORY]

    return {
        column: int(df[column].nunique(dropna=True))
        for column in columns
    }


# =========================================================
# 5. SCREEN TIME SUMMARY
# =========================================================

def screen_time_summary(df: pd.DataFrame) -> dict[str, float]:

    values = df[SCREEN_TIME].dropna().to_numpy(dtype=float)

    return {
        "total_screen_time_minutes": round(float(np.sum(values)), 2),
        "average_screen_time_minutes": round(float(np.mean(values)), 2),
        "median_screen_time_minutes": round(float(np.median(values)), 2),
        "minimum_screen_time_minutes": round(float(np.min(values)), 2),
        "maximum_screen_time_minutes": round(float(np.max(values)), 2),
        "std_screen_time_minutes": round(float(np.std(values)), 2),
    }


# =========================================================
# 6. SCREEN TIME IN HOURS
# =========================================================

def screen_time_hours(df: pd.DataFrame) -> dict[str, float]:

    values = df[SCREEN_TIME].dropna().to_numpy(dtype=float)

    return {
        "total_screen_time_hours": round(float(np.sum(values) / 60), 2),
        "average_screen_time_hours": round(float(np.mean(values) / 60), 2),
        "maximum_screen_time_hours": round(float(np.max(values) / 60), 2),
    }


# =========================================================
# 7. APP USAGE SUMMARY
# =========================================================

def app_usage_summary(df: pd.DataFrame) -> dict:

    opens = df[APP_OPENS].dropna().to_numpy(dtype=float)
    notifications = df[NOTIFICATIONS].dropna().to_numpy(dtype=float)

    most_opened_index = df[APP_OPENS].idxmax()

    return {
        "total_app_opens": int(np.sum(opens)),
        "average_app_opens": round(float(np.mean(opens)), 2),
        "maximum_app_opens": int(np.max(opens)),
        "most_opened_app": str(df.loc[most_opened_index, APP]),
        "total_notifications": int(np.sum(notifications)),
        "average_notifications": round(float(np.mean(notifications)), 2),
        "maximum_notifications": int(np.max(notifications)),
    }


# =========================================================
# 8. CATEGORY-WISE SCREEN TIME
# =========================================================

def category_screen_time(df: pd.DataFrame) -> dict[str, float]:

    rates = (
        df.groupby(CATEGORY, observed=True)[SCREEN_TIME]
        .mean()
        .sort_values(ascending=False)
    )

    return {
        str(category): round(float(value), 2)
        for category, value in rates.items()
    }


# =========================================================
# 9. CATEGORY USAGE SPREAD
# =========================================================

def category_usage_spread(df: pd.DataFrame) -> dict:

    rates = (
        df.groupby(CATEGORY, observed=True)[SCREEN_TIME]
        .mean()
        .sort_values()
    )

    if rates.empty:
        return {
            "levels": 0,
            "spread_minutes": 0.0
        }

    return {
        "levels": int(len(rates)),
        "minimum_average_minutes": round(float(rates.iloc[0]), 2),
        "maximum_average_minutes": round(float(rates.iloc[-1]), 2),
        "spread_minutes": round(
            float(rates.iloc[-1] - rates.iloc[0]), 2
        ),
        "lowest_category": str(rates.index[0]),
        "highest_category": str(rates.index[-1]),
    }


# =========================================================
# 10. APP-WISE SCREEN TIME
# =========================================================

def app_screen_time(df: pd.DataFrame) -> dict[str, float]:

    usage = (
        df.groupby(APP, observed=True)[SCREEN_TIME]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        str(app): round(float(minutes), 2)
        for app, minutes in usage.items()
    }


# =========================================================
# 11. MOST USED APP
# =========================================================

def most_used_app(df: pd.DataFrame) -> dict:

    usage = (
        df.groupby(APP, observed=True)[SCREEN_TIME]
        .sum()
        .sort_values(ascending=False)
    )

    if usage.empty:
        return {
            "most_used_app": "",
            "screen_time_minutes": 0.0
        }

    return {
        "most_used_app": str(usage.index[0]),
        "screen_time_minutes": round(float(usage.iloc[0]), 2),
    }


# =========================================================
# 12. HIGH SCREEN TIME
# =========================================================

def high_screen_time(df: pd.DataFrame) -> dict:

    values = df[SCREEN_TIME].to_numpy(dtype=float)

    high = values >= HIGH_SCREEN_TIME_MINUTES

    return {
        "threshold_minutes": HIGH_SCREEN_TIME_MINUTES,
        "high_screen_time_rows": int(np.sum(high)),
        "high_screen_time_pct": round(
            float(np.mean(high) * 100), 2
        ),
    }


# =========================================================
# 13. CORRELATIONS
# =========================================================

def usage_correlations(df: pd.DataFrame) -> dict[str, float]:

    data = df[
        [
            SCREEN_TIME,
            APP_OPENS,
            NOTIFICATIONS,
        ]
    ].dropna()

    return {
        "screen_time_vs_app_opens": round(
            float(data[SCREEN_TIME].corr(data[APP_OPENS])),
            3
        ),

        "screen_time_vs_notifications": round(
            float(data[SCREEN_TIME].corr(data[NOTIFICATIONS])),
            3
        ),
    }


# =========================================================
# 14. VALUE SANITY CHECK
# =========================================================

def value_sanity(df: pd.DataFrame) -> dict[str, int]:

    return {
        "negative_screen_time_rows":
            int((df[SCREEN_TIME] < 0).sum()),

        "negative_notifications_rows":
            int((df[NOTIFICATIONS] < 0).sum()),

        "negative_app_opens_rows":
            int((df[APP_OPENS] < 0).sum()),
    }


# =========================================================
# 15. USER-WISE SCREEN TIME
# =========================================================

def user_screen_time(df: pd.DataFrame) -> dict[str, float]:

    usage = (
        df.groupby(USER_ID, observed=True)[SCREEN_TIME]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        str(user): round(float(minutes), 2)
        for user, minutes in usage.items()
    }


# =========================================================
# 16. HIGHEST SCREEN TIME USER
# =========================================================

def highest_screen_time_user(df: pd.DataFrame) -> dict:

    usage = (
        df.groupby(USER_ID, observed=True)[SCREEN_TIME]
        .sum()
        .sort_values(ascending=False)
    )

    if usage.empty:
        return {
            "highest_screen_time_user": "",
            "screen_time_minutes": 0.0
        }

    return {
        "highest_screen_time_user": str(usage.index[0]),
        "screen_time_minutes": round(
            float(usage.iloc[0]), 2
        ),
    }


# =========================================================
# RUN ALL FUNCTIONS
# =========================================================

def run_all(df: pd.DataFrame | None = None) -> dict:

    data = df if df is not None else pd.read_csv(CSV_FILE)

    return {
        "shape": shape(data),
        "uniqueness": uniqueness(data),
        "missingness": missingness(data),
        "cardinalities": cardinalities(data),
        "screen_time_summary": screen_time_summary(data),
        "screen_time_hours": screen_time_hours(data),
        "app_usage_summary": app_usage_summary(data),
        "category_screen_time": category_screen_time(data),
        "category_usage_spread": category_usage_spread(data),
        "app_screen_time": app_screen_time(data),
        "most_used_app": most_used_app(data),
        "high_screen_time": high_screen_time(data),
        "usage_correlations": usage_correlations(data),
        "value_sanity": value_sanity(data),
        "highest_screen_time_user": highest_screen_time_user(data),
    }


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    # Read CSV file
    df = pd.read_csv(CSV_FILE)

    # -----------------------------------------------------
    # PRINT COMPLETE CSV DATASET
    # -----------------------------------------------------

    print("\n========== CSV DATASET ==========")
    print(df.to_string())

    # -----------------------------------------------------
    # PRINT DATASET SHAPE
    # -----------------------------------------------------

    print("\n========== SHAPE ==========")
    print(shape(df))

    # -----------------------------------------------------
    # UNIQUENESS
    # -----------------------------------------------------

    print("\n========== UNIQUENESS ==========")
    print(uniqueness(df))

    # -----------------------------------------------------
    # MISSINGNESS
    # -----------------------------------------------------

    print("\n========== MISSINGNESS ==========")
    print(missingness(df))

    # -----------------------------------------------------
    # CARDINALITIES
    # -----------------------------------------------------

    print("\n========== CARDINALITIES ==========")
    print(cardinalities(df))

    # -----------------------------------------------------
    # SCREEN TIME SUMMARY
    # -----------------------------------------------------

    print("\n========== SCREEN TIME SUMMARY ==========")
    print(screen_time_summary(df))

    # -----------------------------------------------------
    # SCREEN TIME HOURS
    # -----------------------------------------------------

    print("\n========== SCREEN TIME HOURS ==========")
    print(screen_time_hours(df))

    # -----------------------------------------------------
    # APP USAGE
    # -----------------------------------------------------

    print("\n========== APP USAGE SUMMARY ==========")
    print(app_usage_summary(df))

    # -----------------------------------------------------
    # CATEGORY SCREEN TIME
    # -----------------------------------------------------

    print("\n========== CATEGORY SCREEN TIME ==========")
    print(category_screen_time(df))

    # -----------------------------------------------------
    # CATEGORY USAGE SPREAD
    # -----------------------------------------------------

    print("\n========== CATEGORY USAGE SPREAD ==========")
    print(category_usage_spread(df))

    # -----------------------------------------------------
    # APP SCREEN TIME
    # -----------------------------------------------------

    print("\n========== APP SCREEN TIME ==========")
    print(app_screen_time(df))

    # -----------------------------------------------------
    # MOST USED APP
    # -----------------------------------------------------

    print("\n========== MOST USED APP ==========")
    print(most_used_app(df))

    # -----------------------------------------------------
    # HIGH SCREEN TIME
    # -----------------------------------------------------

    print("\n========== HIGH SCREEN TIME ==========")
    print(high_screen_time(df))

    # -----------------------------------------------------
    # CORRELATIONS
    # -----------------------------------------------------

    print("\n========== USAGE CORRELATIONS ==========")
    print(usage_correlations(df))

    # -----------------------------------------------------
    # VALUE SANITY
    # -----------------------------------------------------

    print("\n========== VALUE SANITY ==========")
    print(value_sanity(df))

    # -----------------------------------------------------
    # HIGHEST SCREEN TIME USER
    # -----------------------------------------------------

    print("\n========== HIGHEST SCREEN TIME USER ==========")
    print(highest_screen_time_user(df))

    print("\n========== AUDIT COMPLETED ==========")