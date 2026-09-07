# Screen-Time-and-App-Usage-Dataset
📌 Project Description
  This project analyzes screen-time and app-usage data to understand digital device usage patterns.
 The dataset contains information about daily screen time, application usage, and other usage-related details. The analysis helps identify frequently used applications and understand screen-time trends.

 🎯 Objectives
- To analyze daily screen-time usage.
- To identify the most frequently used applications.
- To calculate average screen time.
- To understand app-wise usage patterns.
- To visualize the usage data using graphs.

🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Jupyter Notebook

📂 Dataset

The dataset contains screen-time and application usage information. It can be loaded into Python using the Pandas library.

💻 Python Coding

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("screen_time_app_usage.csv")

# Display first five records
print("First 5 Records:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Calculate average screen time
average_screen_time = data["Screen_Time"].mean()
print("\nAverage Screen Time:", average_screen_time)

# Find app-wise usage
app_usage = data.groupby("App")["Usage_Time"].sum()
print("\nApp-wise Usage:")
print(app_usage)

# Find the most used application
most_used_app = app_usage.idxmax()
print("\nMost Used Application:", most_used_app)

# Visualize app usage
app_usage.sort_values(ascending=False).plot(
    kind="bar",
    figsize=(8, 5),
    title="Application Usage"
)

plt.xlabel("Applications")
plt.ylabel("Usage Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

📊 Analysis

The project uses Pandas to read and analyze the dataset. Different applications are grouped together to calculate their total usage time.
Matplotlib is used to create a bar chart that makes it easier to compare application usage and identify the most frequently used application.

📈 Result

The analysis provides useful information about screen-time patterns and application usage. It helps users understand how much time is spent on digital devices and which applications are used the most.

🔮 Future Scope

This project can be extended by adding interactive dashboards, weekly and monthly usage analysis, prediction of future screen-time patterns, and recommendations for better digital usage habits.