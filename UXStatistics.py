import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tkinter import Tk, filedialog
import os


# Function to select the file
def select_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select file",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    return file_path


# Function to calculate Cronbach's Alpha
def cronbach_alpha(df):
    df = df.dropna(axis=1)  # Drop columns with NaN values for the calculation
    item_scores = df.T
    item_vars = item_scores.var(axis=1, ddof=1)
    total_var = item_scores.sum(axis=0).var(ddof=1)
    n_items = len(df.columns)
    if total_var == 0:
        return np.nan
    alpha = n_items / (n_items - 1) * (1 - item_vars.sum() / total_var)
    return alpha


# Load and clean the data
def load_and_clean_data(file_path):
    df = pd.read_excel(file_path)
    df.replace({"MISSING": np.nan, "DK": np.nan}, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x, axis=0)
    return df


# Descriptive Statistics
def descriptive_statistics(df):
    descriptive_stats = df.describe(include='all')
    print("Descriptive Statistics:")
    print(descriptive_stats)
    return descriptive_stats


# Save descriptive statistics to Excel
def save_descriptive_statistics_to_excel(descriptive_stats, file_path):
    base_path = os.path.splitext(file_path)[0]
    output_path = f"{base_path}_descriptive_statistics.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        descriptive_stats.to_excel(writer, sheet_name='Descriptive Statistics')
    print(f"Descriptive statistics saved to {output_path}")


# Visualization
def visualize_data(df, cols_to_convert):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[cols_to_convert])
    plt.title('Distribution of Responses for Each Question')
    plt.xticks(rotation=45)
    plt.show()


# Item-Total Correlation Analysis
def item_total_correlation(df, cols_to_convert):
    item_total_corr = df[cols_to_convert].corrwith(df[cols_to_convert].mean(axis=1))
    low_corr_items = item_total_corr[item_total_corr < 0.3]  # Threshold can be adjusted
    print("Items with low item-total correlation:")
    print(low_corr_items)
    return item_total_corr, low_corr_items


# Main analysis function
def analyze_phantom_validation():
    file_path = select_file()
    if not file_path:
        raise ValueError("No file selected!")

    df = load_and_clean_data(file_path)
    descriptive_stats = descriptive_statistics(df)
    save_descriptive_statistics_to_excel(descriptive_stats, file_path)

    cols_to_convert = [
        'PhysicalScalp', 'PhysicalBone', 'PhysicalBrain', 'PhysicalLandmarks',
        'RealismLandmarks', 'RealismIncision', 'RealismDrilling', 'RealismInsertion',
        'RealismTunneling', 'RealismFixation', 'RealismClosure', 'Value'
    ]

    if not all(col in df.columns for col in cols_to_convert):
        missing_cols = [col for col in cols_to_convert if col not in df.columns]
        raise ValueError(f"The following required columns are missing from the data: {missing_cols}")

    reliability = cronbach_alpha(df[cols_to_convert])
    print(f"Cronbach's Alpha: {reliability}")
    item_total_corr, low_corr_items = item_total_correlation(df, cols_to_convert)
    visualize_data(df, cols_to_convert)


# Call the function
analyze_phantom_validation()
