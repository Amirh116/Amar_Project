import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ساخت پوشه‌ها
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 1. دانلود داده‌های NHANES 2021-2023
base_url = "https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/"
files = ["DEMO_L.XPT", "BMX_L.XPT", "BPXO_L.XPT"]

for f in files:
    path = os.path.join("data", f)
    if not os.path.exists(path):
        print(f"Downloading {f}...")
        res = requests.get(base_url + f)
        with open(path, "wb") as out:
            out.write(res.content)
    else:
        print(f"{f} already exists.")

# 2. بارگذاری و ادغام داده‌ها
demo = pd.read_sas("data/DEMO_L.XPT")[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'WTMEC2YR']]
bmx = pd.read_sas("data/BMX_L.XPT")[['SEQN', 'BMXBMI', 'BMXWAIST']]
bpx = pd.read_sas("data/BPXO_L.XPT")[['SEQN', 'BPXOSY1', 'BPXODI1']]

df = demo.merge(bmx, on='SEQN').merge(bpx, on='SEQN')

# 3. پیش‌پردازش و فیلتر بزرگسالان (>= 20 سال)
df = df[df['RIDAGEYR'] >= 20].copy()
df = df.rename(columns={
    'RIDAGEYR': 'Age',
    'BMXBMI': 'BMI',
    'BPXOSY1': 'SBP',
    'BPXODI1': 'DBP',
    'BMXWAIST': 'Waist'
})

# تعریف متغیر جنسیت
df['Gender'] = df['RIAGENDR'].map({1.0: 'Male', 2.0: 'Female'})

# حذف رکوردهای بدون داده در متغیرهای اصلی
df = df.dropna(subset=['Age', 'BMI', 'SBP', 'Gender']).copy()

# دسته‌بندی BMI بر اساس استاندارد WHO
bins = [0, 18.5, 24.9, 29.9, 100]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['BMI_Cat'] = pd.cut(df['BMI'], bins=bins, labels=labels)

print(f"\n==========================================")
print(f"Total Valid Adult Sample Size: {len(df)}")
print(f"==========================================")

# 4. محاسبه شاخص‌های توصیفی به تفکیک جنسیت
def get_stats(data, cols):
    records = []
    for g in ['Male', 'Female']:
        sub = data[data['Gender'] == g]
        for col in cols:
            vals = sub[col].dropna()
            q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
            records.append({
                'Gender': g,
                'Variable': col,
                'Mean': round(vals.mean(), 2),
                'SD': round(vals.std(), 2),
                'Median': round(vals.median(), 2),
                'IQR': round(q75 - q25, 2),
                'Min': round(vals.min(), 2),
                'Max': round(vals.max(), 2)
            })
    return pd.DataFrame(records)

stats_df = get_stats(df, ['Age', 'BMI', 'SBP'])
print("\n--- Descriptive Summary by Gender ---")
print(stats_df.to_string(index=False))

# توزیع دسته‌های BMI
print("\n--- BMI WHO Categories Distribution ---")
print(pd.crosstab(df['Gender'], df['BMI_Cat'], margins=True))

# 5. ترسیم نمودارها
sns.set_theme(style="whitegrid")

# نمودار 1: هیستوگرام و KDE فشار خون انقباضی
plt.figure(figsize=(7, 4.5))
sns.histplot(data=df, x='SBP', hue='Gender', kde=True, bins=30, alpha=0.5)
plt.title("Distribution of Systolic Blood Pressure (SBP)")
plt.xlabel("SBP (mmHg)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/plot1_sbp_distribution.png", dpi=200)
plt.close()

# نمودار 2: نمودار جعبه‌ای BMI به تفکیک جنسیت
plt.figure(figsize=(6, 4.5))
sns.boxplot(data=df, x='Gender', y='BMI', palette='Set2')
plt.title("Body Mass Index (BMI) by Gender")
plt.xlabel("Gender")
plt.ylabel("BMI (kg/m²)")
plt.tight_layout()
plt.savefig("reports/plot2_bmi_boxplot.png", dpi=200)
plt.close()

# نمودار 3: نمودار پراکندگی سن و SBP به همراه خط رگرسیون
plt.figure(figsize=(7, 4.5))
sns.regplot(data=df, x='Age', y='SBP',
            scatter_kws={'alpha': 0.15, 'color': 'gray'},
            line_kws={'color': 'red'})
plt.title("Age vs. Systolic Blood Pressure (SBP)")
plt.xlabel("Age (Years)")
plt.ylabel("SBP (mmHg)")
plt.tight_layout()
plt.savefig("reports/plot3_scatter_age_sbp.png", dpi=200)
plt.close()

# نمودار 4: نمودار ستونی دسته‌های BMI
plt.figure(figsize=(7, 4.5))
sns.countplot(data=df, x='BMI_Cat', hue='Gender', palette='Blues')
plt.title("WHO BMI Categories Count by Gender")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/plot4_bmi_categories.png", dpi=200)
plt.close()

print("\nAll 4 plots were saved successfully in the 'reports' folder.")
