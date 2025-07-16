import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# === STEP 1: Extract ZIP ===
zip_path = r"C:\Users\RISHITA\Downloads\archive (5).zip"  # Your ZIP file
extract_dir = "titanic_dataset"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("[✅] ZIP extracted!")

# === STEP 2: Load CSV (UPDATE THIS LINE with actual name & path inside zip) ===
csv_file = os.path.join(extract_dir, "train.csv")  # Update if needed
df = pd.read_csv(csv_file)

# === STEP 3: Clean Data ===
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# === STEP 4: Create PDF ===
pdf_path = "titanic_eda_output.pdf"
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Survived')
    plt.title('Survival Count')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Sex', hue='Survived')
    plt.title('Survival by Gender')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Pclass', hue='Survived')
    plt.title('Survival by Passenger Class')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='Age', bins=20, kde=True)
    plt.title('Age Distribution')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Pclass', y='Age')
    plt.title('Age vs Passenger Class')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    pdf.savefig()
    plt.close()


