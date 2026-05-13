import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd


def main() -> None:
    print("=== DEMO GOM NHOM GIAO DICH ===")
    print("1) Doc du lieu goc...")
    df = pd.read_excel("dataset.xlsx")

    # Lam sach co ban
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["InvoiceNo", "Description"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Bo trung 1 san pham trong cung 1 hoa don
    df = df.drop_duplicates(subset=["InvoiceNo", "Description"])

    # Loc hoa don hop le: 2 <= so mon <= 50
    item_counts = df.groupby("InvoiceNo")["Description"].count()
    valid_invoices = item_counts[(item_counts >= 2) & (item_counts <= 50)].index
    df_filtered = df[df["InvoiceNo"].isin(valid_invoices)].copy()

    print(f"So hoa don hop le: {len(valid_invoices):,}")
    print(f"So dong sau loc: {len(df_filtered):,}")

    print("\n2) Gom nhom theo hoa don (groupby InvoiceNo)...")
    transactions = (
        df_filtered.groupby("InvoiceNo")
        .agg(
            {
                "Description": lambda x: list(x),  # Hoac: '|'.join(x)
                "InvoiceDate": "first",
            }
        )
        .reset_index()
    )
    transactions.columns = ["TID", "Items", "Date"]

    print("\n=== MAU DU LIEU SAU KHI GOM NHOM ===")
    for _, row in transactions.head(10).iterrows():
        print(f"TID: {row['TID']} | Date: {row['Date']} | Items: {row['Items']}")

    # Neu ban muon luu kieu chuoi de chup them
    transactions["Items_str"] = transactions["Items"].apply(lambda x: "|".join(x))
    transactions.to_csv("data/transactions_grouped_demo.csv", index=False)
    print("\nDa luu file demo: data/transactions_grouped_demo.csv")


if __name__ == "__main__":
    main()
