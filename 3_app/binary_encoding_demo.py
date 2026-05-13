import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


def main() -> None:
    print("=== DEMO MA HOA NHI PHAN (0-1) CHO FP-GROWTH ===")
    print("1) Doc du lieu da gom giao dich...")
    df = pd.read_csv("data/transactions_ready_for_fpgrowth.csv")

    if "Items_str" not in df.columns:
        raise ValueError("Khong tim thay cot 'Items_str' trong data/transactions_ready_for_fpgrowth.csv")

    print("2) Chuyen Items_str thanh list san pham...")
    transactions = df["Items_str"].dropna().astype(str).apply(lambda x: x.split("|")).tolist()
    print(f"   -> So giao dich hop le: {len(transactions):,}")

    print("3) Ma hoa one-hot (0-1) bang TransactionEncoder...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_).astype(int)

    print("\n=== THONG TIN MA TRAN SAU KHI MA HOA ===")
    print(f"So hang (giao dich): {df_encoded.shape[0]:,}")
    print(f"So cot (san pham):   {df_encoded.shape[1]:,}")
    print(f"Kich thuoc ma tran:   {df_encoded.shape}")
    print(f"Kieu du lieu:         {df_encoded.dtypes.iloc[0]}")

    print("\n=== MAU 5 DONG X 10 COT DAU TIEN ===")
    sample_cols = list(df_encoded.columns[:10])
    print(df_encoded[sample_cols].head(5).to_string(index=False))

    output_file = "data/binary_matrix_sample.csv"
    df_encoded.head(200).to_csv(output_file, index=False)
    print(f"\n4) Da luu mau 200 dong dau vao file: {output_file}")
    print("=== HOAN THANH DEMO ===")


if __name__ == "__main__":
    main()
