import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd

print("1. Đang đọc dữ liệu sạch...")
df = pd.read_csv('data/transactions_ready_for_fpgrowth.csv')

print("2. Đang tiến hành chia tập Train / Test ngẫu nhiên (Tỷ lệ 80/20)...")
if 'Date' not in df.columns:
    raise ValueError("Không tìm thấy cột 'Date' trong data/transactions_ready_for_fpgrowth.csv")

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).reset_index(drop=True)

# Random split: 80% train, 20% test (cố định random_state để tái lập kết quả)
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

print("3. Đang lưu ra các file riêng biệt...")
train_df.to_csv('data/train_transactions.csv', index=False)
test_df.to_csv('data/test_transactions.csv', index=False)

print("\n✅ HOÀN THÀNH CHIA DỮ LIỆU!")
print(f"Tổng số hóa đơn: {len(df):,}")
print(f"-> Tập Train (Huấn luyện): {len(train_df):,} hóa đơn")
print(f"-> Tập Test (Kiểm thử): {len(test_df):,} hóa đơn")
print("\nRandom split sizes:", len(train_df), len(test_df))
print("Random test months:\n", test_df['Date'].dt.to_period('M').value_counts().sort_index())