import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import joblib
import matplotlib.pyplot as plt

print("1. Đang tải và xử lý dữ liệu gốc...")
# Đọc dữ liệu gốc để lấy Đơn giá (UnitPrice) và Thời gian (InvoiceDate)
df_raw = pd.read_excel('dataset.xlsx')
df_raw['Description'] = df_raw['Description'].str.strip().str.upper()

# Lọc bỏ dữ liệu lỗi (số lượng/giá <= 0) và bỏ giá trị rỗng [10]
df_clean = df_raw[(df_raw['Quantity'] > 0) & (df_raw['UnitPrice'] > 0)].copy()
df_clean = df_clean.dropna(subset=['InvoiceNo', 'Description', 'InvoiceDate'])

# =====================================================================
# PHẦN 1: TÍNH TOÁN DOANH THU TIỀM NĂNG VÀ VẼ BIỂU ĐỒ LUẬT
# =====================================================================
print("2. Đang tính Doanh thu tiềm năng (Revenue Score)...")
# Lấy đơn giá trung bình cho mỗi sản phẩm [2]
price_dict = df_clean.groupby('Description')['UnitPrice'].mean().to_dict()

# Tải tập luật đã lưu từ Bước 4
rules = joblib.load('recommendation_rules.pkl')

# Tạo tập các sản phẩm gộp từ vế trái và vế phải [11]
rules['bundle_items'] = rules.apply(lambda row: set(row['antecedents']) | set(row['consequents']), axis=1)

# Tính tổng giá combo (bundle_price) và điểm doanh thu (revenue_score) [1, 2]
rules['bundle_price'] = rules['bundle_items'].apply(lambda s: sum(price_dict.get(i, 0) for i in s))
rules['revenue_score'] = rules['support'] * rules['bundle_price']

# Lấy Top 10 luật có doanh thu cao nhất [11]
top10_rev = rules.sort_values(by='revenue_score', ascending=False).head(10).copy()

# Chuẩn hóa dữ liệu bằng cách chia cho giá trị max để vẽ chung trên 1 biểu đồ [3]
top10_rev['support_norm'] = top10_rev['support'] / top10_rev['support'].max()
top10_rev['confidence_norm'] = top10_rev['confidence'] / top10_rev['confidence'].max()
top10_rev['lift_norm'] = top10_rev['lift'] / top10_rev['lift'].max()
top10_rev['bundle_price_norm'] = top10_rev['bundle_price'] / top10_rev['bundle_price'].max()

# Tạo nhãn cho trục Y (A -> B)
labels = top10_rev.apply(lambda x: f"{list(x['antecedents'])} -> {list(x['consequents'])}", axis=1)
top10_rev.index = labels

# Vẽ biểu đồ ngang so sánh 4 chỉ số [4]
ax = top10_rev[['support_norm', 'confidence_norm', 'lift_norm', 'bundle_price_norm']].plot(
    kind='barh', figsize=(14, 8), width=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
)
plt.title('So sánh Support, Confidence, Lift và Bundle_price của Top 10 Luật sinh lời nhất', fontsize=14)
plt.xlabel('Giá trị đã chuẩn hóa (0.0 đến 1.0)')
plt.ylabel('Luật kết hợp')
plt.legend(['Support', 'Confidence', 'Lift', 'Bundle Price'])
plt.gca().invert_yaxis() # Đảo ngược trục Y để top 1 ở trên cùng
plt.tight_layout()
plt.savefig('revenue_rules_chart.png')
print(" -> Đã lưu biểu đồ Top Doanh Thu: revenue_rules_chart.png")

# =====================================================================
# PHẦN 2: VẼ BẢNG ĐIỀU KHIỂN EDA (DASHBOARD)
# =====================================================================
print("3. Đang vẽ biểu đồ Phân tích khám phá (EDA)...")

# Chuyển đổi InvoiceDate sang định dạng datetime [12]
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

fig, axes = plt.subplots(3, 1, figsize=(12, 18))
plt.subplots_adjust(hspace=0.4)

# Biểu đồ 1: Top 10 sản phẩm bán chạy nhất [5]
top_10_items = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
top_10_items.sort_values().plot(kind='barh', ax=axes[0], color='teal')
axes[0].set_title('Top 10 Sản phẩm có số lượng bán ra cao nhất', fontsize=14)
axes[0].set_xlabel('Tổng số lượng bán')

# Biểu đồ 2: Biến động giao dịch theo tháng [7]
monthly_transactions = df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M'))['InvoiceNo'].nunique()
monthly_transactions.plot(kind='line', marker='o', ax=axes[1], color='orange', linewidth=2)
axes[1].set_title('Biến động số lượng giao dịch theo Tháng', fontsize=14)
axes[1].set_ylabel('Số lượng hóa đơn')
axes[1].grid(True, linestyle='--', alpha=0.6)

# Biểu đồ 3: Biến động giao dịch theo giờ trong ngày [8]
hourly_transactions = df_clean.groupby(df_clean['InvoiceDate'].dt.hour)['InvoiceNo'].nunique()
hourly_transactions.plot(kind='line', marker='s', ax=axes[2], color='green', linewidth=2)
axes[2].set_title('Biến động số lượng giao dịch theo Giờ trong ngày', fontsize=14)
axes[2].set_xlabel('Giờ (Khung giờ 24h)')
axes[2].set_ylabel('Số lượng hóa đơn')
axes[2].set_xticks(range(6, 21))
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.savefig('admin_eda_dashboard.png', bbox_inches='tight')
print(" -> Đã lưu biểu đồ EDA Dashboard: admin_eda_dashboard.png")
print("\nHOÀN THÀNH! Bạn có thể đưa 2 file ảnh (.png) này lên giao diện Admin của Web.")