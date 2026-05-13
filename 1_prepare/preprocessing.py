import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import time

file_name = 'dataset.xlsx'

print("=== BƯỚC 1: ĐỌC DỮ LIỆU ===")
start_time = time.time()
df = pd.read_excel(file_name)
df_raw = df.copy()
print(f"-> Đã đọc xong {df.shape[0]:,} dòng trong {time.time() - start_time:.2f} giây.\n")

print("=== BƯỚC 2: LÀM SẠCH DỮ LIỆU CƠ BẢN ===")
# Chuẩn hóa chữ hoa và xóa khoảng trắng thừa
df['Description'] = df['Description'].str.strip().str.upper() 
# Bỏ dòng thiếu dữ liệu ở mã hóa đơn hoặc tên sản phẩm
df = df.dropna(subset=['InvoiceNo', 'Description'])
# Loại bỏ các hóa đơn bị hủy (bắt đầu bằng 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
# Chỉ lấy các giao dịch có số lượng và giá tiền hợp lệ
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

print("=== BƯỚC 3: LỌC SẢN PHẨM KHÔNG HỢP LỆ ===")
# Loại bỏ các mã phí, dịch vụ không phải hàng hóa vật lý
junk_keywords = ['POSTAGE', 'DOTCOM POSTAGE', 'CRUK COMMISSION',
                 'MANUAL', 'DISCOUNT', 'BANK CHARGES',
                 'SAMPLES', 'AMAZON FEE']
df = df[~df['Description'].isin(junk_keywords)]
# Loại bỏ các sản phẩm có tên quá ngắn (thường là rác dữ liệu)
df = df[df['Description'].str.len() >= 4]

print("=== BƯỚC 4: LỌC ĐỘ DÀI GIỎ HÀNG (2 <= Items <= 50) ===")
# Bỏ các dòng trùng lặp cùng 1 sản phẩm trong cùng 1 hóa đơn
df = df.drop_duplicates(subset=['InvoiceNo', 'Description'])
# Đếm số lượng sản phẩm mỗi hóa đơn
item_counts = df.groupby('InvoiceNo')['Description'].count()
# Hóa đơn phải có từ 2 đến 50 mặt hàng
valid_invoices = item_counts[(item_counts >= 2) & (item_counts <= 50)].index
df_filtered = df[df['InvoiceNo'].isin(valid_invoices)]

print("=== BƯỚC 5: CHUYỂN ĐỔI SANG TRANSACTION LIST ===")
# Gom nhóm sản phẩm theo mã hóa đơn
transactions = df_filtered.groupby('InvoiceNo').agg({
    'Description': lambda x: list(set(x)),
    'InvoiceDate': 'first'
}).reset_index()

transactions.columns = ['TID', 'Items', 'Date']
transactions = transactions.sort_values(by='Date')

# Tính độ dài thực tế để xuất báo cáo
transactions['item_count'] = transactions['Items'].apply(len)

# Lưu Items dạng string phân cách bởi dấu '|' để sau này đọc bằng Pandas dễ dàng
transactions['Items_str'] = transactions['Items'].apply(lambda x: '|'.join(x))

print("=== BƯỚC 6: TỔNG KẾT & XUẤT FILE ===")
print(f"Tổng số dòng ban đầu: {df_raw.shape[0]:,} dòng")
print(f"Số dòng sau khi lọc:  {df_filtered.shape[0]:,} dòng")
print(f"Tổng số hóa đơn hợp lệ: {len(transactions):,}")
print(f"Tổng số sản phẩm (Unique): {df_filtered['Description'].nunique():,}")
print(f"Trung bình Items/Hóa đơn: {transactions['item_count'].mean():.1f}")
print(f"Hóa đơn lớn nhất: {transactions['item_count'].max()} sản phẩm")

output_file = 'data/transactions_ready_for_fpgrowth.csv'
transactions.to_csv(output_file, index=False)
print(f"\n✅ HOÀN THÀNH! Đã lưu dữ liệu sạch vào file: {output_file}")
print("=== DỮ LIỆU ĐÃ SẴN SÀNG ĐỂ CHẠY THUẬT TOÁN FP-GROWTH ===")