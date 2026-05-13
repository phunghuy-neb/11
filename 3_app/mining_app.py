import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from rule_policy import filter_rules_recommend_format


BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)
MODEL_FILE = BASE_DIR / "recommendation_rules.pkl"
TRAIN_FILE = DATA_DIR / "train_transactions.csv"
NEW_DATA_FILE = DATA_DIR / "new_transactions.csv"
DATASET_FILE = BASE_DIR / "dataset.xlsx"

MIN_SUPPORT = 0.01
MIN_LIFT = 1.2
AUTO_RETRAIN_THRESHOLD = 100

PRODUCTS = [
    {
        "name": "REGENCY CAKESTAND 3 TIER",
        "price": 14.95,
    },
    {
        "name": "GREEN REGENCY TEACUP AND SAUCER",
        "price": 3.95,
    },
    {
        "name": "ROSES REGENCY TEACUP AND SAUCER",
        "price": 4.25,
    },
    {
        "name": "PAPER CHAIN KIT 50'S CHRISTMAS",
        "price": 2.79,
    },
    {
        "name": "JUMBO BAG RED RETROSPOT",
        "price": 2.45,
    },
    {
        "name": "PARTY BUNTING",
        "price": 4.65,
    },
]


def ensure_new_data_file() -> None:
    if not NEW_DATA_FILE.exists():
        pd.DataFrame(columns=["TID", "Items", "Date", "item_count", "Items_str"]).to_csv(
            NEW_DATA_FILE, index=False
        )


def rules_available() -> bool:
    return MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 0


def load_rules() -> pd.DataFrame:
    if not rules_available():
        return pd.DataFrame()
    rules = joblib.load(MODEL_FILE)
    if not isinstance(rules, pd.DataFrame):
        return pd.DataFrame()
    return rules


def get_recommendations(cart: List[str], rules_df: pd.DataFrame) -> List[str]:
    if rules_df.empty:
        return []
    cart_set = set(x.strip().upper() for x in cart)
    recs: Set[str] = set()
    for _, row in rules_df.iterrows():
        ant = set(str(x).strip().upper() for x in row["antecedents"])
        con = [str(x).strip().upper() for x in row["consequents"]]
        if ant.issubset(cart_set):
            for item in con:
                if item not in cart_set:
                    recs.add(item)
    return sorted(recs)


def get_known_items_from_rules(rules_df: pd.DataFrame) -> List[str]:
    if rules_df.empty:
        return []
    items: Set[str] = set()
    for _, row in rules_df.iterrows():
        for x in row["antecedents"]:
            items.add(str(x).strip().upper())
        for x in row["consequents"]:
            items.add(str(x).strip().upper())
    return sorted(items)


def format_rule(antecedents, consequents) -> str:
    left = ", ".join(sorted(str(i).strip() for i in antecedents))
    right = ", ".join(sorted(str(i).strip() for i in consequents))
    return f"{left}  →  {right}"


def get_unit_price(name: str, price_dict: Dict[str, float]) -> float:
    if name in price_dict and price_dict[name] > 0:
        return float(price_dict[name])
    for p in PRODUCTS:
        if p["name"] == name:
            return float(p.get("price", 0) or 0)
    return 0.0


def append_new_transaction(items: List[str]) -> bool:
    cleaned = sorted(set(x.strip().upper() for x in items if x.strip()))
    # Đơn chỉ có 1 sản phẩm không sinh được luật → bỏ qua không lưu vào tập train
    if len(cleaned) < 2:
        return False

    ensure_new_data_file()
    df = pd.read_csv(NEW_DATA_FILE)
    now = datetime.now()
    row = {
        "TID": f"NEW_{now.strftime('%Y%m%d%H%M%S%f')}",
        "Items": str(cleaned),
        "Date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "item_count": len(cleaned),
        "Items_str": "|".join(cleaned),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(NEW_DATA_FILE, index=False)
    return True


def retrain_model() -> Dict[str, str]:
    if not TRAIN_FILE.exists():
        return {"ok": False, "message": "Không tìm thấy train_transactions.csv"}

    base_df = pd.read_csv(TRAIN_FILE)[["Items_str"]].dropna()
    ensure_new_data_file()
    new_df = pd.read_csv(NEW_DATA_FILE)
    if "Items_str" in new_df.columns:
        new_df = new_df[["Items_str"]].dropna()
    else:
        new_df = pd.DataFrame(columns=["Items_str"])

    merged = pd.concat([base_df, new_df], ignore_index=True)
    merged["Items_str"] = merged["Items_str"].astype(str).str.strip()
    merged = merged[merged["Items_str"] != ""].drop_duplicates().reset_index(drop=True)

    transactions = merged["Items_str"].apply(lambda x: x.split("|")).tolist()
    if len(transactions) < 10:
        return {"ok": False, "message": "Dữ liệu huấn luyện quá ít sau khi gộp."}

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(encoded_df, min_support=MIN_SUPPORT, use_colnames=True)
    if frequent_itemsets.empty:
        return {"ok": False, "message": "Không tạo được frequent itemsets. Hãy giảm min_support."}

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=MIN_LIFT)
    if rules.empty:
        return {"ok": False, "message": "Không tạo được luật phù hợp. Hãy giảm min_lift."}

    rules = filter_rules_recommend_format(rules)
    if rules.empty:
        return {"ok": False, "message": "Sau khi lọc dạng luật không còn rule phù hợp."}

    rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
    rules["antecedents"] = rules["antecedents"].apply(list)
    rules["consequents"] = rules["consequents"].apply(list)
    joblib.dump(rules, MODEL_FILE)
    merged.to_csv(DATA_DIR / "train_transactions_latest.csv", index=False)

    # Backup TRAIN_FILE cũ trước khi đè, để có thể rollback nếu cần
    if TRAIN_FILE.exists():
        backup_name = f"train_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy2(TRAIN_FILE, BACKUP_DIR / backup_name)

    # Hấp thụ giao dịch mới vào tập train, reset hàng đợi để counter quay về 0
    merged.to_csv(TRAIN_FILE, index=False)
    pd.DataFrame(columns=["TID", "Items", "Date", "item_count", "Items_str"]).to_csv(
        NEW_DATA_FILE, index=False
    )

    return {
        "ok": True,
        "message": f"Huấn luyện lại thành công. Tổng giao dịch: {len(merged):,}, tổng luật: {len(rules):,}",
    }


def count_new_transactions() -> int:
    if not NEW_DATA_FILE.exists():
        return 0
    try:
        return len(pd.read_csv(NEW_DATA_FILE))
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def load_price_dict() -> Dict[str, float]:
    if not DATASET_FILE.exists():
        return {}
    try:
        df_raw = pd.read_excel(DATASET_FILE)
    except ImportError:
        # openpyxl chưa có: vẫn cho app chạy, Revenue Score tạm = 0.
        return {}
    except Exception:
        return {}
    df_raw["Description"] = df_raw["Description"].astype(str).str.strip().str.upper()
    df_raw = df_raw[(df_raw["UnitPrice"] > 0) & df_raw["Description"].notna()]
    return df_raw.groupby("Description")["UnitPrice"].mean().to_dict()


def rules_manager_table(rules_df: pd.DataFrame) -> pd.DataFrame:
    if rules_df.empty:
        return pd.DataFrame()
    price_dict = load_price_dict()
    view = rules_df.copy()
    view["X"] = view["antecedents"].apply(lambda x: ", ".join(sorted(str(i) for i in x)))
    view["Y"] = view["consequents"].apply(lambda x: ", ".join(sorted(str(i) for i in x)))
    view["bundle_items"] = view.apply(
        lambda row: set(str(i) for i in row["antecedents"]) | set(str(i) for i in row["consequents"]),
        axis=1,
    )
    view["Revenue Score"] = view.apply(
        lambda row: row["support"] * sum(price_dict.get(i, 0.0) for i in row["bundle_items"]), axis=1
    )
    view = view.rename(
        columns={"support": "Support", "confidence": "Confidence", "lift": "Lift"}
    )
    cols = ["X", "Y", "Support", "Confidence", "Lift", "Revenue Score"]
    return view[cols].sort_values(by="Revenue Score", ascending=False).reset_index(drop=True)


def top_products_from_transactions(limit: int = 10) -> pd.DataFrame:
    if not TRAIN_FILE.exists():
        return pd.DataFrame(columns=["Sản phẩm", "Số lần xuất hiện"])
    df = pd.read_csv(TRAIN_FILE)
    if "Items_str" not in df.columns:
        return pd.DataFrame(columns=["Sản phẩm", "Số lần xuất hiện"])
    exploded = (
        df["Items_str"]
        .dropna()
        .astype(str)
        .str.split("|")
        .explode()
        .dropna()
        .str.strip()
    )
    vc = exploded.value_counts().head(limit)
    return pd.DataFrame({"Sản phẩm": vc.index, "Số lần xuất hiện": vc.values})


def render_storefront(rules_df: pd.DataFrame) -> None:
    price_dict = load_price_dict()

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown("## 🛍️ KKPDL Gift Store")
        st.caption("Cửa hàng demo gợi ý sản phẩm bằng FP-Growth từ lịch sử mua hàng.")
    with header_right:
        st.metric("🛒 Giỏ hàng", f"{len(st.session_state['cart'])} món")

    st.divider()

    keyword = st.text_input(
        "Tìm sản phẩm",
        placeholder="🔍 Nhập tên sản phẩm…",
        label_visibility="collapsed",
    )
    keyword = keyword.strip().upper() if keyword else ""

    catalog: List[Dict] = [dict(p) for p in PRODUCTS]
    known_items = get_known_items_from_rules(rules_df)
    demo_names = {p["name"] for p in PRODUCTS}
    for item in known_items:
        if item not in demo_names:
            catalog.append({"name": item, "price": float(price_dict.get(item, 0.0))})
    for p in catalog:
        if (p.get("price") or 0) == 0 and p["name"] in price_dict:
            p["price"] = float(price_dict[p["name"]])

    if keyword:
        terms = [t for t in keyword.split() if t]
        products = [p for p in catalog if all(t in p["name"] for t in terms)]
    else:
        products = catalog[:6]

    if keyword and not products:
        st.warning("Không tìm thấy sản phẩm khớp từ khóa. Hãy thử từ khóa khác.")
        return
    if keyword and len(products) > 30:
        st.info(f"Tìm thấy {len(products)} sản phẩm — đang hiển thị 30 sản phẩm đầu.")
        products = products[:30]

    st.markdown(f"##### Sản phẩm ({len(products)})")
    cols = st.columns(3)
    for idx, product in enumerate(products):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"**{product['name'].title()}**")
                price = float(product.get("price") or 0)
                if price > 0:
                    st.markdown(
                        f"<div style='color:#059669;font-weight:700;font-size:1.05rem;"
                        "margin:4px 0 10px;'>£{:,.2f}</div>".format(price),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div style='color:#9ca3af;font-size:0.85rem;margin:4px 0 10px;'>"
                        "Liên hệ để biết giá</div>",
                        unsafe_allow_html=True,
                    )
                if st.button(
                    "Thêm vào giỏ",
                    key=f"add_{product['name']}",
                    use_container_width=True,
                ):
                    st.session_state["cart"].append(product["name"])
                    st.session_state["latest_recs"] = get_recommendations(
                        st.session_state["cart"], rules_df
                    )
                    st.toast(f"Đã thêm {product['name'].title()}", icon="🛒")
                    st.rerun()

    if st.session_state["latest_recs"]:
        st.divider()
        st.markdown("##### 💡 Có thể bạn cũng muốn mua")
        recs = st.session_state["latest_recs"][:4]
        rec_cols = st.columns(len(recs))
        for i, rec in enumerate(recs):
            with rec_cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{rec.title()}**")
                    rec_price = get_unit_price(rec, price_dict)
                    if rec_price > 0:
                        st.markdown(
                            f"<div style='color:#059669;font-weight:700;'>£{rec_price:,.2f}</div>",
                            unsafe_allow_html=True,
                        )
                    if st.button(
                        "Thêm vào giỏ",
                        key=f"rec_add_{rec}",
                        use_container_width=True,
                    ):
                        st.session_state["cart"].append(rec)
                        st.session_state["latest_recs"] = get_recommendations(
                            st.session_state["cart"], rules_df
                        )
                        st.toast(f"Đã thêm {rec.title()}", icon="🛒")
                        st.rerun()

    st.divider()
    st.markdown("##### 🛒 Giỏ hàng hiện tại")

    if not st.session_state["cart"]:
        st.info("Giỏ hàng đang trống. Hãy chọn sản phẩm phía trên.")
        return

    cart_counts = pd.Series(st.session_state["cart"]).value_counts()
    rows = []
    total = 0.0
    for name, qty in cart_counts.items():
        unit = get_unit_price(name, price_dict)
        subtotal = unit * qty
        total += subtotal
        rows.append(
            {
                "Sản phẩm": name.title(),
                "Số lượng": int(qty),
                "Đơn giá": f"£{unit:,.2f}" if unit > 0 else "—",
                "Thành tiền": f"£{subtotal:,.2f}" if unit > 0 else "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown(
        f"<div style='text-align:right;font-size:1.1rem;margin:8px 0;'>"
        f"Tổng cộng: <span style='color:#dc2626;font-weight:700;font-size:1.3rem;'>"
        f"£{total:,.2f}</span></div>",
        unsafe_allow_html=True,
    )

    bt1, bt2 = st.columns([2, 1])
    with bt1:
        if st.button(
            "✅ Thanh toán (Checkout)",
            type="primary",
            use_container_width=True,
        ):
            saved = append_new_transaction(st.session_state["cart"])
            if saved:
                st.success(
                    f"Đã thanh toán thành công! Tổng: £{total:,.2f}. "
                    "Giao dịch đã được lưu để hệ thống học liên tục."
                )
            else:
                st.success(
                    f"Đã thanh toán thành công! Tổng: £{total:,.2f}. "
                    "(Đơn 1 sản phẩm không dùng để học mô hình.)"
                )
            st.session_state["cart"] = []
            st.session_state["latest_recs"] = []

            fresh_count = count_new_transactions()
            if fresh_count >= AUTO_RETRAIN_THRESHOLD:
                with st.spinner(
                    f"Đã đủ {fresh_count} giao dịch mới — đang tự động huấn luyện lại..."
                ):
                    result = retrain_model()
                if result["ok"]:
                    st.success(f"🤖 Auto-retrain: {result['message']}")
                else:
                    st.warning(f"Auto-retrain thất bại: {result['message']}")
            st.rerun()
    with bt2:
        if st.button("🗑️ Xóa giỏ hàng", use_container_width=True):
            st.session_state["cart"] = []
            st.session_state["latest_recs"] = []
            st.rerun()


def render_admin_dashboard(rules_df: pd.DataFrame) -> None:
    st.markdown("## ⚙️ Admin Dashboard")
    st.caption("Quản trị kinh doanh, luật gợi ý và chu trình học liên tục.")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Tổng quan Kinh doanh", "📋 Quản lý Luật Gợi ý", "🔁 Học liên tục"]
    )

    with tab1:
        train_count = len(pd.read_csv(TRAIN_FILE)) if TRAIN_FILE.exists() else 0
        new_count = len(pd.read_csv(NEW_DATA_FILE)) if NEW_DATA_FILE.exists() else 0
        rule_count = len(rules_df)
        known_count = len(get_known_items_from_rules(rules_df))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giao dịch train", f"{train_count:,}")
        c2.metric(
            "Giao dịch mới",
            f"{new_count:,}",
            delta=f"+{new_count}" if new_count > 0 else None,
        )
        c3.metric("Số luật hiện có", f"{rule_count:,}")
        c4.metric("Sản phẩm có trong luật", f"{known_count:,}")

        st.divider()

        left, right = st.columns([1, 1])
        with left:
            st.markdown("##### 🏆 Top 10 sản phẩm bán chạy")
            st.dataframe(
                top_products_from_transactions(10),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.markdown("##### 🔝 Top 10 luật theo Lift")
            if rules_df.empty:
                st.warning("Chưa có model luật để thống kê.")
            else:
                top_rules = (
                    rules_df.copy().sort_values(by="lift", ascending=False).head(10)
                )
                top_rules["Luật gợi ý"] = top_rules.apply(
                    lambda r: format_rule(r["antecedents"], r["consequents"]),
                    axis=1,
                )
                st.dataframe(
                    top_rules[["Luật gợi ý", "support", "confidence", "lift"]].rename(
                        columns={
                            "support": "Support",
                            "confidence": "Confidence",
                            "lift": "Lift",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

    with tab2:
        st.markdown("##### 📋 Bảng quản lý luật gợi ý")
        table = rules_manager_table(rules_df)
        if table.empty:
            st.warning("Chưa có model luật để hiển thị.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption("💡 Bấm vào tiêu đề cột để sắp xếp.")

    with tab3:
        old_count = len(pd.read_csv(TRAIN_FILE)) if TRAIN_FILE.exists() else 0
        fresh_count = count_new_transactions()

        c1, c2, c3 = st.columns(3)
        c1.metric("Giao dịch đã train", f"{old_count:,}")
        c2.metric(
            "Giao dịch mới chờ học",
            f"{fresh_count:,}",
            delta=f"+{fresh_count}" if fresh_count > 0 else None,
        )
        c3.metric("Ngưỡng auto-retrain", f"{AUTO_RETRAIN_THRESHOLD:,}")

        progress_value = min(fresh_count / AUTO_RETRAIN_THRESHOLD, 1.0)
        st.progress(
            progress_value,
            text=f"Tiến độ tới auto-retrain: {fresh_count}/{AUTO_RETRAIN_THRESHOLD}",
        )

        st.info(
            f"Hệ thống sẽ **tự động** gộp dữ liệu và chạy lại FP-Growth khi đạt "
            f"{AUTO_RETRAIN_THRESHOLD} giao dịch mới. Bạn cũng có thể bấm nút bên dưới "
            "để train sớm bất cứ lúc nào."
        )
        if st.button(
            "🔁 Cập nhật & Huấn luyện lại mô hình",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Đang huấn luyện lại mô hình..."):
                result = retrain_model()
            if result["ok"]:
                st.success(result["message"])
                st.balloons()
            else:
                st.error(result["message"])


st.set_page_config(page_title="KKPDL Mining Web", page_icon="🛍️", layout="wide")

if "cart" not in st.session_state:
    st.session_state["cart"] = []
if "latest_recs" not in st.session_state:
    st.session_state["latest_recs"] = []

rules_df = load_rules()

with st.sidebar:
    st.title("🛍️ KKPDL")
    st.caption("Hệ thống gợi ý sản phẩm bằng FP-Growth")
    st.divider()
    page = st.radio(
        "Chọn giao diện",
        ["🏪 Cửa hàng (User)", "⚙️ Bảng điều khiển (Admin)"],
        label_visibility="collapsed",
    )
    st.divider()
    if rules_df.empty:
        st.warning("⚠️ Chưa có model luật")
    else:
        st.success("✅ Model sẵn sàng")
        st.caption(f"Tổng số luật: **{len(rules_df):,}**")

if page.startswith("🏪"):
    render_storefront(rules_df)
else:
    render_admin_dashboard(rules_df)
