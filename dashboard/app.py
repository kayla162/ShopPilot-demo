"""ShopPilot Dashboard。在專案根目錄執行：streamlit run dashboard/app.py"""
from __future__ import annotations

import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard.views import (  # noqa: E402
    common, competitor, exposure_view, external, header, keywords_view, overview, price_sales, products_view, qa_view,
)
from shoppilot import config, db, pipeline, shops  # noqa: E402
from shoppilot.analytics.products import selection_order  # noqa: E402
from shoppilot.analytics.report import Analysis, default_period, load_dataset  # noqa: E402
from shoppilot.llm import local_provider  # noqa: E402
from shoppilot.llm.explainer import AIStatus, ai_status, make_provider, with_env_provider  # noqa: E402
from shoppilot.llm.local_provider import TransformersProvider  # noqa: E402

TABS = [("總覽", overview.render), ("競品分析", competitor.render), ("價格與銷售", price_sales.render),
        ("曝光因子", exposure_view.render), ("標題關鍵字", keywords_view.render),
        ("商品總表", products_view.render), ("外部競品", external.render), ("AI 問答", qa_view.render)]

# 數值太長時換行，不要被截成「…」
# Streamlit 1.66 把數值放在 <div><p> 裡，<p> 的 nowrap 來自比這裡更具體的樣式，所以連同裡面所有元素一起改、
# 加上 !important，不然長的值（例如「第 4 便宜／共 10 個」）會蓋到隔壁欄
METRIC_WRAP_CSS = ('<style>[data-testid="stMetricValue"], [data-testid="stMetricValue"] * '
                   '{white-space: normal !important; overflow: visible !important; text-overflow: clip !important; '
                   'overflow-wrap: anywhere;}</style>')
# 展示模式：本機模型在啟動時就在背景載入（preload_model），不是按下按鈕才載入
DEMO_LOCAL_LABEL = "可使用（本機模型，已在背景預先載入；剛啟動時第一次回答可能要等載入完成）"

st.set_page_config(page_title="ShopPilot 商品競品分析", page_icon="📊", layout="wide")
config.load_dotenv()


@st.cache_resource(show_spinner="分析中…", max_entries=4)
def get_analysis(db_file: str, shop: str, version: tuple, mtimes: tuple, period: tuple | None) -> Analysis:
    """資料庫檔案、賣場、資料庫版本、設定檔修改時間、期間都一樣時，直接用上次的分析結果。
    db_file 也放進快取鍵：資料資料夾（SHOPPILOT_DATA_DIR）不同時，同名賣場是不同的資料庫。"""
    with closing(db.connect(db_file)) as conn:
        dataset = load_dataset(conn, period)
    return Analysis(dataset, shops.load_config(shop, "keywords"), shops.load_config(shop, "rules"))


def inbox_label(shop: str) -> str:
    """畫面上提示報表要放哪裡：相對於專案根目錄的路徑（在別的位置時顯示完整路徑）。"""
    folder = shops.inbox_dir(shop)
    try:
        return folder.relative_to(ROOT).as_posix() + "/"
    except ValueError:
        return str(folder)


def choose_shop() -> str:
    """側欄最上方選賣場；新增賣場後下一次執行才切過去（widget 建立後不能改它的值）。"""
    names = shops.list_shops()
    pending = st.session_state.pop("_switch_shop", None)
    if pending in names:
        st.session_state["shop"] = pending
        st.session_state.pop("new_shop_name", None)      # 建立成功：清空輸入框，避免再按一次「建立」
    if st.session_state.get("shop") not in names:
        st.session_state.pop("shop", None)
    shop = st.sidebar.selectbox("賣場", names, key="shop") if len(names) > 1 else names[0]
    if not common.demo_mode():
        with st.sidebar.expander("新增賣場"):
            name = st.text_input("賣場名稱", key="new_shop_name", max_chars=shops.MAX_NAME_LEN)
            copy = st.checkbox("複製一份設定檔給這家店", value=True, key="new_shop_copy",
                               help="複製 keywords.yaml 和 rules.yaml 到這家店的 config/，之後可以各自修改（例如賣不同品類）；"
                                    "不複製就和預設賣場共用。")
            template = st.selectbox("設定檔來源", list(shops.CONFIG_TEMPLATES), key="new_shop_template",
                                    disabled=not copy, help="賣保養品選「保養品範例」：類別、品牌、描述建議都已經調好。")
            if st.button("建立", key="new_shop_create"):
                try:
                    folder = shops.create_shop(name, copy_config=copy, template=template)
                except (ValueError, OSError) as e:
                    st.error(common.md_escape(str(e)))
                else:
                    st.session_state["_switch_shop"] = folder.name
                    st.rerun()
    return shop


def config_caption(shop: str) -> None:
    if shop == shops.DEFAULT_SHOP:
        return
    parts = [f"{k}.yaml {'這家店自己的' if shops.has_own_config(shop, k) else '共用'}" for k in shops.SHOP_CONFIGS]
    st.sidebar.caption(f"設定檔：{'、'.join(parts)}")


def auto_import(shop: str) -> None:
    """每個工作階段第一次打開某個賣場時，匯入它 inbox/ 的新報表（hash 相同的會略過）。"""
    if common.demo_mode():       # 示範資料庫已經建好；inbox 不會有新報表
        return
    done = st.session_state.setdefault("_auto_imported", set())
    if shop in done:
        return
    with closing(db.connect(shops.db_path(shop))) as conn:
        st.session_state.setdefault("_import_results", {})[shop] = pipeline.import_folder(conn, shops.inbox_dir(shop))
    done.add(shop)


def upload_reports(shop: str) -> None:
    if common.demo_mode():
        st.sidebar.caption(common.DEMO_UPLOAD_NOTE)
        return
    files = st.sidebar.file_uploader("上傳新報表（選用）", type=["xlsx"], accept_multiple_files=True,
                                     help="商品基本資料、價格與庫存、商品表現（parentskudetail）", key=f"upload::{shop}")
    done = st.session_state.setdefault("_uploaded_ids", set())
    new = [f for f in files or [] if (shop, f.file_id) not in done]
    if not new:
        return
    inbox = shops.inbox_dir(shop)
    inbox.mkdir(parents=True, exist_ok=True)
    results = []
    with closing(db.connect(shops.db_path(shop))) as conn:
        for f in new:
            name, data = Path(f.name).name, f.getvalue()
            (inbox / name).write_bytes(data)
            results.append(pipeline.import_file(conn, name, data))
            done.add((shop, f.file_id))
    st.session_state.setdefault("_import_results", {})[shop] = results
    st.rerun()


def demo_banner() -> None:
    if not common.demo_mode():
        return
    link = common.repo_url()
    st.sidebar.info(common.DEMO_BANNER)
    if link:
        st.sidebar.markdown(f"[原始碼與說明（GitHub）]({link})")


@st.cache_resource(show_spinner=False)
def preload_model(model: str) -> bool:
    """展示模式：每個程序只在背景載入一次本機模型（cache_resource 跨工作階段共用）。"""
    llm_cfg, status = ai_settings()
    provider = make_provider(llm_cfg) if status.available else None
    if isinstance(provider, TransformersProvider):
        local_provider.preload_in_background(provider)
        return True
    return False


def ai_settings() -> tuple[dict, AIStatus]:
    """讀取 AI 設定並判斷狀態；設定檔壞掉時只停用 AI，其他功能照常。"""
    try:
        cfg = with_env_provider(config.load_yaml("llm"))     # 線上 demo 用環境變數切成只讀快取
        return cfg, ai_status(cfg)
    except Exception:  # noqa: BLE001
        return {}, AIStatus(False, "目前未啟用", "AI 設定檔（config/llm.yaml）無法讀取")


def status_block(analysis: Analysis, status) -> None:
    ok = analysis.data_status
    missing = [common.REPORT_NAMES[k] for k in ("perf", "stock") if ok[k] != "ok"]
    core = "正常" if not missing else f"資料不足（缺少{'、'.join(missing)}）"
    compare = "正常" if ok["stock"] == "ok" else "資料不足（缺少價格與庫存）"
    st.sidebar.markdown(f"**系統狀態**  \n核心商品分析：{core}  \n競品比較：{compare}  \nDashboard：正常")
    st.sidebar.markdown(f"AI 文字解釋：  \n{status.label}")
    if status.reason:
        st.sidebar.caption(status.reason)


def import_log(shop: str) -> None:
    results = [r for r in (st.session_state.get("_import_results") or {}).get(shop, [])
               if r.status in ("imported", "failed")]
    if not results:
        return
    with st.sidebar.expander(f"匯入結果（{len(results)}）"):
        for r in results:
            st.caption(f"{pipeline.STATUS_LABELS[r.status]}：{common.md_escape(r.file_name)}"
                       + (f"（{common.md_escape(r.message)}）" if r.message else ""))


def choose_product(analysis: Analysis, shop: str = shops.DEFAULT_SHOP) -> str | None:
    df = analysis.products
    if df.empty:
        return None
    ids = selection_order(df)
    names = {r.product_id: common.label(r.code, r.core_title) + ("（特殊商品）" if r.is_special else "")
             for r in df.itertuples()}
    # 每個賣場用自己的 key：共用一個 key 的話，切換賣場後選單還會顯示上一家店的商品名稱
    key = "product_id" if shop == shops.DEFAULT_SHOP else f"product_id::{shop}"
    if st.session_state.get(key) not in ids:
        st.session_state.pop(key, None)
    return st.sidebar.selectbox("選擇商品", ids, format_func=names.get, key=key)


def main() -> None:
    st.sidebar.title("ShopPilot")
    demo_banner()
    # 設定檔寫錯（YAML 格式、正規表示式）或資料有問題時，只顯示說明，整頁不會變成錯誤訊息；
    # st.rerun()／st.stop() 是 BaseException，不會被這裡攔下
    shop = st.session_state.get("shop", shops.DEFAULT_SHOP)
    try:
        shop = choose_shop()
        config_caption(shop)
        auto_import(shop)
        with closing(db.connect(shops.db_path(shop))) as conn:
            version = db.db_version(conn)
            periods = db.list_periods(conn)
        period = None
        if periods:
            period = st.sidebar.selectbox("資料期間", periods, index=periods.index(default_period(periods)),
                                          format_func=lambda p: f"{p[0]} ～ {p[1]}")
        analysis = get_analysis(str(shops.db_path(shop)), shop, version, shops.config_mtimes(shop), period)
    except Exception as e:  # noqa: BLE001
        st.error("分析資料無法建立。如果剛修改過設定檔（keywords.yaml 或 rules.yaml），請檢查格式（縮排、冒號、正規表示式）後重新整理頁面。（"
                 + common.md_escape(f"{type(e).__name__}: {e}") + "）")
        import_log(shop)
        st.stop()
    if analysis.ds.stock_snapshot:
        st.sidebar.caption(f"價格庫存快照：{analysis.ds.stock_snapshot}（不是期間當時的庫存）")
    product_id = choose_product(analysis, shop)
    llm_cfg, status = ai_settings()
    if common.demo_mode() and status.available and llm_cfg.get("provider") == "local":
        if preload_model(str((llm_cfg.get("local") or {}).get("model", ""))):
            # 展示模式啟動時已在背景載入模型，不是「按下才載入」
            status = AIStatus(True, DEMO_LOCAL_LABEL, status.reason)
    st.sidebar.divider()
    status_block(analysis, status)
    st.sidebar.divider()
    upload_reports(shop)
    import_log(shop)

    theme = getattr(st.context, "theme", None)
    ctx = common.Ctx(analysis=analysis, product_id=product_id, status=status, llm_cfg=llm_cfg,
                     db_path=shops.db_path(shop), inbox_label=inbox_label(shop),
                     dark=getattr(theme, "type", None) == "dark")
    st.title("商品競品與商品表現分析")
    if analysis.ds.period:
        st.caption(f"資料期間：{analysis.ds.period[0]} ～ {analysis.ds.period[1]}｜"
                   "所有數字都由程式計算；AI 只在按下按鈕時用來寫說明")
    st.markdown(METRIC_WRAP_CSS, unsafe_allow_html=True)
    common.section("KPI 與重點建議", header.render, ctx)
    tabs = st.tabs([name for name, _ in TABS])
    for tab, (name, render) in zip(tabs, TABS):
        with tab:
            common.section(name, render, ctx)


main()
