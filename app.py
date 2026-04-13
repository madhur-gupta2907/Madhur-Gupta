import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, classification_report, confusion_matrix)

st.set_page_config(page_title="DataLens – Insights", page_icon="🛒",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%);color:#e0e0ff;}
section[data-testid="stSidebar"]{background:rgba(255,255,255,0.04)!important;border-right:1px solid rgba(100,100,255,0.15);}
.metric-card{background:linear-gradient(135deg,rgba(100,100,255,0.12),rgba(50,50,150,0.08));border:1px solid rgba(100,100,255,0.25);border-radius:16px;padding:20px 24px;text-align:center;}
.metric-card h2{font-size:2.2rem;font-weight:700;color:#a78bfa;margin:0;font-family:'JetBrains Mono',monospace;}
.metric-card p{color:#94a3b8;font-size:0.85rem;margin:4px 0 0 0;text-transform:uppercase;letter-spacing:1px;}
.pred-card{background:linear-gradient(135deg,rgba(16,185,129,0.12),rgba(5,150,105,0.06));border:1px solid rgba(16,185,129,0.3);border-radius:16px;padding:20px 24px;text-align:center;}
.pred-card h2{font-size:2rem;font-weight:700;color:#34d399;margin:0;font-family:'JetBrains Mono',monospace;}
.pred-card p{color:#94a3b8;font-size:0.85rem;margin:4px 0 0 0;text-transform:uppercase;letter-spacing:1px;}
.insight-box{background:rgba(167,139,250,0.08);border-left:4px solid #a78bfa;border-radius:0 12px 12px 0;padding:14px 18px;margin:10px 0;font-size:0.93rem;color:#c4b5fd;}
.insight-box strong{color:#e9d5ff;}
.rec-box{background:rgba(16,185,129,0.07);border-left:4px solid #10b981;border-radius:0 12px 12px 0;padding:14px 18px;margin:10px 0;font-size:0.93rem;color:#6ee7b7;}
.rec-box strong{color:#a7f3d0;}
.pred-result-box{background:linear-gradient(135deg,rgba(96,165,250,0.1),rgba(167,139,250,0.08));border:1px solid rgba(96,165,250,0.3);border-radius:16px;padding:24px 28px;text-align:center;margin:16px 0;}
.pred-result-box h1{font-size:3rem;font-weight:700;color:#60a5fa;margin:0;font-family:'JetBrains Mono',monospace;}
.pred-result-box p{color:#94a3b8;font-size:0.9rem;margin:8px 0 0 0;}
.main-header{text-align:center;padding:30px 0 10px 0;}
.main-header h1{font-size:3rem;font-weight:700;background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:6px;}
.main-header p{color:#64748b;font-size:1rem;letter-spacing:2px;text-transform:uppercase;}
.section-title{font-size:1.3rem;font-weight:600;color:#a78bfa;border-bottom:1px solid rgba(167,139,250,0.2);padding-bottom:8px;margin:24px 0 16px 0;}
button[data-baseweb="tab"]{color:#94a3b8!important;font-family:'Space Grotesk',sans-serif!important;}
button[data-baseweb="tab"][aria-selected="true"]{color:#a78bfa!important;border-bottom-color:#a78bfa!important;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-thumb{background:rgba(167,139,250,0.3);border-radius:3px;}
</style>
""", unsafe_allow_html=True)

PALETTE    = ["#a78bfa","#60a5fa","#34d399","#fb923c","#f472b6","#facc15","#38bdf8","#4ade80","#c084fc","#f87171"]
BG_COLOR   = "#0f0f1a"
GRID_COLOR = "#1e1e3a"
TEXT_COLOR = "#94a3b8"

def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in (ax_list or fig.get_axes()):
        ax.set_facecolor(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color("#c4b5fd")
        for spine in ax.spines.values(): spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)

def clean_price(val):
    """Remove ₹ symbol and commas, convert to float."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace('₹', '').replace(',', '').replace('%', '').strip()
    try:
        return float(s)
    except:
        return np.nan

@st.cache_data
def load_and_prepare(filepath=None):
    if filepath is not None:
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except:
            df = pd.read_csv(filepath, encoding="latin1")
    else:
        try:
            df = pd.read_csv("1776058172662_amazon.csv", encoding="utf-8")
        except:
            return None

    # Clean numeric columns
    for col in ['discounted_price', 'actual_price']:
        df[col] = df[col].apply(clean_price)

    for col in ['discount_percentage']:
        df[col] = df[col].apply(clean_price)

    df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace(',', '').str.strip(), errors='coerce')

    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').str.strip()
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

    # Derive main_category from pipe-separated category string
    df['main_category'] = df['category'].astype(str).apply(lambda x: x.split('|')[0] if pd.notna(x) else 'Unknown')

    # Derive sub_category (second level)
    df['sub_category'] = df['category'].astype(str).apply(
        lambda x: x.split('|')[1] if pd.notna(x) and len(x.split('|')) > 1 else 'Unknown'
    )

    # Compute savings
    df['savings'] = df['actual_price'] - df['discounted_price']

    # Rating bucket for classification
    df['rating_bucket'] = pd.cut(
        df['rating'].fillna(0),
        bins=[0, 3.5, 4.0, 4.5, 5.1],
        labels=['Low (<3.5)', 'Average (3.5-4)', 'Good (4-4.5)', 'Excellent (4.5+)']
    ).astype(str)

    return df

def clean_df(df):
    d = df.copy()
    for col in d.select_dtypes(include=np.number).columns:
        d[col].fillna(d[col].median(), inplace=True)
    for col in d.select_dtypes(include="object").columns:
        d[col].fillna(d[col].mode()[0] if len(d[col].mode()) > 0 else "Unknown", inplace=True)
    return d

def encode_df(df):
    d, le_map = df.copy(), {}
    for col in d.select_dtypes(include="object").columns:
        le = LabelEncoder()
        d[col] = le.fit_transform(d[col].astype(str))
        le_map[col] = le
    return d, le_map

def generate_insights(df, num_col, cat_col):
    insights, recs = [], []
    grp = df.groupby(cat_col)[num_col].mean()
    if grp.empty:
        return insights, recs
    top_cat = grp.idxmax(); bottom_cat = grp.idxmin()
    top_val = grp.max(); bottom_val = grp.min()
    mean_val = df[num_col].mean(); std_val = df[num_col].std()
    above = (df[num_col] > mean_val).sum()
    insights.append(f"<strong>{top_cat}</strong> has the highest avg {num_col}: <strong>{top_val:,.2f}</strong>")
    insights.append(f"<strong>{bottom_cat}</strong> has the lowest avg {num_col}: <strong>{bottom_val:,.2f}</strong>")
    insights.append(f"{above} records ({above/len(df)*100:.1f}%) are above overall mean of <strong>{mean_val:,.2f}</strong>")
    insights.append(f"Std deviation = <strong>{std_val:,.2f}</strong> — {'high variability' if std_val > mean_val * 0.3 else 'stable values'}")
    recs.append(f"📈 <strong>Scale {top_cat}</strong> — highest performer. Invest more budget here.")
    recs.append(f"🔍 <strong>Investigate {bottom_cat}</strong> — lowest performer. Run root-cause analysis.")
    recs.append(f"⚖️ <strong>Reduce variance</strong> — standardize processes to bring outliers to mean.")
    recs.append(f"🎯 <strong>Set targets</strong> at top-quartile ({df[num_col].quantile(0.75):,.2f}) for all categories.")
    return insights, recs

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>🛒 DataLens</h1><p>Amazon Products – Insight & Prediction Platform</p></div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    src = st.radio("Data Source", ["📦 Amazon Dataset", "📂 Upload CSV"])

    if src == "📂 Upload CSV":
        upl = st.file_uploader("Upload CSV", type=["csv"])
        df_raw = load_and_prepare(upl) if upl else load_and_prepare()
    else:
        df_raw = load_and_prepare()

    if df_raw is None:
        st.error("Could not load data. Please upload the CSV file.")
        st.stop()

    # Only use model-friendly columns
    NUMERIC_COLS = ['discounted_price', 'actual_price', 'discount_percentage',
                    'rating', 'rating_count', 'savings']
    CAT_COLS = ['main_category', 'sub_category', 'rating_bucket']

    numeric_cols = [c for c in NUMERIC_COLS if c in df_raw.columns]
    cat_cols     = [c for c in CAT_COLS if c in df_raw.columns]

    st.markdown("---")
    sel_num = st.selectbox("Primary Numeric Column", numeric_cols, index=numeric_cols.index('rating') if 'rating' in numeric_cols else 0)
    sel_cat = st.selectbox("Primary Category Column", cat_cols, index=0)
    st.markdown("---")
    st.markdown("`Pandas` · `Matplotlib` · `Seaborn`\n`Scikit-learn` · `Streamlit`")

# Work with a focused dataframe (drop text-heavy columns for analysis)
ANALYSIS_COLS = numeric_cols + cat_cols + ['product_name']
df_work = df_raw[[c for c in ANALYSIS_COLS if c in df_raw.columns]].copy()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🧹 Data Cleaning", "📈 EDA Charts", "🔮 Insights", "💼 Recommendations", "🤖 Predictions"
])

# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    total_products = df_raw['product_id'].nunique() if 'product_id' in df_raw.columns else len(df_raw)
    avg_rating     = df_raw['rating'].mean() if 'rating' in df_raw.columns else 0
    avg_discount   = df_raw['discount_percentage'].mean() if 'discount_percentage' in df_raw.columns else 0
    num_cats       = df_raw['main_category'].nunique() if 'main_category' in df_raw.columns else 0

    for col, lbl, val in [
        (c1, "Total Products",   f"{total_products:,}"),
        (c2, "Avg Rating",       f"{avg_rating:.2f}⭐️"),
        (c3, "Avg Discount",     f"{avg_discount:.1f}%"),
        (c4, "Categories",       f"{num_cats}"),
    ]:
        col.markdown(f'<div class="metric-card"><h2>{val}</h2><p>{lbl}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
    display_cols = ['product_name', 'main_category', 'discounted_price', 'actual_price',
                    'discount_percentage', 'rating', 'rating_count', 'savings']
    display_cols = [c for c in display_cols if c in df_raw.columns]
    st.dataframe(df_raw[display_cols].head(20), width='stretch', height=300)

    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df_raw[numeric_cols].describe().round(2), width='stretch')

# ── TAB 2: Data Cleaning ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Data Cleaning Report</div>', unsafe_allow_html=True)

    analysis_df = df_raw[numeric_cols + cat_cols]
    miss     = analysis_df.isnull().sum()
    miss_pct = (miss / len(analysis_df) * 100).round(2)
    mdf      = pd.DataFrame({"Count": miss, "Pct%": miss_pct})
    mdf      = mdf[mdf["Count"] > 0]

    if not mdf.empty:
        st.warning(f"⚠️ {mdf['Count'].sum()} missing values in {len(mdf)} columns")
        st.dataframe(mdf, width='stretch')
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(mdf.index, mdf["Pct%"], color=PALETTE[0], height=0.5)
        ax.set_title("Missing Values %")
        apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()
    else:
        st.success("✅ No missing values in analysis columns!")

    dups = df_raw.duplicated(subset=['product_id']).sum() if 'product_id' in df_raw.columns else df_raw.duplicated().sum()
    st.success("✅ No duplicate products!" if dups == 0 else f"⚠️ {dups} duplicate product IDs found")

    st.markdown('<div class="section-title">Outlier Detection (IQR)</div>', unsafe_allow_html=True)
    dfc = clean_df(df_work); out = {}
    for col in dfc.select_dtypes(include=np.number).columns:
        Q1, Q3 = dfc[col].quantile(0.25), dfc[col].quantile(0.75); IQR = Q3 - Q1
        out[col] = dfc[(dfc[col] < Q1 - 1.5 * IQR) | (dfc[col] > Q3 + 1.5 * IQR)].shape[0]
    odf = pd.DataFrame.from_dict(out, orient="index", columns=["Outliers"])
    odf = odf[odf["Outliers"] > 0].sort_values("Outliers", ascending=False)
    if not odf.empty:
        st.dataframe(odf, width='stretch')
    else:
        st.success("✅ No significant outliers!")

    st.success(f"✅ Clean dataset: {len(dfc)} rows × {len(numeric_cols + cat_cols)} analysis columns")

# ── TAB 3: EDA Charts ─────────────────────────────────────────────────────────
with tab3:
    dfc = clean_df(df_work)
    st.markdown('<div class="section-title">Distribution Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dfc[sel_num].dropna(), bins=30, color=PALETTE[0], edgecolor=BG_COLOR, alpha=0.9)
        ax.axvline(dfc[sel_num].mean(), color=PALETTE[2], linestyle="--", linewidth=1.5,
                   label=f"Mean: {dfc[sel_num].mean():.2f}")
        ax.axvline(dfc[sel_num].median(), color=PALETTE[1], linestyle=":", linewidth=1.5,
                   label=f"Median: {dfc[sel_num].median():.2f}")
        ax.legend(fontsize=8, facecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        ax.set_title(f"Distribution of {sel_num}")
        apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        cats = dfc[sel_cat].unique()[:10]  # limit to 10 categories
        bp = ax.boxplot([dfc[dfc[sel_cat] == c][sel_num].dropna() for c in cats],
                        patch_artist=True, labels=[str(c)[:15] for c in cats],
                        medianprops=dict(color="#facc15", linewidth=2))
        for p, col in zip(bp["boxes"], PALETTE): p.set_facecolor(col); p.set_alpha(0.7)
        ax.set_title(f"{sel_num} by {sel_cat}")
        plt.xticks(rotation=30, ha="right", fontsize=7)
        apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Category Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        agg = dfc.groupby(sel_cat)[sel_num].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(range(len(agg)), agg.values, color=PALETTE[:len(agg)], edgecolor="none", width=0.6)
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels([str(x)[:12] for x in agg.index], rotation=30, ha="right", fontsize=7)
        for b, v in zip(bars, agg.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + agg.max() * 0.01,
                    f"{v:,.2f}", ha="center", va="bottom", fontsize=7, color=TEXT_COLOR)
        ax.set_title(f"Avg {sel_num} by {sel_cat}")
        apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()
    with c2:
        counts = dfc[sel_cat].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(6, 4))
        _, _, ats = ax.pie(counts, labels=[str(x)[:15] for x in counts.index],
                           autopct="%1.1f%%", colors=PALETTE[:len(counts)],
                           startangle=90, pctdistance=0.8,
                           textprops={"color": TEXT_COLOR, "fontsize": 7})
        for at in ats: at.set_color("#fff"); at.set_fontsize(7)
        ax.set_title(f"{sel_cat} Distribution")
        fig.patch.set_facecolor(BG_COLOR); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = dfc[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f",
                cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 9, "color": "white"}, linewidths=0.5, linecolor=BG_COLOR,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation Matrix")
    apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()

    if len(numeric_cols) >= 2:
        st.markdown('<div class="section-title">Relationship Explorer</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        xc = sc1.selectbox("X-axis", numeric_cols, index=0, key="sx")
        yc = sc2.selectbox("Y-axis", numeric_cols,
                           index=min(1, len(numeric_cols) - 1), key="sy")
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cat in enumerate(dfc[sel_cat].unique()[:10]):
            m = dfc[sel_cat] == cat
            ax.scatter(dfc[m][xc], dfc[m][yc], color=PALETTE[i % len(PALETTE)],
                       alpha=0.65, s=30, label=str(cat)[:15], edgecolors="none")
        ax.set_xlabel(xc); ax.set_ylabel(yc); ax.set_title(f"{xc} vs {yc}")
        ax.legend(fontsize=7, facecolor=GRID_COLOR, labelcolor=TEXT_COLOR, framealpha=0.6,
                  bbox_to_anchor=(1.01, 1), loc='upper left')
        apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()

    # Amazon-specific: Top 10 most reviewed products
    st.markdown('<div class="section-title">Top 10 Most Reviewed Products</div>', unsafe_allow_html=True)
    if 'product_name' in df_raw.columns and 'rating_count' in df_raw.columns:
        top_reviewed = df_raw[['product_name', 'rating', 'rating_count', 'discounted_price', 'main_category']]\
            .sort_values('rating_count', ascending=False).head(10)
        top_reviewed['product_name'] = top_reviewed['product_name'].str[:50]
        st.dataframe(top_reviewed, width='stretch', hide_index=True)

# ── TAB 4: Insights ───────────────────────────────────────────────────────────
with tab4:
    dfc = clean_df(df_work)
    st.markdown('<div class="section-title">🔮 Auto-Generated Insights</div>', unsafe_allow_html=True)
    ins, recs = generate_insights(dfc, sel_num, sel_cat)
    for i in ins:
        st.markdown(f'<div class="insight-box">💡 {i}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top vs Bottom Categories</div>', unsafe_allow_html=True)
    grp = dfc.groupby(sel_cat)[sel_num].agg(["mean", "sum", "count"]).round(2)
    grp.columns = ["Average", "Total", "Count"]
    grp = grp.sort_values("Average", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🏆 Top**"); st.dataframe(grp.head(), width='stretch')
    with c2:
        st.markdown("**📉 Bottom**"); st.dataframe(grp.tail(), width='stretch')

    # Amazon-specific: Discount vs Rating scatter
    st.markdown('<div class="section-title">Discount % vs Rating</div>', unsafe_allow_html=True)
    if 'discount_percentage' in dfc.columns and 'rating' in dfc.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        sc = ax.scatter(dfc['discount_percentage'], dfc['rating'],
                        c=dfc['rating_count'] if 'rating_count' in dfc.columns else PALETTE[0],
                        cmap='plasma', alpha=0.5, s=20, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Rating Count")
        ax.set_xlabel("Discount %"); ax.set_ylabel("Rating")
        ax.set_title("Does Higher Discount = Better Rating?")
        apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()
        corr_val = dfc['discount_percentage'].corr(dfc['rating'])
        direction = "positive" if corr_val > 0 else "negative"
        strength  = "strong" if abs(corr_val) > 0.5 else "weak"
        st.markdown(f'<div class="insight-box">📊 Correlation between Discount% and Rating: <strong>{corr_val:.3f}</strong> ({strength} {direction} relationship)</div>', unsafe_allow_html=True)

    # Price distribution by category
    st.markdown('<div class="section-title">Price Distribution by Main Category</div>', unsafe_allow_html=True)
    if 'discounted_price' in dfc.columns and 'main_category' in dfc.columns:
        cats_sorted = dfc.groupby('main_category')['discounted_price'].median().sort_values(ascending=False).index[:8]
        fig, ax = plt.subplots(figsize=(12, 4))
        bp = ax.boxplot([dfc[dfc['main_category'] == c]['discounted_price'].dropna() for c in cats_sorted],
                        patch_artist=True, labels=[c[:12] for c in cats_sorted],
                        medianprops=dict(color="#facc15", linewidth=2))
        for p, col in zip(bp["boxes"], PALETTE): p.set_facecolor(col); p.set_alpha(0.7)
        ax.set_title("Discounted Price by Category")
        plt.xticks(rotation=20, ha="right", fontsize=8)
        apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()

# ── TAB 5: Recommendations ────────────────────────────────────────────────────
with tab5:
    dfc = clean_df(df_work)
    _, recs = generate_insights(dfc, sel_num, sel_cat)
    st.markdown('<div class="section-title">💼 Business Recommendations</div>', unsafe_allow_html=True)
    for r in recs:
        st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Priority Action Matrix</div>', unsafe_allow_html=True)
    tc = dfc.groupby(sel_cat)[sel_num].mean().idxmax()
    bc = dfc.groupby(sel_cat)[sel_num].mean().idxmin()
    st.dataframe(pd.DataFrame({
        "Action":   [f"Expand {str(tc)[:20]}", "Improve rating quality", f"Review {str(bc)[:20]}", "Pricing strategy", "Review campaigns"],
        "Impact":   ["🔴 High", "🔴 High", "🟡 Medium", "🔴 High", "🟡 Medium"],
        "Effort":   ["🟡 Medium", "🔴 High", "🟡 Medium", "🟢 Low", "🟡 Medium"],
        "Timeline": ["Q1 2025", "Q2 2025", "Q2 2025", "Q1 2025", "Q3 2025"],
        "Owner":    ["Sales", "Quality", "Strategy", "Pricing", "Marketing"],
    }), width='stretch', hide_index=True)

    avg_v  = dfc[sel_num].mean()
    top_v  = dfc.groupby(sel_cat)[sel_num].mean().max()
    st.markdown(f"""<div style="background:rgba(255,255,255,0.04);border-radius:16px;padding:24px 28px;line-height:1.8;color:#c4b5fd;font-size:0.95rem;">
    <strong style="color:#e9d5ff;font-size:1.1rem;">Executive Summary</strong><br><br>
    Analysis covers <strong>{len(dfc):,}</strong> Amazon product records across <strong>{dfc['main_category'].nunique() if 'main_category' in dfc.columns else 'N/A'}</strong> categories.<br><br>
    <strong>Key Finding 1:</strong> {str(tc)[:30]} leads with avg {sel_num} = <strong>{top_v:,.2f}</strong>
    ({((top_v / avg_v - 1) * 100):.1f}% above mean).<br><br>
    <strong>Key Finding 2:</strong> {str(bc)[:30]} underperforms — immediate review needed.<br><br>
    <strong>Key Finding 3:</strong> Average discount across all products: <strong>{dfc['discount_percentage'].mean():.1f}%</strong>
    with avg rating <strong>{dfc['rating'].mean():.2f} ⭐</strong>.
    </div>""", unsafe_allow_html=True)

# ── TAB 6: Predictions ────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-title">🤖 ML Prediction Engine</div>', unsafe_allow_html=True)
    dfc = clean_df(df_work)

    # Only use numeric + encoded categoricals for ML
    ML_NUMERIC = numeric_cols
    ML_CAT     = cat_cols

    df_enc_full = dfc.copy()
    le_map = {}
    for col in ML_CAT:
        le = LabelEncoder()
        df_enc_full[col] = le.fit_transform(df_enc_full[col].astype(str))
        le_map[col] = le

    df_enc = df_enc_full[ML_NUMERIC + ML_CAT].copy()

    pred_type = st.radio("Choose Prediction Type",
        ["📉 Regression — Predict a Number", "🏷️ Classification — Predict a Category"], horizontal=True)
    st.markdown("---")

    if "Regression" in pred_type:
        st.markdown("### 📉 Regression — Predict a Numeric Value")
        col_a, col_b = st.columns(2)
        target_col   = col_a.selectbox("🎯 Target (what to predict)", ML_NUMERIC, key="rgt",
                                        index=ML_NUMERIC.index('rating') if 'rating' in ML_NUMERIC else 0)
        model_choice = col_b.selectbox("🧠 Algorithm",
            ["Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor"])

        feature_pool = [c for c in df_enc.columns if c != target_col]
        sel_features = st.multiselect("📦 Features (input variables)", feature_pool,
                                      default=feature_pool[:min(5, len(feature_pool))])
        test_pct = st.slider("Test Size %", 10, 40, 20, key="rslide")

        if len(sel_features) >= 1 and st.button("🚀 Train & Evaluate", key="rbtn"):
            X = df_enc[sel_features].replace([np.inf, -np.inf], np.nan).fillna(df_enc[sel_features].mean())
            y = df_enc[target_col].replace([np.inf, -np.inf], np.nan).fillna(df_enc[target_col].mean())
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_pct / 100, random_state=42)
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

            if model_choice == "Linear Regression":         mdl = LinearRegression()
            elif model_choice == "Random Forest Regressor": mdl = RandomForestRegressor(n_estimators=100, random_state=42)
            else:                                           mdl = GradientBoostingRegressor(n_estimators=100, random_state=42)

            mdl.fit(Xtr_s, ytr); ypred = mdl.predict(Xte_s)
            mae  = mean_absolute_error(yte, ypred)
            rmse = np.sqrt(mean_squared_error(yte, ypred))
            r2   = r2_score(yte, ypred)

            st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="pred-card"><h2>{r2:.3f}</h2><p>R² Score</p></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="pred-card"><h2>{mae:,.2f}</h2><p>MAE</p></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="pred-card"><h2>{rmse:,.2f}</h2><p>RMSE</p></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><h2>{len(yte)}</h2><p>Test Rows</p></div>', unsafe_allow_html=True)

            q = ("🟢 Excellent! High accuracy." if r2 >= 0.8 else
                 "🟡 Good. Decent accuracy."    if r2 >= 0.6 else
                 "🟠 Fair. Try more features."  if r2 >= 0.4 else
                 "🔴 Weak. Consider different features.")
            st.markdown(f'<div class="insight-box">🧠 <strong>Model:</strong> {q} | R²={r2:.3f} → model explains <strong>{r2*100:.1f}%</strong> of variance in {target_col}.</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Actual vs Predicted + Residuals</div>', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(yte, ypred, color=PALETTE[0], alpha=0.5, s=20, edgecolors="none")
            lims = [min(yte.min(), ypred.min()), max(yte.max(), ypred.max())]
            axes[0].plot(lims, lims, color=PALETTE[2], linestyle="--", linewidth=1.5, label="Perfect fit")
            axes[0].set_xlabel(f"Actual {target_col}"); axes[0].set_ylabel("Predicted")
            axes[0].set_title("Actual vs Predicted")
            axes[0].legend(fontsize=8, facecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            residuals = yte.values - ypred
            axes[1].scatter(ypred, residuals, color=PALETTE[4], alpha=0.5, s=20, edgecolors="none")
            axes[1].axhline(0, color=PALETTE[2], linestyle="--", linewidth=1.5)
            axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual"); axes[1].set_title("Residual Plot")
            apply_dark_style(fig, axes.tolist()); st.pyplot(fig); plt.close()

            st.markdown('<div class="section-title">Feature Importance / Coefficients</div>', unsafe_allow_html=True)
            if hasattr(mdl, "feature_importances_"):
                fi = pd.Series(mdl.feature_importances_, index=sel_features).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(8, max(3, len(fi) * 0.45)))
                ax.barh(fi.index, fi.values, color=PALETTE[:len(fi)], edgecolor="none")
                for i, (idx, v) in enumerate(fi.items()):
                    ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT_COLOR)
                ax.set_title("Feature Importance")
                apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()
            else:
                coef = pd.Series(mdl.coef_, index=sel_features).sort_values()
                fig, ax = plt.subplots(figsize=(8, max(3, len(coef) * 0.45)))
                ax.barh(coef.index, coef.values,
                        color=[PALETTE[2] if v >= 0 else PALETTE[9] for v in coef.values], edgecolor="none")
                ax.axvline(0, color=TEXT_COLOR, linewidth=0.8, linestyle="--")
                ax.set_title("Regression Coefficients (green=positive, red=negative)")
                apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()

            st.session_state["reg_model"]  = mdl
            st.session_state["reg_scaler"] = sc
            st.session_state["reg_feats"]  = sel_features
            st.session_state["reg_target"] = target_col

        if "reg_model" in st.session_state:
            st.markdown('<div class="section-title">⚡ Live Predictor — Enter Values</div>', unsafe_allow_html=True)
            inp   = {}
            feats = st.session_state["reg_feats"]
            groups = [feats[i:i+4] for i in range(0, len(feats), 4)]
            for grp in groups:
                cols_row = st.columns(len(grp))
                for c, f in zip(cols_row, grp):
                    mn = float(df_enc[f].min()); mx = float(df_enc[f].max()); mv = float(df_enc[f].mean())
                    inp[f] = c.number_input(f, min_value=mn, max_value=mx, value=round(mv, 2), key=f"ri_{f}")
            if st.button("🔮 Predict Now", key="r_live"):
                inp_s  = st.session_state["reg_scaler"].transform(pd.DataFrame([inp]))
                result = st.session_state["reg_model"].predict(inp_s)[0]
                t_col  = st.session_state.get("reg_target", target_col)
                st.markdown(f"""<div class="pred-result-box">
                  <p>Predicted {t_col}</p>
                  <h1>{result:,.2f}</h1>
                  <p>Model: {model_choice}</p></div>""", unsafe_allow_html=True)

    else:
        st.markdown("### 🏷️ Classification — Predict a Category")
        col_a, col_b = st.columns(2)
        target_col   = col_a.selectbox("🎯 Target (category to predict)", ML_CAT, key="cgt",
                                        index=ML_CAT.index('rating_bucket') if 'rating_bucket' in ML_CAT else 0)
        model_choice = col_b.selectbox("🧠 Algorithm",
            ["Random Forest Classifier", "Logistic Regression"])

        feature_pool = [c for c in df_enc.columns if c != target_col]
        sel_features = st.multiselect("📦 Features", feature_pool,
                                      default=feature_pool[:min(5, len(feature_pool))], key="cfeats")
        test_pct = st.slider("Test Size %", 10, 40, 20, key="cslide")

        if len(sel_features) >= 1 and st.button("🚀 Train Classifier", key="cbtn"):
            X = df_enc[sel_features]; y = df_enc[target_col]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_pct / 100, random_state=42)
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

            if model_choice == "Random Forest Classifier":
                mdl = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                mdl = LogisticRegression(max_iter=500, random_state=42)

            mdl.fit(Xtr_s, ytr); ypred = mdl.predict(Xte_s)
            acc = accuracy_score(yte, ypred)

            st.markdown('<div class="section-title">📊 Classifier Performance</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="pred-card"><h2>{acc*100:.1f}%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="pred-card"><h2>{len(np.unique(y))}</h2><p>Classes</p></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><h2>{len(Xte)}</h2><p>Test Samples</p></div>', unsafe_allow_html=True)

            q = ("🟢 Excellent classifier!" if acc >= 0.85 else
                 "🟡 Good classifier."       if acc >= 0.70 else
                 "🟠 Fair — try more features." if acc >= 0.55 else
                 "🔴 Weak — consider different features.")
            st.markdown(f'<div class="insight-box">🧠 {q} Accuracy = <strong>{acc*100:.1f}%</strong> on unseen data.</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
            le = le_map.get(target_col)
            class_labels = le.classes_ if le else np.unique(y)
            cm = confusion_matrix(yte, ypred)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=class_labels, yticklabels=class_labels,
                        annot_kws={"size": 11, "color": "white"},
                        linewidths=0.5, linecolor=BG_COLOR)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
            apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()

            st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
            rpt = classification_report(yte, ypred,
                                        target_names=[str(c) for c in class_labels], output_dict=True)
            st.dataframe(pd.DataFrame(rpt).T.round(3), width='stretch')

            if hasattr(mdl, "feature_importances_"):
                st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
                fi = pd.Series(mdl.feature_importances_, index=sel_features).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(8, max(3, len(fi) * 0.45)))
                ax.barh(fi.index, fi.values, color=PALETTE[:len(fi)], edgecolor="none")
                for i, (idx, v) in enumerate(fi.items()):
                    ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT_COLOR)
                ax.set_title("Feature Importance")
                apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()

            st.session_state["cls_model"]  = mdl
            st.session_state["cls_scaler"] = sc
            st.session_state["cls_feats"]  = sel_features
            st.session_state["cls_target"] = target_col
            st.session_state["cls_labels"] = class_labels
            st.session_state["cls_le"]     = le

        if "cls_model" in st.session_state:
            st.markdown('<div class="section-title">⚡ Live Category Predictor</div>', unsafe_allow_html=True)
            inp   = {}
            feats = st.session_state["cls_feats"]
            groups = [feats[i:i+4] for i in range(0, len(feats), 4)]
            for grp in groups:
                cols_row = st.columns(len(grp))
                for c, f in zip(cols_row, grp):
                    mn = float(df_enc[f].min()); mx = float(df_enc[f].max()); mv = float(df_enc[f].mean())
                    inp[f] = c.number_input(f, min_value=mn, max_value=mx, value=round(mv, 2), key=f"ci_{f}")
            if st.button("🔮 Classify Now", key="c_live"):
                inp_s    = st.session_state["cls_scaler"].transform(pd.DataFrame([inp]))
                pred_enc = st.session_state["cls_model"].predict(inp_s)[0]
                probas   = st.session_state["cls_model"].predict_proba(inp_s)[0]
                le_c     = st.session_state.get("cls_le")
                label    = le_c.inverse_transform([pred_enc])[0] if le_c else pred_enc
                conf     = probas.max() * 100
                clabels  = st.session_state["cls_labels"]

                st.markdown(f"""<div class="pred-result-box">
                  <p>Predicted {st.session_state['cls_target']}</p>
                  <h1>{label}</h1>
                  <p>Confidence: {conf:.1f}% | Model: {model_choice}</p></div>""", unsafe_allow_html=True)

                prob_df = pd.DataFrame({
                    "Class": [str(c) for c in clabels],
                    "Prob%": (probas * 100).round(1)
                }).sort_values("Prob%", ascending=True)
                fig, ax = plt.subplots(figsize=(8, max(3, len(clabels) * 0.5)))
                cols_p = [PALETTE[2] if str(c) == str(label) else PALETTE[0] for c in prob_df["Class"]]
                ax.barh(prob_df["Class"], prob_df["Prob%"], color=cols_p, edgecolor="none")
                ax.set_xlabel("Probability (%)"); ax.set_title("Class Probabilities")
                for i, v in enumerate(prob_df["Prob%"].values):
                    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9, color=TEXT_COLOR)
                apply_dark_style(fig, [ax]); st.pyplot(fig); plt.close()

st.markdown('<div style="text-align:center;padding:40px 0 20px;color:#334155;font-size:0.8rem;">DataLens · Amazon Products · Streamlit · Pandas · Matplotlib · Seaborn · Scikit-learn</div>', unsafe_allow_html=True)
