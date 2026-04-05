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

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DataLens – Insight & Prediction", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
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

# ── Colors ────────────────────────────────────────────────────────────────────
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

# ── Sample data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "Month":        np.random.choice(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], n),
        "Category":     np.random.choice(["Electronics","Clothing","Food","Books","Sports"], n),
        "Region":       np.random.choice(["North","South","East","West","Central"], n),
        "Sales":        np.random.normal(5000,1500,n).clip(500).round(2),
        "Units_Sold":   np.random.randint(10,200,n),
        "Customer_Age": np.random.randint(18,65,n),
        "Rating":       np.round(np.random.uniform(2.5,5.0,n),1),
        "Discount_%":   np.random.choice([0,5,10,15,20,25],n),
        "Return_Flag":  np.random.choice([0,1],n,p=[0.85,0.15]),
        "Ad_Spend":     np.random.normal(800,250,n).clip(100).round(2),
    })
    for col in ["Sales","Rating","Customer_Age"]:
        df.loc[df.sample(frac=0.04).index, col] = np.nan
    return df

def clean_df(df):
    d = df.copy()
    for col in d.select_dtypes(include=np.number).columns:   d[col].fillna(d[col].median(), inplace=True)
    for col in d.select_dtypes(include="object").columns:    d[col].fillna(d[col].mode()[0], inplace=True)
    return d

def encode_df(df):
    d, le_map = df.copy(), {}
    for col in d.select_dtypes(include="object").columns:
        le = LabelEncoder(); d[col] = le.fit_transform(d[col].astype(str)); le_map[col] = le
    return d, le_map

def generate_insights(df, num_col, cat_col):
    insights, recs = [], []
    grp = df.groupby(cat_col)[num_col].mean()
    top_cat, bottom_cat = grp.idxmax(), grp.idxmin()
    top_val, bottom_val = grp.max(), grp.min()
    mean_val = df[num_col].mean(); std_val = df[num_col].std()
    above = (df[num_col] > mean_val).sum()
    insights.append(f"<strong>{top_cat}</strong> has highest avg {num_col}: <strong>{top_val:,.1f}</strong>")
    insights.append(f"<strong>{bottom_cat}</strong> has lowest avg {num_col}: <strong>{bottom_val:,.1f}</strong>")
    insights.append(f"{above} records ({above/len(df)*100:.1f}%) are above overall mean of <strong>{mean_val:,.1f}</strong>")
    insights.append(f"Std deviation = <strong>{std_val:,.1f}</strong> — {'high variability' if std_val>mean_val*0.3 else 'stable values'}")
    recs.append(f"📈 <strong>Scale {top_cat}</strong> — highest performer. Invest more budget here.")
    recs.append(f"🔍 <strong>Investigate {bottom_cat}</strong> — lowest performer. Run root-cause analysis.")
    recs.append(f"⚖️ <strong>Reduce variance</strong> — standardize processes to bring outliers to mean.")
    recs.append(f"🎯 <strong>Set targets</strong> at top-quartile ({df[num_col].quantile(0.75):,.1f}) for all categories.")
    return insights, recs

# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header"><h1>🔍 DataLens</h1><p>Data-Driven Insight & Prediction Platform</p></div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    src = st.radio("Data Source", ["📦 Sample Dataset","📂 Upload CSV"])
    if src == "📂 Upload CSV":
        upl = st.file_uploader("Upload CSV", type=["csv"])
        df_raw = pd.read_csv(upl) if upl else load_sample_data()
    else:
        df_raw = load_sample_data()
    st.markdown("---")
    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df_raw.select_dtypes(include="object").columns.tolist()
    sel_num = st.selectbox("Primary Numeric Column", numeric_cols)
    sel_cat = st.selectbox("Primary Category Column", cat_cols)
    st.markdown("---")
    st.markdown("`Pandas` · `Matplotlib` · `Seaborn`\n`Scikit-learn` · `Streamlit`")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊 Overview","🧹 Data Cleaning","📈 EDA Charts","🔮 Insights","💼 Recommendations","🤖 Predictions"
])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val in [(c1,"Total Rows",f"{len(df_raw):,}"),(c2,"Columns",f"{df_raw.shape[1]}"),
                        (c3,"Numeric Cols",f"{len(numeric_cols)}"),(c4,"Missing",f"{df_raw.isnull().sum().sum()}")]:
        col.markdown(f'<div class="metric-card"><h2>{val}</h2><p>{lbl}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(20), use_container_width=True, height=300)
    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe().round(2), use_container_width=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Data Cleaning Report</div>', unsafe_allow_html=True)
    miss = df_raw.isnull().sum(); miss_pct = (miss/len(df_raw)*100).round(2)
    mdf  = pd.DataFrame({"Count":miss,"Pct%":miss_pct}); mdf = mdf[mdf["Count"]>0]
    if not mdf.empty:
        st.warning(f"⚠️ {mdf['Count'].sum()} missing values in {len(mdf)} columns")
        st.dataframe(mdf, use_container_width=True)
        fig,ax = plt.subplots(figsize=(10,3))
        ax.barh(mdf.index, mdf["Pct%"], color=PALETTE[0], height=0.5)
        ax.set_title("Missing Values %"); apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()
    else: st.success("✅ No missing values!")
    dups = df_raw.duplicated().sum()
    st.success("✅ No duplicates!" if dups==0 else f"⚠️ {dups} duplicate rows found")
    st.markdown('<div class="section-title">Outlier Detection (IQR)</div>', unsafe_allow_html=True)
    dfc = clean_df(df_raw); out = {}
    for col in dfc.select_dtypes(include=np.number).columns:
        Q1,Q3 = dfc[col].quantile(0.25), dfc[col].quantile(0.75); IQR=Q3-Q1
        out[col] = dfc[(dfc[col]<Q1-1.5*IQR)|(dfc[col]>Q3+1.5*IQR)].shape[0]
    odf = pd.DataFrame.from_dict(out,orient="index",columns=["Outliers"])
    odf = odf[odf["Outliers"]>0].sort_values("Outliers",ascending=False)
    if not odf.empty: st.dataframe(odf, use_container_width=True)
    else: st.success("✅ No significant outliers!")
    st.success(f"✅ Clean dataset: {len(dfc)} rows × {dfc.shape[1]} columns")

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    dfc = clean_df(df_raw)
    st.markdown('<div class="section-title">Distribution Analysis</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig,ax = plt.subplots(figsize=(6,4))
        ax.hist(dfc[sel_num],bins=30,color=PALETTE[0],edgecolor=BG_COLOR,alpha=0.9)
        ax.axvline(dfc[sel_num].mean(),color=PALETTE[2],linestyle="--",linewidth=1.5,label=f"Mean:{dfc[sel_num].mean():.1f}")
        ax.axvline(dfc[sel_num].median(),color=PALETTE[1],linestyle=":",linewidth=1.5,label=f"Median:{dfc[sel_num].median():.1f}")
        ax.legend(fontsize=8,facecolor=GRID_COLOR,labelcolor=TEXT_COLOR)
        ax.set_title(f"Distribution of {sel_num}"); apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()
    with c2:
        fig,ax = plt.subplots(figsize=(6,4))
        cats = dfc[sel_cat].unique()
        bp = ax.boxplot([dfc[dfc[sel_cat]==c][sel_num].dropna() for c in cats], patch_artist=True,
                        labels=cats, medianprops=dict(color="#facc15",linewidth=2))
        for p,col in zip(bp["boxes"],PALETTE): p.set_facecolor(col); p.set_alpha(0.7)
        ax.set_title(f"{sel_num} by {sel_cat}"); plt.xticks(rotation=20,ha="right")
        apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Category Analysis</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        agg = dfc.groupby(sel_cat)[sel_num].mean().sort_values(ascending=False)
        fig,ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(agg.index,agg.values,color=PALETTE[:len(agg)],edgecolor="none",width=0.6)
        for b,v in zip(bars,agg.values):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+agg.max()*0.01,f"{v:,.0f}",ha="center",va="bottom",fontsize=8,color=TEXT_COLOR)
        ax.set_title(f"Avg {sel_num} by {sel_cat}"); plt.xticks(rotation=20,ha="right")
        apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()
    with c2:
        counts = dfc[sel_cat].value_counts()
        fig,ax = plt.subplots(figsize=(6,4))
        _,_,ats = ax.pie(counts,labels=counts.index,autopct="%1.1f%%",colors=PALETTE[:len(counts)],
                         startangle=90,pctdistance=0.8,textprops={"color":TEXT_COLOR,"fontsize":9})
        for at in ats: at.set_color("#fff"); at.set_fontsize(8)
        ax.set_title(f"{sel_cat} Distribution"); fig.patch.set_facecolor(BG_COLOR); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = dfc[numeric_cols].corr()
    fig,ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr,mask=np.triu(np.ones_like(corr,dtype=bool)),annot=True,fmt=".2f",
                cmap=sns.diverging_palette(240,10,as_cmap=True),center=0,vmin=-1,vmax=1,ax=ax,
                annot_kws={"size":9,"color":"white"},linewidths=0.5,linecolor=BG_COLOR,cbar_kws={"shrink":0.7})
    ax.set_title("Correlation Matrix"); apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()

    if len(numeric_cols)>=2:
        st.markdown('<div class="section-title">Relationship Explorer</div>', unsafe_allow_html=True)
        sc1,sc2 = st.columns(2)
        xc = sc1.selectbox("X-axis",numeric_cols,index=0,key="sx")
        yc = sc2.selectbox("Y-axis",numeric_cols,index=min(1,len(numeric_cols)-1),key="sy")
        fig,ax = plt.subplots(figsize=(10,5))
        for i,cat in enumerate(dfc[sel_cat].unique()):
            m = dfc[sel_cat]==cat
            ax.scatter(dfc[m][xc],dfc[m][yc],color=PALETTE[i%len(PALETTE)],alpha=0.65,s=30,label=cat,edgecolors="none")
        ax.set_xlabel(xc); ax.set_ylabel(yc); ax.set_title(f"{xc} vs {yc}")
        ax.legend(fontsize=8,facecolor=GRID_COLOR,labelcolor=TEXT_COLOR,framealpha=0.6)
        apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()

# ══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    dfc = clean_df(df_raw)
    st.markdown('<div class="section-title">🔮 Auto-Generated Insights</div>', unsafe_allow_html=True)
    ins,recs = generate_insights(dfc,sel_num,sel_cat)
    for i in ins: st.markdown(f'<div class="insight-box">💡 {i}</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top vs Bottom</div>', unsafe_allow_html=True)
    grp = dfc.groupby(sel_cat)[sel_num].agg(["mean","sum","count"]).round(2)
    grp.columns = ["Average","Total","Count"]; grp = grp.sort_values("Average",ascending=False)
    c1,c2 = st.columns(2)
    with c1: st.markdown("**🏆 Top**"); st.dataframe(grp.head(),use_container_width=True)
    with c2: st.markdown("**📉 Bottom**"); st.dataframe(grp.tail(),use_container_width=True)
    if "Month" in dfc.columns:
        st.markdown('<div class="section-title">Monthly Trend</div>', unsafe_allow_html=True)
        mo = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        mdf = dfc.groupby("Month")[sel_num].mean().reindex(mo).dropna()
        fig,ax = plt.subplots(figsize=(10,4))
        ax.plot(mdf.index,mdf.values,color=PALETTE[0],linewidth=2.5,marker="o",markersize=7,markerfacecolor=PALETTE[2],markeredgecolor="none")
        ax.fill_between(mdf.index,mdf.values,alpha=0.15,color=PALETTE[0])
        ax.set_title(f"Monthly Avg {sel_num}"); apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()

# ══ TAB 5 ════════════════════════════════════════════════════════════════════
with tab5:
    dfc = clean_df(df_raw)
    _,recs = generate_insights(dfc,sel_num,sel_cat)
    st.markdown('<div class="section-title">💼 Business Recommendations</div>', unsafe_allow_html=True)
    for r in recs: st.markdown(f'<div class="rec-box">{r}</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Priority Action Matrix</div>', unsafe_allow_html=True)
    tc = dfc.groupby(sel_cat)[sel_num].mean().idxmax()
    bc = dfc.groupby(sel_cat)[sel_num].mean().idxmin()
    st.dataframe(pd.DataFrame({"Action":[f"Expand {tc}","Fix data gaps",f"Review {bc}","Dashboards","A/B Testing"],
        "Impact":["🔴 High","🟡 Medium","🔴 High","🟡 Medium","🟢 Low"],
        "Effort":["🟡 Medium","🟢 Low","🔴 High","🟢 Low","🟡 Medium"],
        "Timeline":["Q1 2025","Immediate","Q2 2025","Q1 2025","Q3 2025"],
        "Owner":["Sales","Data Team","Strategy","Analytics","Marketing"]}),
        use_container_width=True, hide_index=True)
    avg_v = dfc[sel_num].mean(); top_v = dfc.groupby(sel_cat)[sel_num].mean().max()
    st.markdown(f"""<div style="background:rgba(255,255,255,0.04);border-radius:16px;padding:24px 28px;line-height:1.8;color:#c4b5fd;font-size:0.95rem;">
    <strong style="color:#e9d5ff;font-size:1.1rem;">Executive Summary</strong><br><br>
    Analysis covers <strong>{len(dfc):,}</strong> records across <strong>{dfc.shape[1]}</strong> variables.
    Missing values imputed (median/mode). Outliers flagged via IQR.<br><br>
    <strong>Key Finding 1:</strong> {tc} leads with avg {sel_num} = <strong>{top_v:,.1f}</strong>
    ({((top_v/avg_v-1)*100):.1f}% above mean).<br><br>
    <strong>Key Finding 2:</strong> {bc} underperforms — immediate review needed.<br><br>
    <strong>Key Finding 3:</strong> ML models predict {sel_num} with measurable accuracy — see Predictions tab.
    </div>""", unsafe_allow_html=True)

# ══ TAB 6 — PREDICTIONS 🤖 ════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">🤖 ML Prediction Engine</div>', unsafe_allow_html=True)
    dfc = clean_df(df_raw)
    df_enc, le_map = encode_df(dfc)

    pred_type = st.radio("Choose Prediction Type",
        ["📉 Regression — Predict a Number", "🏷️ Classification — Predict a Category"], horizontal=True)
    st.markdown("---")

    # ════════════════ REGRESSION ════════════════════════════════════════════
    if "Regression" in pred_type:
        st.markdown("### 📉 Regression — Predict a Numeric Value")
        col_a, col_b = st.columns(2)
        target_col   = col_a.selectbox("🎯 Target (what to predict)", numeric_cols, key="rgt")
        model_choice = col_b.selectbox("🧠 Algorithm",
            ["Linear Regression","Random Forest Regressor","Gradient Boosting Regressor"])

        feature_pool = [c for c in df_enc.columns if c != target_col]
        sel_features = st.multiselect("📦 Features (input variables)", feature_pool,
                                      default=feature_pool[:min(5,len(feature_pool))])
        test_pct = st.slider("Test Size %", 10, 40, 20, key="rslide")

        if len(sel_features) >= 1 and st.button("🚀 Train & Evaluate", key="rbtn"):
            X = df_enc[sel_features]; y = df_enc[target_col]
            Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=test_pct/100, random_state=42)
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

            if model_choice == "Linear Regression":          mdl = LinearRegression()
            elif model_choice == "Random Forest Regressor":  mdl = RandomForestRegressor(n_estimators=100,random_state=42)
            else:                                             mdl = GradientBoostingRegressor(n_estimators=100,random_state=42)

            mdl.fit(Xtr_s, ytr); ypred = mdl.predict(Xte_s)
            mae  = mean_absolute_error(yte, ypred)
            rmse = np.sqrt(mean_squared_error(yte, ypred))
            r2   = r2_score(yte, ypred)

            # Metrics
            st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.markdown(f'<div class="pred-card"><h2>{r2:.3f}</h2><p>R² Score</p></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="pred-card"><h2>{mae:,.1f}</h2><p>MAE</p></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="pred-card"><h2>{rmse:,.1f}</h2><p>RMSE</p></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><h2>{len(yte)}</h2><p>Test Rows</p></div>', unsafe_allow_html=True)

            q = ("🟢 Excellent! High accuracy." if r2>=0.8 else
                 "🟡 Good. Decent accuracy." if r2>=0.6 else
                 "🟠 Fair. Try more features." if r2>=0.4 else
                 "🔴 Weak. Consider different features.")
            st.markdown(f'<div class="insight-box">🧠 <strong>Model:</strong> {q} | R²={r2:.3f} → model explains <strong>{r2*100:.1f}%</strong> of variance in {target_col}.</div>', unsafe_allow_html=True)

            # Actual vs Predicted + Residuals
            st.markdown('<div class="section-title">Actual vs Predicted + Residuals</div>', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(yte, ypred, color=PALETTE[0], alpha=0.5, s=20, edgecolors="none")
            lims = [min(yte.min(),ypred.min()), max(yte.max(),ypred.max())]
            axes[0].plot(lims, lims, color=PALETTE[2], linestyle="--", linewidth=1.5, label="Perfect fit")
            axes[0].set_xlabel(f"Actual {target_col}"); axes[0].set_ylabel("Predicted"); axes[0].set_title("Actual vs Predicted")
            axes[0].legend(fontsize=8, facecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            residuals = yte.values - ypred
            axes[1].scatter(ypred, residuals, color=PALETTE[4], alpha=0.5, s=20, edgecolors="none")
            axes[1].axhline(0, color=PALETTE[2], linestyle="--", linewidth=1.5)
            axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual"); axes[1].set_title("Residual Plot")
            apply_dark_style(fig, axes.tolist()); st.pyplot(fig); plt.close()

            # Feature Importance / Coefficients
            st.markdown('<div class="section-title">Feature Importance / Coefficients</div>', unsafe_allow_html=True)
            if hasattr(mdl, "feature_importances_"):
                fi = pd.Series(mdl.feature_importances_, index=sel_features).sort_values(ascending=True)
                fig,ax = plt.subplots(figsize=(8, max(3,len(fi)*0.45)))
                ax.barh(fi.index, fi.values, color=PALETTE[:len(fi)], edgecolor="none")
                for i,(idx,v) in enumerate(fi.items()):
                    ax.text(v+0.001, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT_COLOR)
                ax.set_title("Feature Importance"); apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()
            else:
                coef = pd.Series(mdl.coef_, index=sel_features).sort_values()
                fig,ax = plt.subplots(figsize=(8, max(3,len(coef)*0.45)))
                ax.barh(coef.index, coef.values, color=[PALETTE[2] if v>=0 else PALETTE[9] for v in coef.values], edgecolor="none")
                ax.axvline(0, color=TEXT_COLOR, linewidth=0.8, linestyle="--")
                ax.set_title("Regression Coefficients (green=positive, red=negative)"); apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()

            # Prediction Distribution
            st.markdown('<div class="section-title">Prediction Error Distribution</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(10,4))
            ax.hist(residuals, bins=30, color=PALETTE[1], edgecolor=BG_COLOR, alpha=0.85)
            ax.axvline(0, color=PALETTE[2], linestyle="--", linewidth=1.5, label="Zero error")
            ax.axvline(residuals.mean(), color=PALETTE[3], linestyle=":", linewidth=1.5, label=f"Mean error: {residuals.mean():.2f}")
            ax.legend(fontsize=8, facecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            ax.set_title("Residual Distribution"); ax.set_xlabel("Error (Actual - Predicted)"); ax.set_ylabel("Frequency")
            apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()

            # Store model in session
            st.session_state["reg_model"]  = mdl
            st.session_state["reg_scaler"] = sc
            st.session_state["reg_feats"]  = sel_features

        # Live Predictor — always shown if model trained
        if "reg_model" in st.session_state:
            st.markdown('<div class="section-title">⚡ Live Predictor — Enter Values</div>', unsafe_allow_html=True)
            st.markdown("Adjust input values and click **Predict** to get an instant forecast:")
            inp = {}
            feats = st.session_state["reg_feats"]
            groups = [feats[i:i+4] for i in range(0,len(feats),4)]
            for grp in groups:
                cols_row = st.columns(len(grp))
                for c,f in zip(cols_row,grp):
                    mn=float(df_enc[f].min()); mx=float(df_enc[f].max()); mv=float(df_enc[f].mean())
                    inp[f] = c.number_input(f, min_value=mn, max_value=mx, value=round(mv,2), key=f"ri_{f}")
            if st.button("🔮 Predict Now", key="r_live"):
                inp_s = st.session_state["reg_scaler"].transform(pd.DataFrame([inp]))
                result = st.session_state["reg_model"].predict(inp_s)[0]
                st.markdown(f"""<div class="pred-result-box">
                  <p>Predicted {target_col if "reg_model" in st.session_state else sel_num}</p>
                  <h1>{result:,.2f}</h1>
                  <p>Model: {model_choice}</p></div>""", unsafe_allow_html=True)

    # ════════════════ CLASSIFICATION ════════════════════════════════════════
    else:
        st.markdown("### 🏷️ Classification — Predict a Category")
        col_a, col_b = st.columns(2)
        target_col   = col_a.selectbox("🎯 Target (category to predict)", cat_cols, key="cgt")
        model_choice = col_b.selectbox("🧠 Algorithm",
            ["Random Forest Classifier","Logistic Regression"])

        feature_pool = [c for c in df_enc.columns if c != target_col]
        sel_features = st.multiselect("📦 Features", feature_pool,
                                      default=feature_pool[:min(5,len(feature_pool))], key="cfeats")
        test_pct = st.slider("Test Size %", 10, 40, 20, key="cslide")

        if len(sel_features) >= 1 and st.button("🚀 Train Classifier", key="cbtn"):
            X = df_enc[sel_features]; y = df_enc[target_col]
            Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=test_pct/100, random_state=42)
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

            if model_choice == "Random Forest Classifier": mdl = RandomForestClassifier(n_estimators=100,random_state=42)
            else:                                          mdl = LogisticRegression(max_iter=500,random_state=42)

            mdl.fit(Xtr_s, ytr); ypred = mdl.predict(Xte_s)
            acc = accuracy_score(yte, ypred)

            # Metrics
            st.markdown('<div class="section-title">📊 Classifier Performance</div>', unsafe_allow_html=True)
            m1,m2,m3 = st.columns(3)
            m1.markdown(f'<div class="pred-card"><h2>{acc*100:.1f}%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="pred-card"><h2>{len(np.unique(y))}</h2><p>Classes</p></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><h2>{len(Xte)}</h2><p>Test Samples</p></div>', unsafe_allow_html=True)

            q = ("🟢 Excellent classifier!" if acc>=0.85 else
                 "🟡 Good classifier." if acc>=0.70 else
                 "🟠 Fair — try more features." if acc>=0.55 else
                 "🔴 Weak — consider different features.")
            st.markdown(f'<div class="insight-box">🧠 {q} Accuracy = <strong>{acc*100:.1f}%</strong> on unseen data.</div>', unsafe_allow_html=True)

            # Confusion Matrix
            st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
            le = le_map.get(target_col)
            class_labels = le.classes_ if le else np.unique(y)
            cm = confusion_matrix(yte, ypred)
            fig,ax = plt.subplots(figsize=(8,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=class_labels, yticklabels=class_labels,
                        annot_kws={"size":11,"color":"white"}, linewidths=0.5, linecolor=BG_COLOR)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
            apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()

            # Classification Report
            st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
            rpt = classification_report(yte, ypred, target_names=[str(c) for c in class_labels], output_dict=True)
            st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)

            # Feature Importance
            if hasattr(mdl,"feature_importances_"):
                st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
                fi = pd.Series(mdl.feature_importances_, index=sel_features).sort_values(ascending=True)
                fig,ax = plt.subplots(figsize=(8,max(3,len(fi)*0.45)))
                ax.barh(fi.index, fi.values, color=PALETTE[:len(fi)], edgecolor="none")
                for i,(idx,v) in enumerate(fi.items()):
                    ax.text(v+0.001,i,f"{v:.3f}",va="center",fontsize=8,color=TEXT_COLOR)
                ax.set_title("Feature Importance"); apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()

            st.session_state["cls_model"]  = mdl
            st.session_state["cls_scaler"] = sc
            st.session_state["cls_feats"]  = sel_features
            st.session_state["cls_target"] = target_col
            st.session_state["cls_labels"] = class_labels
            st.session_state["cls_le"]     = le

        # Live Classifier
        if "cls_model" in st.session_state:
            st.markdown('<div class="section-title">⚡ Live Category Predictor</div>', unsafe_allow_html=True)
            inp = {}
            feats = st.session_state["cls_feats"]
            groups = [feats[i:i+4] for i in range(0,len(feats),4)]
            for grp in groups:
                cols_row = st.columns(len(grp))
                for c,f in zip(cols_row,grp):
                    mn=float(df_enc[f].min()); mx=float(df_enc[f].max()); mv=float(df_enc[f].mean())
                    inp[f] = c.number_input(f, min_value=mn, max_value=mx, value=round(mv,2), key=f"ci_{f}")
            if st.button("🔮 Classify Now", key="c_live"):
                inp_s   = st.session_state["cls_scaler"].transform(pd.DataFrame([inp]))
                pred_enc= st.session_state["cls_model"].predict(inp_s)[0]
                probas  = st.session_state["cls_model"].predict_proba(inp_s)[0]
                le_c    = st.session_state.get("cls_le")
                label   = le_c.inverse_transform([pred_enc])[0] if le_c else pred_enc
                conf    = probas.max()*100
                clabels = st.session_state["cls_labels"]

                st.markdown(f"""<div class="pred-result-box">
                  <p>Predicted {st.session_state['cls_target']}</p>
                  <h1>{label}</h1>
                  <p>Confidence: {conf:.1f}% | Model: {model_choice}</p></div>""", unsafe_allow_html=True)

                # Probability bars
                prob_df = pd.DataFrame({"Class":[str(c) for c in clabels],"Prob%":(probas*100).round(1)}).sort_values("Prob%",ascending=True)
                fig,ax = plt.subplots(figsize=(8,max(3,len(clabels)*0.5)))
                cols_p = [PALETTE[2] if str(c)==str(label) else PALETTE[0] for c in prob_df["Class"]]
                ax.barh(prob_df["Class"], prob_df["Prob%"], color=cols_p, edgecolor="none")
                ax.set_xlabel("Probability (%)"); ax.set_title("Class Probabilities")
                for i,v in enumerate(prob_df["Prob%"].values):
                    ax.text(v+0.5,i,f"{v:.1f}%",va="center",fontsize=9,color=TEXT_COLOR)
                apply_dark_style(fig,[ax]); st.pyplot(fig); plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div style="text-align:center;padding:40px 0 20px;color:#334155;font-size:0.8rem;">DataLens · Streamlit · Pandas · Matplotlib · Seaborn · Scikit-learn</div>', unsafe_allow_html=True)
