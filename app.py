import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy import stats
from scipy.stats import chi2_contingency
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Headline Engagement Analysis", layout="wide")

DATA_DIR = "data/"

@st.cache_data
def load_data():
    def load_and_clean(path, label):
        df = pd.read_csv(path, index_col=0, low_memory=False)
        df["source"] = label
        df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce")
        df["clicks"]      = pd.to_numeric(df["clicks"],      errors="coerce")
        df = df.dropna(subset=["headline", "impressions", "clicks"])
        df = df[df["impressions"] > 0]
        df["ctr"] = df["clicks"] / df["impressions"]
        return df

    exp  = load_and_clean(DATA_DIR + "upworthy-archive-exploratory-packages-03.12.2020.csv",  "exploratory")
    conf = load_and_clean(DATA_DIR + "upworthy-archive-confirmatory-packages-03.12.2020.csv", "confirmatory")

    POSITIVE_WORDS = {"amazing","awesome","beautiful","best","brilliant","celebrate","champion",
        "courage","delight","excellent","fantastic","good","great","happy","hero",
        "hope","incredible","inspiring","joy","love","outstanding","perfect",
        "positive","proud","remarkable","stunning","success","superb","triumph",
        "wonderful","win","winning"}
    NEGATIVE_WORDS = {"abuse","awful","bad","crisis","danger","death","destroy","disaster",
        "evil","fail","failure","fear","hate","horrible","kill","loss","lost",
        "murder","nightmare","pain","problem","sad","scary","shocking","suffer",
        "terrible","tragedy","tragic","violence","war","worst","wrong"}

    for df in [exp, conf]:
        df["has_number"]  = df["headline"].apply(lambda x: bool(re.search(r'\b\d+\b', str(x))))
        df["is_question"] = df["headline"].apply(lambda x: str(x).strip().endswith("?"))
        df["word_count"]  = df["headline"].apply(lambda x: len(str(x).split()))
        df["length_cat"]  = pd.cut(df["word_count"], bins=[0,8,14,100],
                                   labels=["Short (≤8)", "Medium (9–14)", "Long (15+)"])
        def sentiment(text):
            words = set(re.findall(r'\b[a-z]+\b', str(text).lower()))
            pos = len(words & POSITIVE_WORDS)
            neg = len(words & NEGATIVE_WORDS)
            return "positive" if pos > neg else ("negative" if neg > pos else "neutral")
        df["sentiment"] = df["headline"].apply(sentiment)

    return exp, conf

def two_prop_ztest(df, col, val_true=True, val_false=False):
    g1 = df[df[col] == val_true]
    g2 = df[df[col] == val_false]
    c1, n1 = g1["clicks"].sum(), g1["impressions"].sum()
    c2, n2 = g2["clicks"].sum(), g2["impressions"].sum()
    p1, p2 = c1/n1, c2/n2
    p_pool = (c1+c2)/(n1+n2)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    z  = (p1-p2)/se
    p  = 2*(1-stats.norm.cdf(abs(z)))
    return p1, p2, z, p, (p1-p2)/p2*100

def sig_label(p):
    return "p < 0.001 ***" if p < 0.001 else ("p < 0.01 **" if p < 0.01 else (f"p = {p:.4f} *" if p < 0.05 else f"p = {p:.4f}"))

exp, conf = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Jump to section", [
    "1. Background & Overview",
    "2. Data Structure Overview",
    "3. Executive Summary",
    "4. Insights Deep Dive",
    "5. Recommendations",
    "6. Code"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset filter**")
dataset_choice = st.sidebar.radio(
    "Active dataset (applies to Section 4)",
    ["Exploratory", "Confirmatory", "Both"],
    help="Exploratory tests generate hypotheses. Confirmatory tests validate them."
)

def get_df():
    if dataset_choice == "Exploratory":  return exp
    if dataset_choice == "Confirmatory": return conf
    return pd.concat([exp, conf])

df = get_df()

# ══════════════════════════════════════════════════════════════════════════════
# 1. BACKGROUND & OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

st.title("What Makes a Headline People Click?")
st.markdown("*Analyzing 150,000+ A/B headline tests from Upworthy (2013–2015)*")

st.markdown('<p style="color:#4C78A8; font-weight:700; font-size:1.05rem;">Key Recommendations</p>', unsafe_allow_html=True)
col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
col_r1.success("**Avoid question headlines** \n\nThey reduce CTR by 13-15%")
col_r2.success("**Write longer headlines** \n\n15+ words outperform short ones")
col_r3.success("**Lead with urgency** \n\nNegative framing outperforms positive")
col_r4.success("**Invest in headline testing** \n\nCopy drives CTR, not the image")
col_r5.warning("**Numbers are neutral** \n\nNo reliable effect in either direction")

st.divider()

st.header("1. Background & Overview")

st.markdown("""
Have you ever wondered why some news headlines immediately grab your attention while others
get scrolled past in seconds? What if editorial teams could reliably predict which headline
would drive more clicks before running a single test? How much revenue and readership is
lost every day by teams guessing at headline strategy instead of relying on evidence?
""")

st.markdown('<p style="color:#4C78A8; font-weight:700; font-size:1.05rem;">About Upworthy</p>', unsafe_allow_html=True)
st.markdown("""
Upworthy is a media company known for its socially conscious, viral content. Between 2013 and
2015, the editorial team ran thousands of randomized A/B experiments:

- Different headline variants for the same article were shown to separate groups of real readers
- The version that generated more clicks was declared the winner
- Over two years, this process produced one of the richest headline testing datasets ever made public
""")

st.markdown('<p style="color:#4C78A8; font-weight:700; font-size:1.05rem;">Core Question</p>', unsafe_allow_html=True)
st.info("Which headline characteristics (numbers, question framing, length, or emotional tone) reliably drive higher click-through rates, and are those patterns strong enough to act on?")

st.markdown('<p style="color:#4C78A8; font-weight:700; font-size:1.05rem;">Key Metric: Click-Through Rate (CTR)</p>', unsafe_allow_html=True)
st.markdown("""
- Defined as the share of people who saw a headline and clicked on it
- A higher CTR means a headline is more compelling to readers
- Even small CTR improvements compound at scale: a 10% lift across millions of monthly readers translates directly into more traffic, ad revenue, and mission impact
""")

st.markdown('<p style="color:#4C78A8; font-weight:700; font-size:1.05rem;">How This Analysis Is Structured</p>', unsafe_allow_html=True)
st.markdown("""
▸ **Stage 1 (Exploratory experiments):** identify candidate patterns in reader behavior

▸ **Stage 2 (Confirmatory experiments):** independently validate whether those patterns hold up

Only findings that replicate in both stages are treated as actionable.
""")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA STRUCTURE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("2. Data Structure Overview")

st.markdown("""
The dataset comes from the **Upworthy Research Archive**, a public release of randomized headline experiments.
It contains four files; two are the focus of this analysis:

- **Exploratory dataset:** experiments run freely to discover what works, used for hypothesis generation
- **Confirmatory dataset:** experiments run to validate earlier findings, held to a higher evidential standard

A finding that appears in both datasets is far more trustworthy than one from a single dataset.
The remaining two files (holdout and undeployed) are available but not the primary focus here.

Each row represents one headline variant tested in an A/B experiment. The table below describes the key fields:
""")

schema = pd.DataFrame({
    "Field": ["headline", "impressions", "clicks", "clickability_test_id", "eyecatcher_id"],
    "Type": ["Text", "Integer", "Integer", "String (ID)", "String (ID)"],
    "Description": [
        "The full text of the tested headline variant",
        "Number of times this headline was shown to readers",
        "Number of times readers clicked through after seeing this headline",
        "Unique experiment ID that links all variants tested in the same A/B test",
        "ID of the image shown alongside the headline, used to isolate text-only effects"
    ]
})
st.dataframe(schema, use_container_width=True, hide_index=True)

st.markdown('<p style="color:#4C78A8; font-weight:700; font-size:1.05rem;">Derived Features</p>', unsafe_allow_html=True)
st.markdown("""
The following new columns were engineered from the raw data:

| Feature | Description |
|---|---|
| `has_number` | True if the headline contains any standalone digit |
| `is_question` | True if the headline ends with a "?" |
| `word_count` | Total number of words in the headline |
| `length_cat` | Short (≤8 words), Medium (9–14), or Long (15+) |
| `sentiment` | Positive, negative, or neutral based on a word lexicon |
| `ctr` | Computed as `clicks / impressions` |
""")

st.markdown('<p style="color:#4C78A8; font-weight:700; font-size:1.05rem;">Data Quality Steps</p>', unsafe_allow_html=True)
st.markdown("""
▸ Removed rows with missing headline text, zero impressions, or non-numeric values

▸ No imputation performed the dataset is large enough that dropping incomplete rows does not meaningfully reduce coverage
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Exploratory headlines", f"{len(exp):,}")
col2.metric("Confirmatory headlines", f"{len(conf):,}")
col3.metric("Total A/B experiments", f"{pd.concat([exp,conf])['clickability_test_id'].nunique():,}")
col4.metric("Overall median CTR", f"{pd.concat([exp,conf])['ctr'].median():.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("3. Executive Summary")

st.markdown("""
The analysis tested six hypotheses about headline effectiveness. The table below summarizes findings across both datasets.

▸ **Replicates:** appeared in the same direction in both datasets, safe to act on

▸ **Inconclusive:** effect flips direction between datasets, not reliable enough to act on
""")

summary_data = pd.DataFrame({
    "Question": [
        "Do numbers improve CTR?",
        "Do question headlines improve CTR?",
        "Do longer headlines outperform shorter?",
        "Does negative sentiment outperform positive?",
        "Does headline text drive CTR (vs. image)?",
        "Do exploratory findings replicate?"
    ],
    "Finding": [
        "Mixed: effect flips direction across datasets",
        "Questions hurt CTR by ~13-15%",
        "Longer headlines (15+ words) outperform short ones (<=8)",
        "Negative framing outperforms positive framing",
        "Text alone drives significant CTR variation",
        "Questions, length, sentiment replicate; numbers do not"
    ],
    "Status": [
        "Inconclusive",
        "Replicates",
        "Replicates",
        "Replicates",
        "Replicates",
        "Partial replication"
    ]
})
st.dataframe(summary_data, use_container_width=True, hide_index=True)

st.markdown('<p style="color:#4C78A8; font-weight:700; font-size:1.05rem;">Top 3 Actionable Findings for Editorial Teams</p>', unsafe_allow_html=True)
st.markdown("""
1. **Avoid question framing:** a 13-15% CTR penalty is one of the strongest and most consistent effects in the data; default to declarative headlines

2. **Write longer, not shorter:** headlines with 15+ words consistently outperform short ones, contradicting conventional advice

3. **Lead with urgency, not inspiration:** negative or urgent framing outperforms uplifting framing on this platform
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4. INSIGHTS DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("4. Insights Deep Dive")
st.markdown(f"*Using dataset: **{dataset_choice}** change in the sidebar to compare.*")

# ── Q1: Numbers ───────────────────────────────────────────────────────────────
st.subheader("1. Do Headlines with Numbers Get More Clicks?")

st.markdown("""
**Background**

- "Listicle" headlines like *"7 Things You Didn't Know About..."* became a defining format for viral content in the early 2010s
- The theory: numbers signal specificity and set clear reader expectations, lowering the perceived risk of clicking

**Method**

- Headlines were tagged as containing a number if they included any standalone digit (e.g., "5", "100")
- CTR was compared between groups using a **two-proportion z-test**, which tests whether the difference is statistically distinguishable from random chance

**How to Read the Chart**

▸ Each bar shows pooled CTR (total clicks divided by total impressions) for that group

▸ A higher bar means more clicks per impression on average
""")

p1, p2, z, p, lift = two_prop_ztest(df, "has_number")
fig1 = px.bar(
    x=["With Number", "Without Number"], y=[p1*100, p2*100],
    labels={"x": "", "y": "CTR (%)"},
    color=["With Number", "Without Number"],
    color_discrete_map={"With Number": "#4C78A8", "Without Number": "#9ECAE9"},
    text=[f"{p1:.2%}", f"{p2:.2%}"],
)
fig1.update_traces(textposition="outside")
fig1.update_layout(showlegend=False, yaxis_range=[0, max(p1,p2)*100*1.3],
                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig1, use_container_width=True)
st.caption(f"Relative lift: **{lift:+.1f}%** | z = {z:.2f} | {sig_label(p)}")

st.markdown("""
**Interpretation**

- Exploratory dataset: slight negative lift (−2.1%)
- Confirmatory dataset: slight positive lift (+1.1%)
- Both results are statistically significant due to large sample sizes, but the direction flips between datasets

◦ Whether numbers help or hurt likely depends on context topic, framing, and audience not on the digit itself

◦ **This finding should not drive headline strategy**
""")

st.divider()

# ── Q2: Questions ─────────────────────────────────────────────────────────────
st.subheader("2. Do Question Headlines Drive More Clicks?")

st.markdown("""
**Background**

- Question headlines are a common editorial tactic based on the "curiosity gap" theory
- The idea: an open question compels the reader to click just to find the answer
- Example: *"Could This One Food Be Causing Your Fatigue?"* creates an itch only the article can scratch

**Method**

- Headlines ending with "?" were classified as questions
- CTR was compared using a two-proportion z-test

**How to Read the Chart**

▸ Each bar shows pooled CTR for that group

▸ The gap between bars, combined with the z-test, tells us whether the effect is real or random noise
""")

p1, p2, z, p, lift = two_prop_ztest(df, "is_question")
fig2 = px.bar(
    x=["Question", "Non-Question"], y=[p1*100, p2*100],
    labels={"x": "", "y": "CTR (%)"},
    color=["Question", "Non-Question"],
    color_discrete_map={"Question": "#E45756", "Non-Question": "#72B7B2"},
    text=[f"{p1:.2%}", f"{p2:.2%}"],
)
fig2.update_traces(textposition="outside")
fig2.update_layout(showlegend=False, yaxis_range=[0, max(p1,p2)*100*1.3],
                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig2, use_container_width=True)
st.caption(f"Relative lift: **{lift:+.1f}%** | z = {z:.2f} | {sig_label(p)}")

st.markdown("""
**Interpretation**

- Question headlines consistently underperform by 13–15% across both datasets
- This is one of the clearest and most reliable findings in the entire analysis

◦ The curiosity-gap theory does not hold up in practice for this audience

◦ Readers respond better to confident, direct statements than open questions

◦ **Avoid question framing as a default strategy**
""")

st.divider()

# ── Q3: Length ────────────────────────────────────────────────────────────────
st.subheader("3. Are Shorter Headlines More Effective?")

st.markdown("""
**Background**

Conventional content wisdom recommends short headlines because they are:
- Easier to scan quickly
- Less likely to be truncated on mobile
- Less demanding on the reader

The question is whether brevity actually translates into more clicks.

**Method**

- Headlines were grouped into three buckets: Short (≤8 words), Medium (9–14), Long (15+)
- CTR was compared using the **Kruskal-Wallis test**, a non-parametric equivalent of ANOVA
- This test is appropriate here because CTR distributions are heavily right-skewed, violating standard ANOVA assumptions

**How to Read the Chart**

▸ Each bar shows pooled CTR for that length category

▸ The Kruskal-Wallis result tells us whether at least one group is significantly different from the others
""")

length_summary = (df.groupby("length_cat", observed=True)
                    .apply(lambda g: pd.Series({
                        "Pooled CTR": g["clicks"].sum() / g["impressions"].sum() * 100,
                        "N": len(g)
                    })).reset_index())
length_summary.columns = ["Length", "Pooled CTR (%)", "N"]

fig3 = px.bar(length_summary, x="Length", y="Pooled CTR (%)",
              color="Length",
              color_discrete_sequence=["#F4845F","#72B7B2","#4C78A8"],
              text=length_summary["Pooled CTR (%)"].apply(lambda x: f"{x:.2f}%"))
fig3.update_traces(textposition="outside")
fig3.update_layout(showlegend=False,
                   yaxis_range=[0, length_summary["Pooled CTR (%)"].max()*1.3],
                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig3, use_container_width=True)

groups = [g["ctr"].values for _, g in df.groupby("length_cat", observed=True)]
h, p_kw = stats.kruskal(*groups)
st.caption(f"Kruskal-Wallis H = {h:.2f} | {sig_label(p_kw)}")

st.markdown("""
**Interpretation**

- Headlines with 15+ words achieve the highest CTR in both datasets
- Headlines with ≤8 words consistently perform worst
- The effect is statistically significant (p < 0.001) and replicates across datasets

◦ Upworthy's audience responds to context-rich, story-driven headlines with enough detail to justify the click

◦ Very short headlines may feel vague or low-information by comparison

◦ **Writers should not artificially shorten headlines in the name of brevity**
""")

st.divider()

# ── Q4: Sentiment ─────────────────────────────────────────────────────────────
st.subheader("4. Does Emotional Tone Affect Engagement?")

st.markdown("""
**Background**

- Emotional framing is central to viral content strategy
- The core question: does positive or negative tone work better?
- This has real editorial implications it determines whether writers frame stories around hope and inspiration or urgency and alarm

**Method**

- Each headline was scored using a curated lexicon of positive and negative words
- Headlines were labeled based on which tone dominated: positive, negative, or neutral
- CTR differences were tested using a **chi-square test** on the full clicks vs. non-clicks contingency table

**How to Read the Chart**

▸ Each bar shows pooled CTR for that sentiment category

▸ The chi-square test tells us whether click distributions differ significantly across sentiment groups
""")

sent_summary = (df.groupby("sentiment")
                  .apply(lambda g: pd.Series({
                      "Pooled CTR": g["clicks"].sum() / g["impressions"].sum() * 100,
                      "N": len(g)
                  })).reset_index())
sent_summary.columns = ["Sentiment", "Pooled CTR (%)", "N"]
sent_summary["Sentiment"] = sent_summary["Sentiment"].str.capitalize()
order = {"Negative": 0, "Neutral": 1, "Positive": 2}
sent_summary = sent_summary.sort_values("Sentiment", key=lambda x: x.map(order))

fig4 = px.bar(sent_summary, x="Sentiment", y="Pooled CTR (%)",
              color="Sentiment",
              color_discrete_map={"Negative":"#E45756","Neutral":"#9ECAE9","Positive":"#54A24B"},
              text=sent_summary["Pooled CTR (%)"].apply(lambda x: f"{x:.2f}%"))
fig4.update_traces(textposition="outside")
fig4.update_layout(showlegend=False,
                   yaxis_range=[0, sent_summary["Pooled CTR (%)"].max()*1.3],
                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig4, use_container_width=True)

ct = df.groupby("sentiment")[["clicks","impressions"]].sum()
ct["non_clicks"] = ct["impressions"] - ct["clicks"]
chi2_val, p_chi, dof, _ = chi2_contingency(ct[["clicks","non_clicks"]].values)
st.caption(f"Chi-square = {chi2_val:.2f}, df = {dof} | {sig_label(p_chi)}")

st.markdown("""
**Interpretation**

- CTR ranking is consistently: **Negative > Neutral > Positive** in both datasets (p < 0.001)
- Negative-sentiment headlines outperform positive ones

◦ This makes sense given Upworthy's editorial identity the platform focuses on social injustice, inequality, and underreported crises

◦ Its audience is primed to engage with alarming, urgent content; positive framing may feel out of place

◦ **This finding may not generalize to other platforms** a lifestyle or wellness brand might see the opposite effect
""")

st.divider()

# ── Q5: Text vs Image ──────────────────────────────────────────────────────────
st.subheader("5. Is It the Headline or the Image Driving Clicks?")

st.markdown("""
**Background**

- Before attributing CTR differences to headline copy, we need to rule out the image as the real driver
- If images explained most of the variation, optimizing headlines would be the wrong strategy

**Method**

- Filtered for experiments where all headline variants shared the exact same image (same `eyecatcher_id`)
- Measured how much CTR varied across headline variants within each experiment using standard deviation
- If images drove all variation, variance would be near zero every variant with the same image would perform identically
- A **one-sample t-test** tested whether the observed spread is significantly greater than zero

**How to Read the Chart**

▸ The histogram shows the distribution of within-experiment CTR spread

▸ A spike at zero would mean images do all the work; spread shifted right means text matters
""")

same_img = (df.groupby("clickability_test_id")
              .filter(lambda g: g["eyecatcher_id"].nunique() == 1 and len(g) >= 2))
per_exp = same_img.groupby("clickability_test_id")["ctr"].std().dropna()
t, p_t = stats.ttest_1samp(per_exp, 0)

fig5 = px.histogram(per_exp, nbins=60,
                    labels={"value": "Within-Experiment CTR Std Dev", "count": "Number of Experiments"},
                    color_discrete_sequence=["#4C78A8"])
fig5.update_layout(showlegend=False,
                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig5, use_container_width=True)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Same-image experiments", f"{len(per_exp):,}")
col_b.metric("Median CTR spread (std dev)", f"{per_exp.median():.4f}")
col_c.metric("t-statistic", f"{t:.1f}")
st.caption(f"One-sample t-test vs 0: t = {t:.2f} | {sig_label(p_t)}")

st.markdown("""
**Interpretation**

- The histogram is clearly shifted away from zero most experiments show meaningful CTR variation even with a fixed image
- The t-test confirms this spread is statistically significant (t > 70, p ≈ 0)

◦ **Headline copy materially affects click rates, independent of the image**

◦ Image quality matters, but it is not the primary engagement lever on this platform

◦ Resources invested in headline A/B testing are well spent
""")

st.divider()

# ── Q6: Replication ────────────────────────────────────────────────────────────
st.subheader("6. Do Exploratory Findings Hold Up in Confirmatory Data?")

st.markdown("""
**Background**

- A pattern found once might be noise
- In data science, as in medical research, a finding that replicates in an independent dataset is far more credible and safer to act on
- This section checks whether effects from exploratory experiments are consistent with confirmatory experiments

**How to Read the Tables**

▸ "Same direction?" tells you whether the effect went the same way in both datasets

▸ The lift columns show how large the effect was in each dataset

▸ Both direction and magnitude matter an effect that replicates directionally but shrinks dramatically may not be worth acting on
""")

rep_data = []
for fname, fcol in [("Numbers", "has_number"), ("Questions", "is_question")]:
    p1e, p2e, ze, pe, lifte = two_prop_ztest(exp,  fcol)
    p1c, p2c, zc, pc, liftc = two_prop_ztest(conf, fcol)
    rep_data.append({
        "Feature": fname,
        "Exploratory lift": f"{lifte:+.1f}%",
        "Confirmatory lift": f"{liftc:+.1f}%",
        "Same direction?": "Yes" if np.sign(lifte)==np.sign(liftc) else "No",
        "Sig (Exp)":  "***" if pe < 0.001 else ("*" if pe < 0.05 else "ns"),
        "Sig (Conf)": "***" if pc < 0.001 else ("*" if pc < 0.05 else "ns"),
        "Verdict": "Inconclusive" if np.sign(lifte)!=np.sign(liftc) else "Replicates"
    })
st.dataframe(pd.DataFrame(rep_data), use_container_width=True, hide_index=True)

st.markdown("**Sentiment and length replication:**")
col_s, col_l = st.columns(2)

with col_s:
    sent_rep = []
    for sent in ["positive", "negative", "neutral"]:
        ctr_e = exp[exp["sentiment"]==sent]["clicks"].sum() / exp[exp["sentiment"]==sent]["impressions"].sum()
        ctr_c = conf[conf["sentiment"]==sent]["clicks"].sum() / conf[conf["sentiment"]==sent]["impressions"].sum()
        sent_rep.append({"Sentiment": sent.capitalize(), "Exploratory CTR": f"{ctr_e:.4f}", "Confirmatory CTR": f"{ctr_c:.4f}"})
    st.dataframe(pd.DataFrame(sent_rep), use_container_width=True, hide_index=True)

with col_l:
    len_rep = []
    for cat in ["Short (≤8)", "Medium (9–14)", "Long (15+)"]:
        ctr_e = exp[exp["length_cat"]==cat]["clicks"].sum() / exp[exp["length_cat"]==cat]["impressions"].sum()
        ctr_c = conf[conf["length_cat"]==cat]["clicks"].sum() / conf[conf["length_cat"]==cat]["impressions"].sum()
        len_rep.append({"Length": cat, "Exploratory CTR": f"{ctr_e:.4f}", "Confirmatory CTR": f"{ctr_c:.4f}"})
    st.dataframe(pd.DataFrame(len_rep), use_container_width=True, hide_index=True)

st.markdown("""
**Interpretation**

Findings that replicate:
- Question penalty (13–15% CTR loss): same direction and similar magnitude in both datasets
- Length gradient (longer is better): consistent ordering across both datasets
- Sentiment ordering (negative > neutral > positive): holds in both datasets

Finding that does not replicate:
- Numbers effect: direction flips between datasets; treat as inconclusive

◦ 3 out of 4 findings replicate, increasing confidence that these are genuine patterns in reader behavior rather than statistical artifacts
""")

# ══════════════════════════════════════════════════════════════════════════════
# 5. RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("5. Recommendations")

st.markdown("""
The following recommendations are offered to Upworthy's editorial and growth teams.
Each is grounded in statistically significant, replicated evidence not single-dataset findings.
""")

st.subheader("1. Stop Writing Question Headlines")
st.markdown("""
**Evidence**
- Question headlines produce 13–15% lower CTR than declarative headlines
- This is the single largest and most reliable effect in the dataset
- Replicates across both exploratory and confirmatory experiments with near-identical magnitude

**Action**

▸ Remove question headlines as a default template

▸ When a writer frames something as a question, rephrase it as a statement:
  - Before: *"Is This the Most Important Climate Story of the Year?"*
  - After: *"This May Be the Most Important Climate Story of the Year"*

▸ The content does not change only the framing does
""")

st.subheader("2. Write Longer, More Descriptive Headlines")
st.markdown("""
**Evidence**
- Headlines with 15+ words outperform short headlines (≤8 words) in both datasets
- The relationship is monotonic: more words, more clicks
- Counterintuitive but robust and replicated

**Action**

▸ Encourage writers to include context, stakes, or a key detail rather than truncating for brevity

▸ Length should come from information density, not padding:
  - Weak: *"A Big Supreme Court Decision"*
  - Strong: *"The Supreme Court Just Made a Decision That Will Affect Every Renter in America"*
""")

st.subheader("3. Lead with Urgency and Stakes, Not Inspiration")
st.markdown("""
**Evidence**
- Headlines signaling urgency, injustice, or alarm consistently outperform positive framing
- Effect replicates across both datasets
- Reflects the expectation Upworthy's audience brings to the platform

**Action**

▸ When two framings are equally accurate, prefer the one that emphasizes stakes or consequences

▸ This is about matching the emotional register of the audience, not being misleading

▸ Note: this recommendation is platform-specific and should be re-evaluated if the editorial direction shifts
""")

st.subheader("4. Invest in Headline Testing, Not Just Image Selection")
st.markdown("""
**Evidence**
- Headline copy independently drives significant CTR variation even when the image is fixed
- A compelling image does not compensate for a weak headline

**Action**

▸ Maintain or expand the headline A/B testing program

▸ Treat headline copy as a first-class optimization lever, not an afterthought after image selection
""")

st.subheader("5. Treat Numbers as a Neutral Feature")
st.markdown("""
**Evidence**
- No strong effect in either direction effect flips between exploratory and confirmatory datasets
- Likely context-dependent rather than a reliable tactic

**Action**

▸ Do not add numbers to headlines purely as a conversion tactic

▸ Use numbers when they add genuine specificity (a real statistic, a meaningful count)

▸ Avoid them when they feel forced the data gives no reason to systematically favor or avoid them
""")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CODE
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("6. Code")

st.markdown("""
The core statistical methods used in this analysis are shown below.
Full source code is available in `analysis.py` (batch script) and `app.py` (this dashboard).
""")

st.subheader("Data Loading & Feature Engineering")
st.code("""
import pandas as pd
import numpy as np
import re

def load_and_clean(path, label):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce")
    df["clicks"]      = pd.to_numeric(df["clicks"],      errors="coerce")
    df = df.dropna(subset=["headline", "impressions", "clicks"])
    df = df[df["impressions"] > 0]
    df["ctr"] = df["clicks"] / df["impressions"]
    return df

# Feature engineering
df["has_number"]  = df["headline"].apply(lambda x: bool(re.search(r'\\b\\d+\\b', str(x))))
df["is_question"] = df["headline"].apply(lambda x: str(x).strip().endswith("?"))
df["word_count"]  = df["headline"].apply(lambda x: len(str(x).split()))
df["length_cat"]  = pd.cut(df["word_count"], bins=[0, 8, 14, 100],
                            labels=["Short (≤8)", "Medium (9–14)", "Long (15+)"])
""", language="python")

st.subheader("Two-Proportion Z-Test")
st.code("""
from scipy import stats

def two_prop_ztest(df, col, val_true=True, val_false=False):
    g1 = df[df[col] == val_true]
    g2 = df[df[col] == val_false]
    c1, n1 = g1["clicks"].sum(), g1["impressions"].sum()
    c2, n2 = g2["clicks"].sum(), g2["impressions"].sum()
    p1, p2 = c1 / n1, c2 / n2
    p_pool = (c1 + c2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z  = (p1 - p2) / se
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    lift = (p1 - p2) / p2 * 100
    return p1, p2, z, p, lift

# Example: test whether question headlines differ from non-questions
ctr_q, ctr_nonq, z, p, lift = two_prop_ztest(df, "is_question")
print(f"Question CTR: {ctr_q:.4f} | Non-question CTR: {ctr_nonq:.4f}")
print(f"Lift: {lift:+.1f}% | z = {z:.2f} | p = {p:.4e}")
""", language="python")

st.subheader("Chi-Square Test for Sentiment")
st.code("""
from scipy.stats import chi2_contingency

# Build contingency table: clicks vs. non-clicks by sentiment group
ct = df.groupby("sentiment")[["clicks", "impressions"]].sum()
ct["non_clicks"] = ct["impressions"] - ct["clicks"]
table = ct[["clicks", "non_clicks"]].values

chi2, p, dof, expected = chi2_contingency(table)
print(f"Chi-square = {chi2:.2f}, df = {dof}, p = {p:.4e}")
""", language="python")

st.subheader("Kruskal-Wallis Test for Headline Length")
st.code("""
# Non-parametric ANOVA appropriate for skewed CTR distributions
groups = [g["ctr"].values for _, g in df.groupby("length_cat", observed=True)]
h, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis H = {h:.2f}, p = {p:.4e}")
""", language="python")

st.subheader("Text-Only Effect (Same Image Isolation)")
st.code("""
# Filter for experiments where all variants share the same image
same_img = (df.groupby("clickability_test_id")
              .filter(lambda g: g["eyecatcher_id"].nunique() == 1 and len(g) >= 2))

# Measure within-experiment CTR spread
per_exp = same_img.groupby("clickability_test_id")["ctr"].std().dropna()

# Test whether spread is significantly above zero
t, p = stats.ttest_1samp(per_exp, 0)
print(f"Median CTR std dev: {per_exp.median():.4f}")
print(f"t = {t:.2f}, p = {p:.4e}")
""", language="python")

st.divider()
st.header("7. How I Used Claude Code in This Analysis")

st.markdown("AI tools are only as good as the analyst directing them. Here is where Claude Code added real leverage, and where human judgment still mattered.")

st.markdown('<p style="color:#4C78A8; font-weight:700; font-size:1.05rem;">What Claude Code Did</p>', unsafe_allow_html=True)
st.markdown("""
▸ **Pipeline scaffolding.** Full data loading, cleaning, feature engineering, and statistical testing code generated in one pass. Hours of boilerplate compressed into minutes.

▸ **Test selection.** Claude matched the right test to each question: z-test for binary comparisons, Kruskal-Wallis for skewed distributions, chi-square for contingency tables.

▸ **App development.** The entire Streamlit dashboard (layout, charts, sidebar, metric cards) was built and refined through conversation with no manual front-end work.

▸ **Rapid iteration.** Structural edits across 800 lines of code (reformatting, reordering, rewording) executed in seconds.
""")

st.markdown('<p style="color:#E45756; font-weight:700; font-size:1.05rem;">Where I Applied Critical Judgment</p>', unsafe_allow_html=True)
st.markdown("""
▸ **Caught a flawed summary.** Claude's first draft said numbers improve CTR in both datasets. The data showed the effect flipping direction. I flagged it and forced a correction. AI summarizes what it expects to find if you do not verify the output.

▸ **Validated test choices.** Kruskal-Wallis was appropriate given right-skewed CTR distributions. I confirmed this rather than accepting it blindly; the wrong test would have produced misleading p-values.

▸ **Wrote the business interpretation.** Claude has no knowledge of Upworthy's audience or editorial identity. Every "why this finding makes sense" section was written by me.

▸ **Owned the narrative structure.** The five-section report design came from deliberate choices about what a business audience needs to see. Claude executed the structure; the judgment was mine.
""")

st.markdown('<p style="color:#54A24B; font-weight:700; font-size:1.05rem;">The Bottom Line</p>', unsafe_allow_html=True)
st.markdown("""
Claude Code cut the time from raw dataset to polished interactive analysis from days to hours.
But that speed was only possible because each prompt was precise: specifying the test, the comparison, the business question.
Vague prompts produce generic output.

The real risk with AI-assisted analysis is not wrong code. It is an unquestioned summary that looks plausible.
Every finding here was verified against raw statistical output before being reported. That step is non-negotiable.
""")

st.divider()
st.markdown("""
**Data:** Upworthy Research Archive (2013-2015) ·
**Methods:** Two-proportion z-test, Chi-square, Kruskal-Wallis, One-sample t-test ·
**Built with** Streamlit, Plotly and Claude Code
""")
