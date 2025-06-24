import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── User‐configurable boost settings ───────────────────────────────────────────
boost_list = [
    "ACWI", "AGG", "BRLN", "EFA", "EMB", "EMXC", "EMLC", "EZU", "EWG", "EWJ",
    "FLOT", "HYG", "IAGG", "IEF", "IWM", "LQD", "LVHI", "SLV", "SPY", "SPYD",
    "SPYV", "SPYG", "TIP", "TIPS", "VGSH", "XLI", "IWB", "IWM", "SPSM",
    "SPMD", "EEM", "QQQ", "SPTM", "RSP", "IWV"
]
boost_value = 0.01
# ────────────────────────────────────────────────────────────────────────────────

# 1. Load data
untracked = pd.read_excel("ETF_to_proxy.xlsx")
tracked   = pd.read_excel("Pave_ETFSMay2025.xlsx", sheet_name="Sheet1")

# 2. Preprocess text columns (normalize for exact matching)
cols = ["SECURITY-NAME", "SYMBOL", "Objective", "Bench", "Index", "Bnch ID"]
for df in (untracked, tracked):
    for c in cols:
        df[c] = (
            df[c]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.strip()
        )

# 3. Prepare semantic vectorization only for 'Objective'
vect = TfidfVectorizer()
combined_obj = pd.concat([untracked["Objective"], tracked["Objective"]], ignore_index=True)
vect.fit(combined_obj)
tfidf_obj = {
    "u": vect.transform(untracked["Objective"]),
    "t": vect.transform(tracked["Objective"])
}

# 4. Define the other columns for exact-match
other_cols = ["Index", "Bench", "SECURITY-NAME", "SYMBOL", "Bnch ID"]

n_u, n_t = untracked.shape[0], tracked.shape[0]

# ─── Phase 1: original weight scheme ───────────────────────────────────────────
weights_1 = {
    "Objective":       0.00,
    "Index":           0.50,
    "Bench":           0.50,
    "SECURITY-NAME":   0.00,
    "SYMBOL":          0.00,
    "Bnch ID":         0.00,
}

# compute sim1
sim1 = np.zeros((n_u, n_t))
# semantic part
sim1 += weights_1["Objective"] * cosine_similarity(tfidf_obj["u"], tfidf_obj["t"])
# exact-match parts
for col in other_cols:
    w = weights_1.get(col, 0)
    if w > 0:
        # broadcast comparison matrix: 1 if equal, else 0
        a = untracked[col].values[:, None]
        b = tracked[col].values[None, :]
        sim1 += w * (a == b).astype(float)

# apply boosts
for j, sym in enumerate(tracked["SYMBOL"]):
    if sym.upper() in boost_list:
        sim1[:, j] += boost_value

# build initial results
results = []
for i in range(n_u):
    ranked = np.argsort(sim1[i])[::-1]
    best_j, second_j = ranked[0], ranked[1]
    b1, b2 = sim1[i, best_j], sim1[i, second_j]

    if b1 > b2 and b1 > 0.5:
        name = tracked.loc[best_j, "SECURITY-NAME"]
        sym  = tracked.loc[best_j, "SYMBOL"]
        score = float(b1)
        boosted = sym.upper() in boost_list
    else:
        name = np.nan
        sym  = np.nan
        score = np.nan
        boosted = False

    results.append({
        "Untracked SECURITY-NAME":      untracked.loc[i, "SECURITY-NAME"],
        "Untracked SYMBOL":             untracked.loc[i, "SYMBOL"],
        "Phase1 Proxy SECURITY-NAME":   name,
        "Phase1 Proxy SYMBOL":          sym,
        "Phase1 Match Score":           score,
        "Phase1 Boost Applied":         boosted
    })

df = pd.DataFrame(results)

df.to_excel("ETF_proxy_matches_phase1.xlsx", index=False)
print("✅ Phase 1 results saved to ETF_proxy_matches_phase1.xlsx")

# ─── Phase 2: secondary weight scheme ──────────────────────────────────────────
weights_2 = {
    "Objective":       0.50,
    "Index":           0.25,
    "Bench":           0.25,
    "SECURITY-NAME":   0.00,
    "SYMBOL":          0.00,
    "Bnch ID":         0.00,
}

# compute sim2
sim2 = np.zeros((n_u, n_t))
sim2 += weights_2["Objective"] * cosine_similarity(tfidf_obj["u"], tfidf_obj["t"])
for col in other_cols:
    w = weights_2.get(col, 0)
    if w > 0:
        a = untracked[col].values[:, None]
        b = tracked[col].values[None, :]
        sim2 += w * (a == b).astype(float)
for j, sym in enumerate(tracked["SYMBOL"]):
    if sym.upper() in boost_list:
        sim2[:, j] += boost_value

# fill Phase2 only where Phase1 left NA
df["Phase2 Proxy SECURITY-NAME"] = np.nan
df["Phase2 Proxy SYMBOL"]        = np.nan
df["Phase2 Match Score"]         = np.nan

for i in range(n_u):
    if pd.isna(df.loc[i, "Phase1 Proxy SYMBOL"]):
        ranked = np.argsort(sim2[i])[::-1]
        b1, b2 = sim2[i, ranked[0]], sim2[i, ranked[1]]
        if b1 > b2 and b1 > 0.5:
            df.loc[i, "Phase2 Proxy SECURITY-NAME"] = tracked.loc[ranked[0], "SECURITY-NAME"]
            df.loc[i, "Phase2 Proxy SYMBOL"]        = tracked.loc[ranked[0], "SYMBOL"]
            df.loc[i, "Phase2 Match Score"]         = float(b1)

df.to_excel("ETF_proxy_matches_phase2.xlsx", index=False)
print("✅ Phase 2 saved to ETF_proxy_matches_phase2.xlsx")

# ─── Phase 3: objective-only semantic ─────────────────────────────────────────
weights_3 = {
    "Objective":       0.70,
    "Index":           0.15,
    "Bench":           0.15,
    "SECURITY-NAME":   0.00,
    "SYMBOL":          0.00,
    "Bnch ID":         0.00,
}

# compute sim3
sim3 = np.zeros((n_u, n_t))
sim3 += weights_3["Objective"] * cosine_similarity(tfidf_obj["u"], tfidf_obj["t"])
for j, sym in enumerate(tracked["SYMBOL"]):
    if sym.upper() in boost_list:
        sim3[:, j] += boost_value

# fill Phase3 only where Phases 1 & 2 are NA
df["Phase3 Proxy SECURITY-NAME"] = np.nan
df["Phase3 Proxy SYMBOL"]        = np.nan
df["Phase3 Match Score"]         = np.nan

for i in range(n_u):
    if pd.isna(df.loc[i, "Phase1 Proxy SYMBOL"]) and pd.isna(df.loc[i, "Phase2 Proxy SYMBOL"]):
        ranked = np.argsort(sim3[i])[::-1]
        c1, c2 = sim3[i, ranked[0]], sim3[i, ranked[1]]
        if c1 > c2:
            df.loc[i, "Phase3 Proxy SECURITY-NAME"] = tracked.loc[ranked[0], "SECURITY-NAME"]
            df.loc[i, "Phase3 Proxy SYMBOL"]        = tracked.loc[ranked[0], "SYMBOL"]
            df.loc[i, "Phase3 Match Score"]         = float(c1)

df.to_excel("ETF_proxy_matches_phase3.xlsx", index=False)
print("✅ Phase 3 saved to ETF_proxy_matches_phase3.xlsx")
