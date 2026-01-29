############################################################
# M&A FORECAST DATASET PIPELINE (Final Surgical Master)
#
# GOAL:
#   1. Build Firm-Quarter Panel (Compustat) with STD filtering.
#   2. PIT-RISK FLAG: Detect ANY change in 'AJEXQ/AJEX/CFAC*'.
#      (Used to flag potential back-adjustment leakage).
#   3. SCRUB LEAKAGE: Drop 'CFAC/AJEX' cols from feature set.
#      *PROTECT*: 'PRCCQ', 'MKVALTQ' & the new Risk Flag.
#   4. SNAPSHOT: Max(RDQ, Datadate + 90d).
#      (Ensures strict 90d safe harbor, but respects late filings).
#   5. LABELS (Horizon relative to Snapshot):
#      - label_deal_0_3m: Deal Announced in (Snapshot, Snapshot + 3mo]
#      - label_deal_0_6m: Deal Announced in (Snapshot, Snapshot + 6mo]
#   6. VERIFICATION: Event Study to prove Price jumps at Announcement.
############################################################

import os, glob, re, time, warnings, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

############################################################
# 0. Setup & Paths
############################################################

try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive"
except Exception:
    BASE_DIR = os.getcwd()

log(f"BASE_DIR = {BASE_DIR}")

FUNDQ_PATH = os.path.join(BASE_DIR, "fundq_full.parquet")
FUNDA_PATH = os.path.join(BASE_DIR, "funda_full.parquet")
OUTPUT_FILENAME = "labeled_mna_pit_panel.parquet"
SAVE_PATH = os.path.join(BASE_DIR, OUTPUT_FILENAME)

############################################################
# 1. Load & Filter Compustat (The "Clean Start")
############################################################

def load_and_prep_compustat(q_path, a_path):
    # Load Quarterly
    if os.path.isfile(q_path):
        q_df = pd.read_parquet(q_path)
        q_df["freq"] = "Q"
    else:
        log("[WARN] Quarterly data missing.")
        q_df = pd.DataFrame()

    # Load Annual
    if os.path.isfile(a_path):
        a_df = pd.read_parquet(a_path)
        a_df["freq"] = "A"
    else:
        log("[WARN] Annual data missing.")
        a_df = pd.DataFrame()

    df = pd.concat([q_df, a_df], ignore_index=True, sort=False)

    # 1. Standardize Dates
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")

    # 2. Filter for Standard Feed (STD)
    keyset_col = next((c for c in df.columns if c in ["keyset", "datadate_keyset", "dataset", "datafmt"]), None)
    if keyset_col:
        before_len = len(df)
        keep_mask = df[keyset_col].astype(str).str.upper().eq("STD")
        drop_mask = df[keyset_col].astype(str).str.upper().isin(["PRE","PFO","SFAS","PDIV"])
        df = df[keep_mask & ~drop_mask].copy()
        log(f"Row Filter ({keyset_col}=STD): Dropped {before_len - len(df)} rows.")

    # 3. Sort (Critical for Shift operations in next step)
    sort_cols = [c for c in ["gvkey", "datadate", "freq"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df

log("Loading and filtering Compustat Panel...")
panel_df = load_and_prep_compustat(FUNDQ_PATH, FUNDA_PATH)

if len(panel_df) == 0:
    raise RuntimeError("No Compustat data loaded.")

############################################################
# 2. Comprehensive PIT-Risk Flag (AJEX + CFAC)
#    (Calculated BEFORE we scrub these columns)
############################################################

log("Computing 'price_pit_risk_flag' (AJEX/CFAC coverage)...")

# Identify all adjustment candidates present in the data
adj_candidates = ["ajexq", "ajex", "cfacshr", "cfacpr"]
adj_cols = [c for c in adj_candidates if c in panel_df.columns]

if adj_cols:
    # Ensure numeric
    for c in adj_cols:
        panel_df[c] = pd.to_numeric(panel_df[c], errors="coerce")

    # Shift within firm; detect any change in any adj column
    same_firm = panel_df["gvkey"] == panel_df["gvkey"].shift(1)
    has_change = pd.Series(False, index=panel_df.index)

    for c in adj_cols:
        curr = panel_df[c].fillna(-9999)
        prev = panel_df[c].shift(1).fillna(-9999)
        col_change = (curr != prev) & same_firm
        has_change |= col_change

    # [FIX]: Fill NA with False before casting to int to prevent "cannot convert NA to integer" error
    panel_df["_raw_change"] = has_change.fillna(False).astype(int)

    # Once a firm has any change, treat all rows as PIT-risky
    panel_df["price_pit_risk_flag"] = (
        panel_df.groupby("gvkey")["_raw_change"]
        .transform("cumsum")
        .gt(0)
        .astype("int8")
    )
    panel_df.drop(columns=["_raw_change"], inplace=True)

    log(
        f" -> Flag computed using {adj_cols}. "
        f"Rows flagged: {panel_df['price_pit_risk_flag'].sum()} / {len(panel_df)}"
    )
else:
    log("[INFO] No adjustment columns found. price_pit_risk_flag = 0.")
    panel_df["price_pit_risk_flag"] = 0

############################################################
# 3. Intelligent Scrubbing (Drop CFAC/AJEX Features)
############################################################

def intelligent_column_scrub(df):
    log("--- Starting Intelligent Column Scrub ---")
    orig_cols = set(df.columns)

    # A. The "Protected List" - These survive NO MATTER WHAT.
    PROTECTED_COLS = {
        # Keys & Time
        'gvkey', 'cik', 'datadate', 'freq', 'rdq',
        # Market Data (Kept, but flagged via price_pit_risk_flag)
        'prccq', 'mkvaltq', 'cshoq', 'cshtrq', 'prchq', 'prclq',
        # The Risk Flag itself
        'price_pit_risk_flag',
        # Core Financials
        'atq', 'revtq', 'niq', 'teqq', 'cheq',
        # Valid M&A Accounting
        'acqao', 'acqcshi', 'acqgdwl', 'acqic', 'acqintan', 'acqinvt',
        'acqlntal', 'acqmeth', 'acqniintc', 'acqppe', 'acqsc', 'deracq', 'datacqtr'
    }

    # B. The "Kill List" Patterns - Known Leakage
    PATTERNS = [
        # Rolling/Forward looking
        r'(?:FD\d{2}|TTM|YTD|12M|12MO|_12|L12)$',
        r'\b(?:LEAD|NEXT|T\+?1|FWD|FUTURE|EST)\b',
        # Adjustment Factors (Aggressive Removal)
        # Includes AJEX, ADJEX, CFACSHR, CFACPR
        r'(?:\bAJEX\b|\bADJEX\b|\bCFAC\w*|\bIBADJ\w*|_ADJ(?:Q|Y)?$)',
        r'(?:^|_)PRIOR$',
        r'_(?:FN|DC)[0-9A-Z]*$',
        # Semantic Leakage
        r'TARGET', r'TGT', r'MERGER', r'DEAL_ID', r'PENDING'
    ]
    P_LEAKAGE = re.compile('|'.join(PATTERNS), re.I)

    cols_to_drop = set()

    for col in orig_cols:
        if col.lower() in PROTECTED_COLS: continue

        if P_LEAKAGE.search(col):
            cols_to_drop.add(col)
            continue

        # Generic aggregates check
        if re.search(r'(?:Y|SA)$', col, re.I) and col.isupper() and col.isalnum():
            cols_to_drop.add(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log(f"Scrubbed {len(cols_to_drop)} columns based on leakage rules.")
        # Verify CFAC/AJEX are gone
        remaining = [c for c in df.columns if "cfac" in c.lower() or "ajex" in c.lower()]
        if not remaining:
            log(" -> Verified: All CFAC/AJEX columns dropped from feature set.")
        else:
            log(f" [WARN] Adjustment columns still present: {remaining}")

    return df

panel_df = intelligent_column_scrub(panel_df)

############################################################
# 4. Define "Safe Harbor" Snapshot (RDQ Aware)
############################################################

log("Defining Point-in-Time snapshot_ts (RDQ + Safe Harbor)...")

SAFE_LAG_DAYS = 90

# Ensure datadate and rdq are datetime
panel_df["datadate"] = pd.to_datetime(panel_df["datadate"], errors="coerce")
if "rdq" in panel_df.columns:
    panel_df["rdq"] = pd.to_datetime(panel_df["rdq"], errors="coerce")
else:
    panel_df["rdq"] = pd.NaT

# Baseline: fiscal period end + 90 days
base_snap = panel_df["datadate"] + pd.to_timedelta(SAFE_LAG_DAYS, unit="D")

# Effective snapshot: max(datadate + 90d, rdq) where rdq exists
eff_snap = base_snap.copy()
mask_rdq = panel_df["rdq"].notna()

# elementwise max between base_snap and rdq
eff_snap[mask_rdq] = np.maximum(
    base_snap[mask_rdq].values,
    panel_df["rdq"][mask_rdq].values
)

panel_df["snapshot_ts"] = pd.to_datetime(eff_snap).dt.tz_localize(None)

log(
    f"Snapshot defined as max(datadate + {SAFE_LAG_DAYS} days, rdq). "
    f"Example lag stats (days): "
    f"min={((panel_df['snapshot_ts'] - panel_df['datadate']).dt.days.min())}, "
    f"max={((panel_df['snapshot_ts'] - panel_df['datadate']).dt.days.max())}"
)

############################################################
# 5. Load & Normalize Deal Data (Direct Path, No Recursive Glob)
############################################################

DMA_CSV_PATH = os.path.join(BASE_DIR, "dma_corpus_metadata_with_factset_id.csv")

log("Loading Deal Data (direct path, no recursive glob)...")

def load_deals_data(csv_path):
    # 1. Basic existence check
    if not os.path.isfile(csv_path):
        log(f"[WARN] DMA CSV not found at {csv_path}. Returning empty deals.")
        return pd.DataFrame(columns=["cik", "ann_date", "_deal_value_num", "_deal_value_log"])

    # 2. Sniff delimiter (fallback to comma)
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(2048)
        try:
            dialect = csv.Sniffer().sniff(sample)
            sep = dialect.delimiter
        except Exception:
            sep = ","

    dma_df = pd.read_csv(csv_path, sep=sep, engine="python")

    # 3. Normalize column names
    col_map = {c: c.lower().strip().replace(" ", "").replace("_", "") for c in dma_df.columns}
    rev_map = {v: k for k, v in col_map.items()}

    # Map known variants to canonical names
    if "dateannouncement" in rev_map and "date_announcement" not in dma_df.columns:
        dma_df.rename(columns={rev_map["dateannouncement"]: "date_announcement"}, inplace=True)

    # 4. Ensure CIK
    if "cik" not in dma_df.columns and "url" in dma_df.columns:
        dma_df["cik"] = dma_df["url"].astype(str).str.extract(r'/data/(\d{1,10})/')

    dma_df["cik"] = pd.to_numeric(dma_df["cik"], errors="coerce").astype("Int64")
    dma_df["date_announcement"] = pd.to_datetime(dma_df.get("date_announcement"), errors="coerce")

    # 5. Numeric deal value if present
    val_col = None
    for candidate in ["Deal Value", "deal_value", "transaction_value"]:
        if candidate in dma_df.columns:
            val_col = candidate
            break

    if val_col is not None:
        dma_df["_deal_value_num"] = pd.to_numeric(
            dma_df[val_col].astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce"
        )
    else:
        dma_df["_deal_value_num"] = np.nan

    # 6. Log transform for model / event study
    dma_df["_deal_value_log"] = np.log1p(dma_df["_deal_value_num"].fillna(0))
    dma_df["ann_date"] = dma_df["date_announcement"]

    # 7. Final normalized frame
    out = dma_df[["cik", "ann_date", "_deal_value_num", "_deal_value_log"]].dropna(subset=["cik", "ann_date"])
    return out

deals = load_deals_data(DMA_CSV_PATH)
log(f"Loaded {len(deals)} deals.")

############################################################
# 6. Multi-Horizon Vectorized Labeling
############################################################

log("Labeling targets (0-3m and 0-6m horizons) relative to snapshot_ts...")

panel_labeled = panel_df.copy()
panel_labeled["cik"] = pd.to_numeric(panel_labeled["cik"], errors="coerce").astype("Int64")

# Pre-group deals
deals_by_cik = defaultdict(list)
for _, row in deals[deals['cik'].notna()].iterrows():
    deals_by_cik[int(row["cik"])].append((row["ann_date"], row["_deal_value_log"]))
for k in deals_by_cik:
    deals_by_cik[k] = sorted(deals_by_cik[k], key=lambda x: x[0])

# Prepare Lists
# [REPLACEMENT CODE FOR SECTION 6 in BLOCK 1]
log("SKIPPING slow Python loop labeling (will be handled via vectorized merge_asof in Block 3)...")

# Initialize placeholders so the dataframe schema remains valid for the next block
panel_labeled["label_deal_0_3m"] = 0
panel_labeled["label_deal_0_6m"] = 0
panel_labeled["past_deal_value_3m"] = np.nan

# --- COMMENTED OUT LEGACY LOOP ---
# ciks = panel_labeled["cik"].fillna(-1).astype(int).tolist()
# snaps = panel_labeled["snapshot_ts"].tolist()
# 
# label_0_3m = []
# label_0_6m = []
# past_deal_val = []
# 
# HORIZON_3M = pd.DateOffset(months=3)
# HORIZON_6M = pd.DateOffset(months=6)
# PAST_HORIZON = pd.DateOffset(months=3)
# 
# for cik_val, snap_ts in zip(ciks, snaps):
#     c_deals = deals_by_cik.get(cik_val, [])
# 
#     if not c_deals or pd.isna(snap_ts):
#         label_0_3m.append(0)
#         label_0_6m.append(0)
#         past_deal_val.append(np.nan)
#         continue
# 
#     has_3m = 0
#     has_6m = 0
#     max_past_val = -1.0
#     found_past = False
#     start_past = snap_ts - PAST_HORIZON
# 
#     for d_date, d_val in c_deals:
#         if pd.isna(d_date): continue
# 
#         # Check Future Targets (Strictly > snapshot)
#         if d_date > snap_ts:
#             if d_date <= snap_ts + HORIZON_3M:
#                 has_3m = 1
#                 has_6m = 1
#             elif d_date <= snap_ts + HORIZON_6M:
#                 has_6m = 1
# 
#         # Check Past Feature
#         if start_past < d_date <= snap_ts:
#             if pd.notna(d_val):
#                 if d_val > max_past_val:
#                     max_past_val = d_val
#                     found_past = True
# 
#     label_0_3m.append(has_3m)
#     label_0_6m.append(has_6m)
#     past_deal_val.append(max_past_val if found_past else np.nan)
# 
# # --- ASSIGN LABELS ---
# panel_labeled["label_deal_0_3m"] = label_0_3m
# panel_labeled["label_deal_0_6m"] = label_0_6m
# panel_labeled["past_deal_value_3m"] = past_deal_val

log("Labeling Complete.")
log(f"Positive labels (0-3m): {sum(label_0_3m)}")
log(f"Positive labels (0-6m): {sum(label_0_6m)}")

############################################################
# 7. VERIFICATION: High-Precision Event Study
############################################################

log("--- Running Event Study Verification ---")

try:
    if '_deal_value_num' not in deals.columns:
        log("[SKIP] _deal_value_num missing, skipping plot.")
    else:
        # Filter for Major Deals (> $100M) to ensure clear signal
        precise_events = deals[deals['_deal_value_num'] > 100][['cik', 'ann_date']].copy()
        precise_events = precise_events.rename(columns={'ann_date': 'event_date'})

        price_history = panel_labeled[['cik', 'datadate', 'prccq']].dropna().copy()

        # Merge Price History with Event Dates
        merged = pd.merge(price_history, precise_events, on='cik', how='inner')
        merged['days_rel'] = (merged['datadate'] - merged['event_date']).dt.days
        window = merged[(merged['days_rel'] >= -365) & (merged['days_rel'] <= 365)].copy()

        def normalize_precise(g):
            # Baseline: -90 to -10 days relative to Announcement
            baseline = g[(g['days_rel'] >= -90) & (g['days_rel'] <= -10)]
            if not baseline.empty:
                base_price = baseline.sort_values('days_rel', ascending=False).iloc[0]['prccq']
                if base_price > 0:
                    g['norm_price'] = g['prccq'] / base_price
                    return g
            return None

        norm_df = window.groupby(['cik', 'event_date']).apply(normalize_precise)

        if norm_df is not None and not norm_df.empty:
            norm_df['week_rel'] = (norm_df['days_rel'] / 7).round().astype(int)
            stats = norm_df.groupby('week_rel')['norm_price'].quantile([0.25, 0.50, 0.75]).unstack()

            plt.figure(figsize=(10, 6))
            plt.plot(stats.index, stats[0.50], color='crimson', linewidth=3, label='Median Stock Price')
            plt.fill_between(stats.index, stats[0.25], stats[0.75], color='crimson', alpha=0.1, label='IQR (25-75%)')
            plt.axvline(0, color='black', linestyle='--', linewidth=2, label='Announcement Day')
            plt.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
            plt.title('Verification: Price vs. News Announcement')
            plt.xlabel('Weeks Relative to Announcement')
            plt.ylabel('Normalized Price (1.0 = Pre-News)')
            plt.xlim(-20, 20)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # Verdict
            pre = stats.loc[-2, 0.50] if -2 in stats.index else 1.0
            post = stats.loc[2, 0.50] if 2 in stats.index else 1.0
            jump = post / pre - 1
            log(f"Median Jump (-2wk to +2wk): {jump:.1%}")
            if jump > 0.15:
                log("✅ VERDICT: Sharp Step-Function Detected. Data is aligned.")
            else:
                log("⚠️ VERDICT: Signal muted.")
        else:
            log("[WARN] Not enough data points for Event Study plot.")

except Exception as e:
    log(f"[WARN] Event Study Plot failed: {e}")

############################################################
# 8. Save
############################################################

log(f"Saving to {SAVE_PATH}...")
panel_labeled.to_parquet(SAVE_PATH, index=False)
log("Pipeline Complete.")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Retrieve EXACT Announcement Dates
# ---------------------------------------------------------
# We need to go back to the 'deals' dataframe to get the precise day.
# If 'deals' is not in memory, we fall back, but assuming you ran Step 2, it's there.

if 'deals' not in locals():
    raise RuntimeError("The 'deals' dataframe is missing. Please re-run Step 2 (Load DMA/FactSet).")

# Filter for Major Deals (> $100M nominal) to ensure signal
# Note: We use the numeric column we created earlier
precise_events = deals[deals['_deal_value_num'] > 100][['cik', 'ann_date']].copy()
precise_events = precise_events.rename(columns={'ann_date': 'event_date'})

# 2. Get Price History
# ---------------------------------------------------------
price_history = panel_labeled[['cik', 'datadate', 'prccq']].dropna().copy()

# 3. Merge on CIK
# ---------------------------------------------------------
# This creates a row for every combination of (Price Date) and (Deal Announcement)
merged = pd.merge(price_history, precise_events, on='cik', how='inner')

# Calculate Days Relative to Announcement
merged['days_rel'] = (merged['datadate'] - merged['event_date']).dt.days

# Narrow the window to +/- 1 year (approx 365 days)
window = merged[(merged['days_rel'] >= -365) & (merged['days_rel'] <= 365)].copy()

# 4. Normalize to "Day -30" (approx)
# ---------------------------------------------------------
# We want the price shortly before the rumor mill starts, but not too far back.
# Let's try to find the price closest to t = -30 days.

def normalize_precise(g):
    # Find rows in the "Pre-Deal Baseline" window (-90 to -10 days)
    baseline_window = g[(g['days_rel'] >= -90) & (g['days_rel'] <= -10)]

    if len(baseline_window) > 0:
        # Take the price closest to -10 days (most recent "clean" price)
        base_price = baseline_window.sort_values('days_rel', ascending=False).iloc[0]['prccq']
        if base_price > 0:
            g['norm_price'] = g['prccq'] / base_price
            return g
    return None

# Apply normalization
norm_df = window.groupby(['cik', 'event_date']).apply(normalize_precise)

if norm_df is not None and not norm_df.empty:
    # 5. Binning for the Plot
    # Since days are continuous, we bin them into "Weeks relative to deal"
    norm_df['week_rel'] = (norm_df['days_rel'] / 7).round().astype(int)

    stats = norm_df.groupby('week_rel')['norm_price'].quantile([0.25, 0.50, 0.75]).unstack()

    # 6. Plot
    plt.figure(figsize=(12, 7))

    # Plot Median
    plt.plot(stats.index, stats[0.50], color='crimson', linewidth=3, label='Median Stock Price')

    # Plot IQR
    plt.fill_between(stats.index, stats[0.25], stats[0.75], color='crimson', alpha=0.1, label='IQR (25-75%)')

    plt.axvline(0, color='black', linestyle='--', linewidth=2, label='Announcement Day')
    plt.axhline(1.0, color='gray', linestyle=':', alpha=0.5)

    plt.title('High-Precision Event Study: Price vs. Announcement Date\n(Aligned to Exact Press Release Day)', fontsize=14)
    plt.xlabel('Weeks Relative to Announcement', fontsize=12)
    plt.ylabel('Normalized Price (1.0 = Pre-Deal)', fontsize=12)
    plt.xlim(-20, 20) # +/- 20 weeks
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()

    # 7. Verdict
    # Check Median at Week -1 vs Week +1
    try:
        pre = stats.loc[-2, 0.50] # 2 weeks before
        post = stats.loc[2, 0.50] # 2 weeks after
        jump = post / pre - 1
        print(f"Median Price 2 Weeks Pre-Deal:  {pre:.2f}")
        print(f"Median Price 2 Weeks Post-Deal: {post:.2f}")
        print(f"Immediate Premium Jump:         {jump:.1%}")

        if jump > 0.15:
            print("✅ VERDICT: Sharp Step-Function Detected. Data is Clean.")
        else:
             print("⚠️ VERDICT: Signal is still muted. Check for 'rumor leak' or deal mix.")
    except KeyError:
        print("Not enough density at +/- 2 weeks to calculate simple jump.")
else:
    print("Not enough matching price/deal pairs found.")

from google.colab import drive

# Mount Google Drive
try:
    drive.mount('/content/drive')
    print("[Success] Google Drive mounted.")
except Exception as e:
    print(f"[Error] Could not mount Google Drive: {e}")

import os, glob, re, time, warnings, csv
import numpy as np
import pandas as pd
from google.colab import drive
from collections import defaultdict

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# --- Define Global Context & Helpers from Master Pipeline ---

# Do not overwrite the timestamped log() from the master cell if it exists
if "log" not in globals():
    def log(msg):
        print(f"[Pipeline] {msg}")

# Always ensure BASE_DIR is consistent with where the master pipeline saved
try:
    # If we are in Colab, make sure Drive is mounted
    if "google.colab" in str(get_ipython()):
        drive.mount("/content/drive", force_remount=False)
        candidate = "/content/drive/MyDrive"
        if os.path.isdir(candidate):
            BASE_DIR = candidate
        else:
            BASE_DIR = os.getcwd()
            log(f"[WARN] /content/drive/MyDrive not found. Falling back to {BASE_DIR}")
    else:
        BASE_DIR = os.getcwd()
        log(f"[WARN] Not running in Colab. Using local BASE_DIR = {BASE_DIR}")
except Exception as e:
    BASE_DIR = os.getcwd()
    log(f"[WARN] Failed to mount Drive. Using local BASE_DIR = {BASE_DIR}. Error: {e}")

# Mirror the master context for other paths (even if not strictly needed here)
FUNDQ_PATH = os.path.join(BASE_DIR, "fundq_full.parquet")
FUNDA_PATH = os.path.join(BASE_DIR, "funda_full.parquet")
DMA_CSV_PATH = os.path.join(BASE_DIR, "dma_corpus_metadata_with_factset_id.csv")

SAFE_LAG_DAYS = 90  # must match master pipeline

log("Setting up environment (Loading data and creating core keys)...")
log(f"BASE_DIR resolved to: {BASE_DIR}")

# Ensure SAVE_PATH matches the master pipeline
OUTPUT_FILENAME = "labeled_mna_pit_panel.parquet"
SAVE_PATH = os.path.join(BASE_DIR, OUTPUT_FILENAME)
log(f"Expecting labeled panel at: {SAVE_PATH}")

# Bring in the same deal loader as in the master cell if not already defined
if "load_deals_data" not in globals():
    def load_deals_data(csv_path):
        # 1. Basic existence check
        if not os.path.isfile(csv_path):
            log(f"[WARN] DMA CSV not found at {csv_path}. Returning empty deals.")
            return pd.DataFrame(columns=["cik", "ann_date", "_deal_value_num", "_deal_value_log"])

        # 2. Sniff delimiter (fallback to comma)
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(2048)
            try:
                dialect = csv.Sniffer().sniff(sample)
                sep = dialect.delimiter
            except Exception:
                sep = ","

        dma_df = pd.read_csv(csv_path, sep=sep, engine="python")

        # 3. Normalize column names
        col_map = {c: c.lower().strip().replace(" ", "").replace("_", "") for c in dma_df.columns}
        rev_map = {v: k for k, v in col_map.items()}

        # Map known variants to canonical names
        if "dateannouncement" in rev_map and "date_announcement" not in dma_df.columns:
            dma_df.rename(columns={rev_map["dateannouncement"]: "date_announcement"}, inplace=True)

        # 4. Ensure CIK
        if "cik" not in dma_df.columns and "url" in dma_df.columns:
            dma_df["cik"] = dma_df["url"].astype(str).str.extract(r'/data/(\d{1,10})/')

        dma_df["cik"] = pd.to_numeric(dma_df["cik"], errors="coerce").astype("Int64")
        dma_df["date_announcement"] = pd.to_datetime(dma_df.get("date_announcement"), errors="coerce")

        # 5. Numeric deal value if present
        val_col = None
        for candidate in ["Deal Value", "deal_value", "transaction_value"]:
            if candidate in dma_df.columns:
                val_col = candidate
                break

        if val_col is not None:
            dma_df["_deal_value_num"] = pd.to_numeric(
                dma_df[val_col].astype(str).str.replace(r"[^0-9.]", "", regex=True),
                errors="coerce"
            )
        else:
            dma_df["_deal_value_num"] = np.nan

        # 6. Log transform for model / event study
        dma_df["_deal_value_log"] = np.log1p(dma_df["_deal_value_num"].fillna(0))
        dma_df["ann_date"] = dma_df["date_announcement"]

        # 7. Final normalized frame
        out = dma_df[["cik", "ann_date", "_deal_value_num", "_deal_value_log"]].dropna(subset=["cik", "ann_date"])
        return out

# 1) Get panel_labeled: prefer in-memory, else load from parquet
if "panel_labeled" in globals():
    log("Re-using existing 'panel_labeled' from master pipeline.")
else:
    if os.path.isfile(SAVE_PATH):
        log(f"Loading 'panel_labeled' from {SAVE_PATH}...")
        panel_labeled = pd.read_parquet(SAVE_PATH)
    else:
        raise RuntimeError(
            "panel_labeled not found in memory, and saved parquet not found at "
            f"{SAVE_PATH}. Run the 'M&A FORECAST DATASET PIPELINE (Final Surgical Master)' cell first."
        )

# 2) Get deals: prefer in-memory, else rebuild from CSV
if "deals" in globals():
    log("Re-using existing 'deals' from master pipeline.")
else:
    log(f"Loading 'deals' from DMA CSV at {DMA_CSV_PATH}...")
    deals = load_deals_data(DMA_CSV_PATH)

log(f"Setup complete. 'panel_labeled' ({len(panel_labeled)} rows) and 'deals' ({len(deals)} rows) are ready.")

############################################################
# 6. OPTIMIZED PIPELINE: VECTORIZED PRE-CALCULATION (FIXED)
#    - Fix: Sorts by TIMESTAMP (global) to satisfy merge_asof
#    - Speed: Uses combine_first (Fast In-Place Coalescing)
############################################################

import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from tqdm.auto import tqdm

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def log(msg):
    print(f"[Pipeline] {msg}")

# ==============================================================================
# 1. DEFINE ROBUST VECTORIZED FEATURE FUNCTIONS
# ==============================================================================

EPS = 1e-9
ID_LIKE = {
    "gvkey", "iid", "tic", "cusip", "conm",
    "cik", "datadate", "freq", "rdq", "snapshot_ts", "year"
}

TARGETS = {
    "label_deal_0_3m",
    "label_deal_0_6m",
    "past_deal_value_3m",
    "target_days_to_event",
}

def create_base_vars(df):
    """
    OPTIMIZED: Uses bfill(axis=1) instead of iterative combine_first.
    """
    base_vars = {
        "assets":        ["atq", "at"],
        "revenue":       ["saleq", "sale"],
        "net_income":    ["niq", "ni"],
        "debt_longterm": ["dlttq", "dltt"],
        "equity_book":   ["ceqq", "ceq"],
        "oibdp":         ["oibdpq", "oibdp"],
        "act":           ["actq", "act"],
        "lct":           ["lctq", "lct"],
        "che":           ["cheq", "che"],
        "xint":          ["xintq", "xint"],
        "xrd":           ["xrdq", "xrd"],
        "xsga":          ["xsgaq", "xsga"],
        "capx":          ["capxq", "capx"],
        "oancf":         ["oancfq", "oancf"],
        "csho":          ["cshoq", "csho"],
        "cshtrq":        ["cshtrq", "cshtr"],
        "ppent":         ["ppentq", "ppent"],
        "mkvalt":        ["mkvaltq", "mkvalt"],
    }

    # Vectorized Coalescing
    for var, cands in tqdm(base_vars.items(), desc="Fast Coalescing", leave=False):
        existing = [c for c in cands if c in df.columns]
        if not existing:
            df[var] = np.nan
        elif len(existing) == 1:
            df[var] = df[existing[0]]
        else:
            # Replaces slow combine_first with vectorized backfill
            df[var] = df[existing].bfill(axis=1).iloc[:, 0]

    return df

def create_hand_treated_features(df):
    """
    OPTIMIZED: Uses Global Shift + Masking instead of GroupBy().shift().
    """
    def safe_mag(series):
        return series.fillna(0).abs() + EPS

    # --- Ratios (Unchanged) ---
    rev = safe_mag(df["revenue"])
    at  = safe_mag(df["assets"])

    df["profit_margin"] = df["net_income"] / rev
    df["roa"]           = df["net_income"] / at
    df["oper_margin"]   = df["oibdp"] / rev
    df["leverage"]      = df["debt_longterm"] / at
    df["curr_ratio"]    = df["act"] / safe_mag(df["lct"])
    df["cash_ratio"]    = df["che"] / at
    df["interest_coverage"] = df["oibdp"] / safe_mag(df["xint"])

    # Intensities
    df["rd_int"]   = df["xrd"]   / rev
    df["sgna_int"] = df["xsga"]  / rev
    df["capx_int"] = df["capx"]  / at
    df["ocf_int"]  = df["oancf"] / at

    ratio_cols = [
        "profit_margin", "roa", "oper_margin", "leverage",
        "curr_ratio", "cash_ratio", "interest_coverage",
        "rd_int", "sgna_int", "capx_int", "ocf_int",
    ]
    df[ratio_cols] = df[ratio_cols].fillna(0).clip(-5, 5)

    # Turnover & Size
    df["log_csho"] = np.log1p(df["csho"].fillna(0).clip(lower=0))
    df["turnover"] = df["cshtrq"] / safe_mag(df["csho"])
    df["log_turnover"] = np.log1p(df["turnover"].fillna(0).clip(lower=0))

    # --- Growth (Global Shift Optimization) ---
    # 1. Critical: Sort is required for global shift to work
    df.sort_values(["gvkey", "datadate"], inplace=True)

    log_growth_inputs = ["assets", "revenue", "ppent", "mkvalt"]
    input_cols = []
    for raw in log_growth_inputs:
        col = f"log_{raw}"
        if col not in df.columns:
            df[col] = np.log1p(df[raw].fillna(0).clip(lower=0))
        input_cols.append(col)

    # 2. Compute Global Shifts (Shift whole array regardless of firm)
    df_shifts_1 = df[input_cols].shift(1)
    df_shifts_4 = df[input_cols].shift(4)

    # 3. Compute Masks (Identify where firm changed)
    gvkey = df["gvkey"]
    mask_1 = (gvkey == gvkey.shift(1))
    mask_4 = (gvkey == gvkey.shift(4))
    
    is_q = (df["freq"] == "Q")

    for i, raw in enumerate(log_growth_inputs):
        col = input_cols[i]
        
        # QoQ: Lag 1 (Masked)
        prev_1 = df_shifts_1[col].where(mask_1) 
        
        df[f"dlog_{raw}_qoq"] = np.where(
            is_q,
            df[col] - prev_1,
            np.nan,
        )

        # YoY: Lag 4 if Q (Masked), Lag 1 if A (Masked)
        prev_4 = df_shifts_4[col].where(mask_4)
        yoy_lag_vals = np.where(is_q, prev_4, prev_1)
        df[f"dlog_{raw}_yoy"] = df[col] - yoy_lag_vals

    return df

# ==============================================================================
# 2. PRE-CALCULATION PHASE
# ==============================================================================

log("--- STARTING GLOBAL PRE-CALCULATION (FAST + SORT FIX) ---")

if "panel_labeled" not in locals() or "deals" not in locals():
    raise RuntimeError("Missing 'panel_labeled' or 'deals' in the environment.")

# A. Prepare Targets for labeling (use all deals, like Surgical Master)
precise_events = deals[["cik", "ann_date", "_deal_value_log"]].copy()
precise_events = precise_events.rename(columns={"ann_date": "event_date"})

precise_events = precise_events.dropna(subset=["cik", "event_date"]).copy()
precise_events["cik"] = pd.to_numeric(precise_events["cik"], errors="coerce")
precise_events = precise_events.dropna(subset=["cik"]).copy()
precise_events["cik"] = precise_events["cik"].astype(int)

# SORT FIX: sort by time (and optionally cik) for merge_asof
precise_events = precise_events.sort_values(["event_date"]).reset_index(drop=True)

# B. Prepare Features
log("Generating features for full history...")
full_panel = panel_labeled.copy()

log("Dataframe copied. Starting Base Vars Coalescing (combine_first)...")
full_panel = create_base_vars(full_panel)
log("Base vars done. Calculating Ratios & Growth...")
full_panel = create_hand_treated_features(full_panel)

# C. Vectorized Target Mapping
log("Mapping targets (0-3m, 0-6m, Past Value) using merge_asof...")

# 1. Drop rows with missing snapshot_ts
before_drop = len(full_panel)
full_panel["snapshot_ts"] = pd.to_datetime(full_panel["snapshot_ts"], errors="coerce")
full_panel = full_panel.dropna(subset=["snapshot_ts"])
if len(full_panel) < before_drop:
    log(f"Dropped {before_drop - len(full_panel)} rows with missing snapshot_ts.")

# 2. Clean and standardize CIK on panel side
full_panel["cik"] = full_panel["cik"].replace(np.nan, -1)
full_panel["cik"] = pd.to_numeric(
    full_panel["cik"], errors="coerce"
).fillna(-1).astype(int)

# SORT FIX: sort panel by snapshot_ts (global) for merge_asof
full_panel = full_panel.sort_values(
    "snapshot_ts"    # FIX: sort by time only
).reset_index(drop=True)

# 3. Forward Lookahead (future events)
fwd_merge = pd.merge_asof(
    full_panel[["cik", "snapshot_ts"]],
    precise_events[["cik", "event_date"]],
    left_on="snapshot_ts",
    right_on="event_date",
    by="cik",
    direction="forward",
    tolerance=pd.Timedelta(days=185),
)

fwd_delta = (fwd_merge["event_date"] - fwd_merge["snapshot_ts"]).dt.days

# 4. Backward Lookback (past events, for past_deal_value_3m)
bwd_merge = pd.merge_asof(
    full_panel[["cik", "snapshot_ts"]],
    precise_events[["cik", "event_date", "_deal_value_log"]],
    left_on="snapshot_ts",
    right_on="event_date",
    by="cik",
    direction="backward",
    tolerance=pd.Timedelta(days=92),
)

bwd_delta = (bwd_merge["snapshot_ts"] - bwd_merge["event_date"]).dt.days
has_past = (bwd_delta >= 0) & (bwd_delta <= 92)

# 5. Assign Targets
full_panel["target_days_to_event"] = fwd_delta

full_panel["label_deal_0_3m"] = (
    (fwd_delta > 0) & (fwd_delta <= 92)
).fillna(False).astype(int)

full_panel["label_deal_0_6m"] = (
    (fwd_delta > 0) & (fwd_delta <= 183)
).fillna(False).astype(int)

full_panel["past_deal_value_3m"] = np.where(
    has_past,
    bwd_merge["_deal_value_log"],
    np.nan,
)

# 6. Final sort for user convenience (firm-time)
full_panel = full_panel.sort_values(
    ["cik", "snapshot_ts"]
).reset_index(drop=True)

log(f"Pre-calc complete. Full Panel Shape: {full_panel.shape}")
log(f"Positives (0-3m): {full_panel['label_deal_0_3m'].sum()}")
log(f"Positives (0-6m): {full_panel['label_deal_0_6m'].sum()}")

# [ADD TO END OF BLOCK 3]
FINAL_FEATURE_PATH = os.path.join(BASE_DIR, "full_panel_features_ready.parquet")
log(f"Saving final training-ready panel to {FINAL_FEATURE_PATH}...")
full_panel.to_parquet(FINAL_FEATURE_PATH, index=False)
log("✅ Pipeline Optimization Complete. Ready for training.")


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


# ======================================================================
# 0. CONFIG: which target are we forecasting?
# ======================================================================

# Next-quarter announcement (0–3m window)
TARGET_COL = "label_deal_0_3m"
# For 0–6m horizon, swap:
# TARGET_COL = "label_deal_0_6m"


# ======================================================================
# 1. ROBUST FEATURE SCALER (TRAIN / TEST)
# ======================================================================

def fit_feature_pipeline(df, feature_cols):
    """
    Fit a simple numeric feature pipeline on the TRAIN slice.

    Steps:
      - Select feature columns
      - Replace +/-inf with NaN
      - Convert to float64 via to_numpy (handles pd.NA)
      - Compute column-wise mean and std (ignoring NaNs)
      - Handle all-NaN and zero-variance columns robustly
    """
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    X_vals = X.to_numpy(dtype=np.float64)
    n_rows, n_cols = X_vals.shape

    # Detect columns that are entirely NaN
    all_nan_cols = np.all(np.isnan(X_vals), axis=0)

    # Initialize mean/std
    col_mean = np.zeros(n_cols, dtype=np.float64)
    col_std = np.ones(n_cols, dtype=np.float64)

    # Compute stats only on columns that are not all-NaN
    valid_cols = ~all_nan_cols
    if valid_cols.any():
        col_mean[valid_cols] = np.nanmean(X_vals[:, valid_cols], axis=0)
        col_std[valid_cols] = np.nanstd(X_vals[:, valid_cols], axis=0, ddof=0)

    # Guard against degenerate stds (zero, negative, or non-finite)
    bad_std = (col_std <= 0) | ~np.isfinite(col_std)
    col_std[bad_std] = 1.0

    # Standardize and clean any remaining NaNs/infs
    X_scaled = (X_vals - col_mean) / col_std
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    scalers = {
        "feature_cols": list(feature_cols),
        "mean": col_mean,
        "std": col_std,
    }
    return X_scaled, scalers


def transform_feature_pipeline(df, scalers):
    """
    Apply the fitted pipeline to a TEST slice.
    - Reindexes to the same feature_cols
    - Uses stored mean/std
    - Cleans NaNs/infs to 0
    """
    feature_cols = scalers["feature_cols"]
    col_mean = scalers["mean"]
    col_std = scalers["std"]

    X = df.reindex(columns=feature_cols).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    X_vals = X.to_numpy(dtype=np.float64)
    X_scaled = (X_vals - col_mean) / col_std
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return X_scaled


# ======================================================================
# 2. OPTIONAL: class-imbalance aware downsampling of negatives
# ======================================================================

def downsample_negatives(df, target_col=TARGET_COL,
                         max_neg_per_pos=50,
                         min_neg=5000,
                         random_state=42):
    """
    Downsample negatives while keeping all positives.

    - Keeps all rows where target_col == 1.
    - Samples at most max_neg_per_pos * n_pos negatives.
    - Always keeps at least min_neg negatives if available.
    - If no positives or very few negatives, returns df unchanged.
    """
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]

    n_pos = len(pos)
    n_neg = len(neg)

    if n_neg == 0:
        return df

    if n_pos == 0:
        return df

    target_neg = max(min_neg, min(n_neg, max_neg_per_pos * n_pos))

    if target_neg >= n_neg:
        return df

    neg_sample = neg.sample(n=target_neg, random_state=random_state)

    out = pd.concat([pos, neg_sample], axis=0)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ==============================================================
# 0. Config + guards
# ==============================================================

# Target column
if "TARGET_COL" not in globals():
    TARGET_COL = "label_deal_0_3m"

# ID-like columns
if "ID_LIKE" not in globals():
    ID_LIKE = {"gvkey", "cik", "datadate"}
else:
    ID_LIKE = set(ID_LIKE) | {"gvkey", "cik", "datadate"}

# Target set for exclusion
if "TARGETS" not in globals():
    TARGETS = {TARGET_COL}
else:
    TARGETS = set(TARGETS) | {TARGET_COL}

# Hyperparameters for trading rule
DEAL_PREMIUM_EST = 0.25   # assumed median deal premium (quarter-scale)
COST_PER_TRADE   = 0.003  # 30 bps per roundtrip, in return space
NET_ALPHA_MIN    = 0.0    # threshold for net_alpha > 0 to be considered
K_MAX            = 50     # hard cap on number of bets
K_MIN            = 10     # minimum number of bets (fallback if too few pass filters)
POOL_MULT        = 4.0    # candidate pool size multiple vs K_MAX
CORR_LAG_Q       = 8      # trailing quarters for correlation estimation
RHO_MAX          = 0.80   # max allowed |corr| between selected names
N_SIMS_RANDOM    = 200    # random baseline portfolios per quarter

# Basic type sanity and sort
full_panel["datadate"] = pd.to_datetime(full_panel["datadate"], errors="coerce")
full_panel = full_panel.sort_values(["cik", "datadate"]).reset_index(drop=True)

# Require prices for PnL
if "prccq" not in full_panel.columns:
    raise RuntimeError("full_panel must contain 'prccq' for realized PnL calculation.")

# ==============================================================
# 1. Compute forward and backward 1-quarter returns (IN-PLACE)
# ==============================================================

# Forward returns: used for realized PnL
if "ret_fwd_1q" not in full_panel.columns:
    full_panel["prccq_next"] = (
        full_panel.groupby("cik")["prccq"]
        .shift(-1)
        .astype("float32")
    )
    full_panel["ret_fwd_1q"] = (
        full_panel["prccq_next"] / full_panel["prccq"] - 1.0
    ).astype("float32")
    full_panel.drop(columns=["prccq_next"], inplace=True)

# Backward returns: used only for correlation (historical comovement)
if "ret_back_1q" not in full_panel.columns:
    full_panel["prccq_prev"] = (
        full_panel.groupby("cik")["prccq"]
        .shift(1)
        .astype("float32")
    )
    full_panel["ret_back_1q"] = (
        full_panel["prccq"] / full_panel["prccq_prev"] - 1.0
    ).astype("float32")
    full_panel.drop(columns=["prccq_prev"], inplace=True)

# Ensure all floats are float32 to keep RAM under control
float_cols = full_panel.select_dtypes(include=["float64"]).columns
if len(float_cols) > 0:
    full_panel[float_cols] = full_panel[float_cols].astype("float32")

print("full_panel shape:", full_panel.shape)
print(
    "Approx memory (GB):",
    round(full_panel.memory_usage(deep=True).sum() / 1e9, 3),
)

# Global sanity on forward returns (raw)
r_all = (
    full_panel["ret_fwd_1q"]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)
print("\nGlobal ret_fwd_1q summary (raw):")
print(r_all.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

print("\nTop 5 and bottom 5 raw returns (potential split artifacts):")
print("Top 5:")
print(r_all.sort_values(ascending=False).head(5))
print("\nBottom 5:")
print(r_all.sort_values(ascending=True).head(5))


def clip_returns(s, lower=-0.8, upper=0.8):
    """Clip extreme returns to reduce split / bad data artifacts."""
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.clip(lower=lower, upper=upper)

# ==============================================================
# 2. Quarter coverage and list of holdout quarters (expanding window)
# ==============================================================

panel_q = full_panel["datadate"].dt.to_period("Q")

coverage = (
    full_panel
    .assign(panel_q=panel_q)
    .groupby("panel_q")[TARGET_COL]
    .agg(n_panel="size", n_pos="sum")
)

# We only consider quarters that actually have at least 1 positive
pos_info = coverage[coverage["n_pos"] > 0].copy()

if len(pos_info) < 2:
    raise RuntimeError("Need at least 2 quarters with positives for expanding-window backtest.")

# Holdout quarters: all positive quarters except the first (no prior positives to train on)
holdout_quarters = list(pos_info.index.sort_values()[1:])

print("\nHoldout quarters to be evaluated (expanding window):")
print(holdout_quarters[:10], "...")  # just first few for sanity
print(f"Total holdout quarters: {len(holdout_quarters)}")

# ==============================================================
# 3. Feature set (drop obvious leakage sources)
# ==============================================================

numeric_cols = full_panel.select_dtypes(include=[np.number]).columns

always_exclude = set(ID_LIKE) | set(TARGETS) | {"ret_fwd_1q", "ret_back_1q"}
suspect_meta = {"updq", "upd", "srcq"}
suspect_option = {
    "xoptdq", "xoptepsq", "xoptepsy", "xoptdy",
    "optvolq", "optvol", "optvoly"
}
extra_exclude = suspect_meta | suspect_option

feature_cols = [
    c for c in numeric_cols
    if c not in always_exclude and c not in extra_exclude
]

print(f"\nTotal numeric feature columns used (after exclusions): {len(feature_cols)}")
if suspect_meta & set(numeric_cols):
    print("Dropped vendor metadata flags:", suspect_meta & set(numeric_cols))
if suspect_option & set(numeric_cols):
    print("Dropped option features:", suspect_option & set(numeric_cols))

suspicious_names = [
    c for c in feature_cols
    if any(k in c.lower() for k in ["label", "event", "ann_", "target", "deal"])
]
print("\nSuspicious / label-ish features (should be empty or benign):")
print(suspicious_names if suspicious_names else "[]")

# ==============================================================
# 4. Helpers: downsampling, random baselines, correlation matrix
# ==============================================================

if "downsample_negatives" not in globals():
    def downsample_negatives(df, target_col=TARGET_COL,
                             max_neg_per_pos=50, min_neg=5000,
                             random_state=42):
        pos = df[df[target_col] == 1]
        neg = df[df[target_col] == 0]

        n_pos = len(pos)
        n_neg = len(neg)

        if (n_pos == 0) or (n_neg == 0):
            return df

        target_neg = max(min_neg, min(n_neg, max_neg_per_pos * n_pos))
        if target_neg >= n_neg:
            return df

        neg_sample = neg.sample(n=target_neg, random_state=random_state)
        out = pd.concat([pos, neg_sample], axis=0)
        out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        return out


def simulate_random_portfolios(ret_series, K, n_sims=N_SIMS_RANDOM, seed=123):
    """Simulate equal-weight random K-name portfolios using clipped returns."""
    rng = np.random.default_rng(seed)
    ret_clean = clip_returns(ret_series).dropna()
    n = len(ret_clean)
    if n == 0 or n < K or K <= 0:
        return None

    vals = ret_clean.to_numpy()

    sim_rets = []
    for _ in range(n_sims):
        chosen_idx = rng.choice(n, size=K, replace=False)
        sim_rets.append(vals[chosen_idx].mean())
    return np.array(sim_rets)


def build_corr_matrix_for_candidates(candidate_ciks, holdout_qtr, lag_q=CORR_LAG_Q):
    """
    Build cross-sectional correlation matrix of trailing 1Q *backward* returns
    for the candidate CIKs, over the last `lag_q` quarters before holdout_qtr.

    Handles duplicate (datadate, cik) entries by aggregating (mean) before pivot.
    Returns corr(cik x cik) or None if not enough history.
    """
    if lag_q <= 0 or len(candidate_ciks) < 2:
        return None

    q = panel_q
    # History window: [holdout_qtr - lag_q, holdout_qtr)
    q_min = holdout_qtr - lag_q
    mask_hist = (q >= q_min) & (q < holdout_qtr)

    df_hist = full_panel.loc[mask_hist, ["datadate", "cik", "ret_back_1q"]].copy()
    df_hist = df_hist.dropna(subset=["ret_back_1q"])
    df_hist = df_hist[df_hist["cik"].isin(candidate_ciks)]

    if df_hist["cik"].nunique() < 2:
        return None

    # Aggregate duplicates (datadate, cik) -> mean return
    df_hist = (
        df_hist
        .groupby(["datadate", "cik"], as_index=False)["ret_back_1q"]
        .mean()
    )

    if df_hist.empty:
        return None

    try:
        pivot = df_hist.pivot(index="datadate", columns="cik", values="ret_back_1q")
    except ValueError:
        # Still some weird duplicate pattern; bail out to no corr.
        return None

    # Need at least 2 names and 2 time points for meaningful corr
    if pivot.shape[1] < 2 or pivot.shape[0] < 2:
        return None

    corr = pivot.corr()
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return corr


def select_candidates_with_corr(candidates_pool,
                                corr_mat,
                                k_max=K_MAX,
                                k_min=K_MIN,
                                net_alpha_min=NET_ALPHA_MIN,
                                rho_max=RHO_MAX):
    """
    Greedy selection of candidates with:
      - Priority by descending net_alpha in candidates_pool.
      - Stop once net_alpha <= net_alpha_min AND len(selected) >= k_min.
      - Hard cap at k_max names.
      - Correlation cap |rho| <= rho_max vs already-selected names, if corr_mat given.

    Returns a DataFrame of selected rows (may be empty).
    """
    if candidates_pool.empty:
        return candidates_pool.copy()

    selected_rows = []
    selected_ciks = []

    use_corr = corr_mat is not None and corr_mat.shape[0] >= 2

    for _, row in candidates_pool.iterrows():
        if len(selected_rows) >= k_max:
            break

        net_a = row.get("net_alpha", 0.0)
        if net_a <= net_alpha_min and len(selected_rows) >= k_min:
            # Stop once edge is gone and we already have enough bets
            break

        cik_i = row["cik"]

        # Correlation screen
        if use_corr and selected_ciks and (cik_i in corr_mat.index):
            # Only compare vs selected ciks that are in corr_mat
            valid_selected = [c for c in selected_ciks if c in corr_mat.columns]
            if valid_selected:
                corr_row = corr_mat.loc[cik_i, valid_selected]
                max_abs_corr = np.abs(corr_row).max()
                if max_abs_corr > rho_max:
                    continue  # skip this name

        # If corr_mat is None or cik_i not in corr_mat, treat as okay.
        selected_rows.append(row)
        selected_ciks.append(cik_i)

    if not selected_rows:
        return candidates_pool.head(0).copy()

    return pd.DataFrame(selected_rows)

# ==============================================================
# 5. Quarter runner with dynamic K and correlation screen
# ==============================================================

def run_quarter_experiment(holdout_qtr):
    """
    Train on all quarters strictly before holdout_qtr,
    test on holdout_qtr, construct a dynamic-K portfolio,
    and compute performance vs baselines.
    """
    q = panel_q
    train_mask = q < holdout_qtr
    holdout_mask = q == holdout_qtr

    df_train = full_panel[train_mask].copy()
    df_holdout = full_panel[holdout_mask].copy()

    print("\n" + "=" * 70)
    print(f"Quarter: {holdout_qtr}")
    print("=" * 70)
    print(f"Train size:   {len(df_train)}")
    print(f"Holdout size: {len(df_holdout)}")

    print("\nTrain label counts:")
    print(df_train[TARGET_COL].value_counts(dropna=False))

    print("\nHoldout label counts:")
    print(df_holdout[TARGET_COL].value_counts(dropna=False))

    # If no positives in holdout at all, AUC is undefined and base deal rate is 0
    # but we can still trade; p0 falls back to train base rate.
    n_pos_train = int(df_train[TARGET_COL].sum())
    n_pos_holdout = int(df_holdout[TARGET_COL].sum())

    # Downsample negatives in train
    df_train_ds = downsample_negatives(
        df_train,
        target_col=TARGET_COL,
        max_neg_per_pos=50,
        min_neg=5000,
        random_state=42,
    )

    y_train = df_train_ds[TARGET_COL].astype(int).values
    y_holdout = df_holdout[TARGET_COL].astype(int).values

    print(f"\nAfter downsampling: train size = {len(df_train_ds)}, "
          f"train positives = {int(df_train_ds[TARGET_COL].sum())}")

    if "fit_feature_pipeline" not in globals() or "transform_feature_pipeline" not in globals():
        raise RuntimeError("fit_feature_pipeline / transform_feature_pipeline not defined.")

    X_train, scalers = fit_feature_pipeline(df_train_ds, feature_cols)
    X_holdout = transform_feature_pipeline(df_holdout, scalers)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=150,
        class_weight="balanced",
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X_train, y_train)
    holdout_scores = clf.predict_proba(X_holdout)[:, 1]

    valid_auc = (y_holdout.sum() > 0) and (np.unique(y_holdout).size == 2)
    holdout_q_dt = holdout_qtr.to_timestamp(how="end")
    if valid_auc:
        auc_holdout = roc_auc_score(y_holdout, holdout_scores)
        print(f"\nHoldout AUC ({holdout_q_dt.date()}): {auc_holdout:.3f}")
    else:
        auc_holdout = np.nan
        print("\nHoldout AUC undefined (no positives or only one class).")

    # ------------------
    # Portfolio construction
    # ------------------
    holdout_with_scores = df_holdout.copy()
    holdout_with_scores["prob_acq"] = holdout_scores
    holdout_with_scores = holdout_with_scores.dropna(subset=["ret_fwd_1q"])

    if holdout_with_scores.empty:
        print("No holdout names with non-missing forward 1-quarter return.")
        base_rate_holdout = df_holdout[TARGET_COL].mean() if len(df_holdout) > 0 else np.nan
        return {
            "quarter": holdout_qtr,
            "quarter_end": holdout_q_dt,
            "n_train": len(df_train),
            "n_holdout": len(df_holdout),
            "n_pos_train": n_pos_train,
            "n_pos_holdout": n_pos_holdout,
            "auc": auc_holdout,
            "n_bets": 0,
            "port_ret": np.nan,
            "universe_mean": np.nan,
            "random_mean": np.nan,
            "delta_vs_universe": np.nan,
            "delta_vs_random": np.nan,
            "deal_hit_rate": np.nan,
            "base_rate": base_rate_holdout,
        }

    # Base deal rate in holdout (if zero, fallback to train base rate for alpha)
    base_rate = df_holdout[TARGET_COL].mean()
    if np.isnan(base_rate) or base_rate == 0.0:
        base_rate = df_train[TARGET_COL].mean()
        if np.isnan(base_rate):
            base_rate = 0.0

    # Alpha and net_alpha
    holdout_with_scores["alpha_raw"] = (
        (holdout_with_scores["prob_acq"] - base_rate) * DEAL_PREMIUM_EST
    )
    holdout_with_scores["net_alpha"] = holdout_with_scores["alpha_raw"] - COST_PER_TRADE

    # Candidate pool: positive net_alpha first, sorted descending
    candidates = holdout_with_scores.sort_values("net_alpha", ascending=False).copy()
    candidates_pos = candidates[candidates["net_alpha"] > NET_ALPHA_MIN].copy()

    if candidates_pos.empty:
        # Fallback: use top-K_MAX by prob_acq if nothing clears net_alpha threshold
        print("\nNo names with positive net alpha; falling back to top-K_MAX by prob_acq.")
        candidates_pos = holdout_with_scores.sort_values("prob_acq", ascending=False).copy()

    # Limit candidate pool size
    pool_size = int(min(len(candidates_pos), max(K_MAX * POOL_MULT, K_MAX)))
    candidates_pool = candidates_pos.head(pool_size).copy()

    # Correlation matrix on trailing returns for candidate pool
    candidate_ciks = candidates_pool["cik"].unique().tolist()
    corr_mat = build_corr_matrix_for_candidates(candidate_ciks, holdout_qtr, lag_q=CORR_LAG_Q)

    # Greedy selection under correlation cap and net_alpha threshold
    selected = select_candidates_with_corr(
        candidates_pool,
        corr_mat,
        k_max=K_MAX,
        k_min=K_MIN,
        net_alpha_min=NET_ALPHA_MIN,
        rho_max=RHO_MAX,
    )

    # If we ended up with too few names, fallback to top-K_MIN by prob_acq
    if len(selected) < K_MIN:
        print(f"\nSelected < K_MIN={K_MIN} names under net_alpha/corr filters; "
              "falling back to top-K_MIN by prob_acq.")
        fallback_pool = holdout_with_scores.sort_values("prob_acq", ascending=False).head(K_MIN)
        sel = fallback_pool.copy()
    else:
        sel = selected.copy()

    # Final number of bets
    n_bets = len(sel)
    print(f"\nFinal number of bets in this quarter: {n_bets}")

    # Weights proportional to positive net_alpha (fallback to prob_acq if all non-positive)
    if "net_alpha" in sel.columns and (sel["net_alpha"] > 0).any():
        sel["w_raw"] = sel["net_alpha"].clip(lower=1e-6)
    else:
        sel["w_raw"] = sel["prob_acq"].clip(lower=1e-6)

    total_raw = sel["w_raw"].sum()
    sel["w_base"] = sel["w_raw"] / total_raw

    # Sector cap for comovement hedge
    sector_col = None
    for col in ["ggroup", "gind", "gsubind", "sic", "naics"]:
        if col in sel.columns:
            sector_col = col
            break

    if sector_col is not None:
        max_sector_weight = 0.25
        group_sum = sel.groupby(sector_col)["w_base"].transform("sum")
        scale = np.minimum(1.0, max_sector_weight / group_sum)
        sel["w_adj"] = sel["w_base"] * scale
        total_adj = sel["w_adj"].sum()
        if total_adj > 0:
            sel["weight"] = sel["w_adj"] / total_adj
        else:
            sel["weight"] = sel["w_base"]
        print(f"Applied sector cap using column '{sector_col}'.")
    else:
        sel["weight"] = sel["w_base"]
        print("No sector-ish column found; using net-alpha-weighted portfolio without sector caps.")

    print("\nSample of bets (first 10):")
    cols_to_show = [
        c for c in ["cik", "gvkey", "datadate", "prob_acq", "net_alpha", "weight", TARGET_COL]
        if c in sel.columns
    ]
    print(sel[cols_to_show].head(10))

    # Portfolio PnL (using clipped forward returns)
    portfolio_size = 1000.0
    ret_clip = clip_returns(sel["ret_fwd_1q"])
    sel["position_dollars"] = portfolio_size * sel["weight"]
    sel["pnl_dollars"] = sel["position_dollars"] * ret_clip

    portfolio_pnl = sel["pnl_dollars"].sum()
    portfolio_return = portfolio_pnl / portfolio_size

    print("\n=== Holdout Portfolio Performance (1-quarter horizon, clipped returns) ===")
    print(f"Total invested:       ${portfolio_size:,.2f}")
    print(f"Total PnL:            ${portfolio_pnl:,.2f}")
    print(f"Portfolio return:     {portfolio_return:.2%}")

    # Universe baseline (clipped)
    uni_ret_clip = clip_returns(df_holdout["ret_fwd_1q"]).dropna()
    if len(uni_ret_clip) > 0:
        universe_mean = uni_ret_clip.mean()
        print("\nUniverse forward 1Q returns (clipped) summary:")
        print(uni_ret_clip.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
    else:
        universe_mean = np.nan
        print("Universe has no valid clipped returns in this quarter.")

    # Random portfolio baseline (same K as our selected portfolio)
    if n_bets > 0:
        sim_rets = simulate_random_portfolios(
            df_holdout["ret_fwd_1q"],
            K=n_bets,
            n_sims=N_SIMS_RANDOM,
            seed=123,
        )
    else:
        sim_rets = None

    if sim_rets is not None:
        random_mean = sim_rets.mean()
        print("\nRandom equal-weight K-name portfolio baseline (clipped returns):")
        print(f"Mean random return:   {random_mean:.2%}")
        print(f"Median random return: {np.median(sim_rets):.2%}")
        print(f"5th pct random:       {np.percentile(sim_rets, 5):.2%}")
        print(f"95th pct random:      {np.percentile(sim_rets, 95):.2%}")
        print(f"\nModel portfolio vs random mean (difference): {(portfolio_return - random_mean):.2%}")
    else:
        random_mean = np.nan
        print("Not enough names for random portfolio simulation.")

    # Deal hit rates
    if TARGET_COL in sel.columns and len(sel) > 0:
        deal_hit_rate = sel[TARGET_COL].mean()
    else:
        deal_hit_rate = np.nan

    base_rate_holdout = df_holdout[TARGET_COL].mean() if len(df_holdout) > 0 else np.nan
    if not np.isnan(deal_hit_rate) and not np.isnan(base_rate_holdout):
        print(f"\nDeal hit rate in bets: {deal_hit_rate:.2%}")
        print(f"Deal base rate (holdout): {base_rate_holdout:.2%}")

    return {
        "quarter": holdout_qtr,
        "quarter_end": holdout_q_dt,
        "n_train": len(df_train),
        "n_holdout": len(df_holdout),
        "n_pos_train": n_pos_train,
        "n_pos_holdout": n_pos_holdout,
        "auc": auc_holdout,
        "n_bets": n_bets,
        "port_ret": portfolio_return,
        "universe_mean": universe_mean,
        "random_mean": random_mean,
        "delta_vs_universe": portfolio_return - universe_mean if not np.isnan(universe_mean) else np.nan,
        "delta_vs_random": portfolio_return - random_mean if not np.isnan(random_mean) else np.nan,
        "deal_hit_rate": deal_hit_rate,
        "base_rate": base_rate_holdout,
    }

# ==============================================================
# 6. Run across all holdout quarters and summarize
# ==============================================================

all_results = []
for qtr in holdout_quarters:
    res = run_quarter_experiment(qtr)
    all_results.append(res)

results_df = pd.DataFrame(all_results).sort_values("quarter_end").reset_index(drop=True)

print("\n======================== SUMMARY TABLE (HEAD) ========================")
summary_cols = [
    "quarter", "n_bets", "auc", "port_ret",
    "delta_vs_universe", "delta_vs_random",
    "deal_hit_rate", "base_rate"
]
print(results_df[summary_cols].head(10))

print("\n======================== SUMMARY TABLE (TAIL) ========================")
print(results_df[summary_cols].tail(10))

print("\n======================== OVERALL STATISTICS =========================")
print("Mean AUC (where defined):       ", results_df["auc"].dropna().mean())
print("Median AUC (where defined):     ", results_df["auc"].dropna().median())
print("Mean portfolio return:          ", results_df["port_ret"].dropna().mean())
print("Median portfolio return:        ", results_df["port_ret"].dropna().median())
print("Mean delta vs universe:         ", results_df["delta_vs_universe"].dropna().mean())
print("Mean delta vs random:           ", results_df["delta_vs_random"].dropna().mean())
print("Mean deal hit rate in bets:     ", results_df["deal_hit_rate"].dropna().mean())
print("Mean base deal rate (holdout):  ", results_df["base_rate"].dropna().mean())

# Cumulative returns
results_df["cum_model"] = (1.0 + results_df["port_ret"].fillna(0.0)).cumprod() - 1.0
results_df["cum_random_mean"] = (
    1.0 + results_df["random_mean"].fillna(0.0)
).cumprod() - 1.0

# ==============================================================
# 7. Visualizations
# ==============================================================

plt.figure(figsize=(12, 5))
plt.plot(results_df["quarter_end"], results_df["port_ret"], label="Model 1Q return")
plt.plot(results_df["quarter_end"], results_df["random_mean"], label="Random mean 1Q return", linestyle="--")
plt.axhline(0.0, color="black", linewidth=1)
plt.title("Per-quarter portfolio return: model vs random baseline")
plt.xlabel("Quarter end")
plt.ylabel("Quarterly return")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(results_df["quarter_end"], results_df["cum_model"], label="Model cumulative return")
plt.plot(results_df["quarter_end"], results_df["cum_random_mean"], label="Random cumulative (mean)", linestyle="--")
plt.axhline(0.0, color="black", linewidth=1)
plt.title("Cumulative return: model vs random-mean baseline")
plt.xlabel("Quarter end")
plt.ylabel("Cumulative return")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
valid_mask = ~results_df["auc"].isna() & ~results_df["delta_vs_random"].isna()
plt.scatter(
    results_df.loc[valid_mask, "auc"],
    results_df.loc[valid_mask, "delta_vs_random"]
)
plt.xlabel("Holdout AUC")
plt.ylabel("Delta vs random (quarterly return)")
plt.title("AUC vs excess return over random baseline")
plt.grid(True)
plt.show()
