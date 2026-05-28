# =========================================================
# MONDELEZ VC CHOCOLATES DASHBOARD  -  V4
# Dynamic monthly data — sales loaded from source, converted
# to pieces via an uploaded SKU/month divisor file (done ONCE
# at load-time and cached). User picks which months to focus
# the analysis on via a single month-picker (replaces the old
# fixed L3M / L15M tabs). All filters and column references
# are dynamic — missing columns are skipped, new columns are
# auto-discovered, and matching is case/whitespace-insensitive.
# =========================================================

import io
import json
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mondelez VC Chocolates Dashboard",
    layout="wide"
)

# =========================================================
# CONSTANTS
# =========================================================

COL_OUTLET = "OUTLETUID"

COL_ASM = "ASM_AREA"
COL_CHANNEL = "CHANNEL"
COL_PCTYPE = "PlipType"
COL_RD = "RD_CODE"
COL_RE = "RE"
COL_REGION = "Region"
COL_REGION_CAT = "Region_Cat"
COL_STATUS = "Status"
COL_VC = "VC_MODEL"
COL_VC_CATEGORY = "VC_CAT"
COL_SETTY = "SE_TTY"

COL_BRAND = "Brand"
COL_SKU = "Line Name"
COL_SKU_TIER = "SKU tier"
COL_LINE_NO = "Line No"

# Placeholder used when the source file carries no "SKU tier"
# column. SKU tier is OPTIONAL — we materialise the column with
# this single neutral value so every downstream reference to it
# (filters, Auto-Insights curation, Segment SKU Lists, the SKU
# Priority Lister) keeps working without special-casing. A tier
# multiselect built from this will simply show one option the
# user can ignore.
SKU_TIER_PLACEHOLDER = "Not specified"

# Source files may name the SKU stable-ID column in many ways
# ("Line No", "LINKNO", "Line_No", "LineNo", "Line#"…). We
# accept any of these and rename to COL_LINE_NO on load. Match
# is case/space/punct-insensitive via _normalise_token.
LINE_NO_CANDIDATES = [
    "Line No", "LineNo", "Line_No", "LINKNO", "LINK_NO",
    "Line#", "LineNumber", "Line Number", "LineID", "Line ID",
    "SKU Code", "SKUCode", "SKU_Code", "Product Code",
]

# Optional outlet-name column — used by the action list. The
# source file may or may not have one; we look for any of these
# names at load time and use whichever is present.
OUTLET_NAME_CANDIDATES = [
    "OUTLET_NAME", "OutletName", "Outlet_Name", "OUTLETNAME",
    "OUTLET", "Outlet", "Shop_Name", "SHOP_NAME"
]

# SKU-type classification — case-insensitive, brand-based.
# Cooler = Silk + Bournville + Temptations (the cooler-shelf
# premium portfolio). Ambient = everything else.
COOLER_BRANDS = {"silk", "bournville", "temptations"}


def classify_sku_type(brand_value):
    return (
        "Cooler"
        if str(brand_value).strip().lower() in COOLER_BRANDS
        else "Ambient"
    )


# =========================================================
# CUSTOM ORDERINGS for categorical-sheet column display
# =========================================================
# Applied to VC_CAT and Region/Region_Cat when they appear as
# the segment dimension. Match is case/space-insensitive
# against the values present in the source data; values not
# in the canonical list fall to the end in their natural sort
# order.
VC_CAT_ORDER = [
    # Left-to-right order: smallest VC to largest. The canonical
    # in-app labels are the explicit range form ("16-20L",
    # "30-35L", …, "700-1000L") that the loader rewrites the
    # source data to. The legacy prefixed (`<20L`, `>500L`) and
    # unprefixed (`20L`, `500L`) forms are also listed at the
    # same logical position so the fuzzy matcher (which strips
    # `<`, `>`, `-`, spaces, case) still places them together
    # regardless of which form happens to slip through.
    "16-20L",     "<20L",  "20L",
    "30-35L",     "<35L",  "35L",
    "40L",
    "45-70L",     "<70L",  "70L",
    "110-220L",   "<220L", "220L",
    "280-340L",   "<340L", "340L",
    "380-465L",   "<465L", "465L", "460L",
    "700-1000L",  ">500L", "500L"
]

REGION_ORDER = [
    "Ultra Premium", "Premium", "Semi Premium",
    "Regular", "Aspirational"
]

# =========================================================
# MUST HAVE / SHOULD HAVE / MAY NOT HAVE  —  tier-count mapping
# =========================================================
# Drives the Segment SKU Lists tab. For each VC_CAT segment
# value, defines how many SKUs (ranked by value-weighted score
# within that segment) belong in each tier:
#
#   Must Have   — top-N SKUs in the segment.
#   Should Have — next-M SKUs.
#   May Not Have — the **bottom 4** of the ranked list (the
#                  weakest performers in the segment that can
#                  realistically be dropped). This is fixed at
#                  4 across all segments per the user's spec.
#
# Pattern reflects the user's screenshot:
#   - 16-20L          : 3 / 3 / 4   (smallest cooler)
#   - 30-35L, 40L     : 4 / 4 / 4
#   - everything else : 4 / 5 / 4
#
# Keys are compared via _normalise_token (case/whitespace/
# comparison-prefix insensitive) so "16-20L" matches "16-20L"
# (the canonical post-load form). Legacy "<20L" / "20L" keys
# are kept too as a safety net in case raw source data ever
# slips past the loader rename.
#
# DEFAULT_TIER_COUNTS is used for any segment dimension that
# isn't VC_CAT, and for VC_CAT values that aren't in the
# explicit map (e.g. a future cooler band the file picks up
# that wasn't in the original spec).
#
# NOTE on naming: the in-code variable names (`must_n`,
# `should_n`, `can_n` and `must_df`, `should_df`, `can_df`)
# are kept for back-compatibility with downstream report
# code; only the user-facing labels are renamed.
MAY_NOT_HAVE_N = 4  # fixed: always the last 4 in the ranking

MUST_SHOULD_CAN_BY_VC_CAT = {
    # New canonical labels (post-load)
    "16-20L":    (3, 3, MAY_NOT_HAVE_N),
    "30-35L":    (4, 4, MAY_NOT_HAVE_N),
    "40L":       (4, 4, MAY_NOT_HAVE_N),
    "45-70L":    (4, 5, MAY_NOT_HAVE_N),
    "110-220L":  (4, 5, MAY_NOT_HAVE_N),
    "280-340L":  (4, 5, MAY_NOT_HAVE_N),
    "380-465L":  (4, 5, MAY_NOT_HAVE_N),
    "700-1000L": (4, 5, MAY_NOT_HAVE_N),
    # Legacy labels (kept as a safety net)
    "<20L":      (3, 3, MAY_NOT_HAVE_N),
    "<35L":      (4, 4, MAY_NOT_HAVE_N),
    "<70L":      (4, 5, MAY_NOT_HAVE_N),
    "<220L":     (4, 5, MAY_NOT_HAVE_N),
    "<340L":     (4, 5, MAY_NOT_HAVE_N),
    "<465L":     (4, 5, MAY_NOT_HAVE_N),
    ">500L":     (4, 5, MAY_NOT_HAVE_N),
    "OWN A/C":   (4, 5, MAY_NOT_HAVE_N),
    "NONE":      (4, 5, MAY_NOT_HAVE_N),
}
DEFAULT_TIER_COUNTS = (4, 5, MAY_NOT_HAVE_N)


def get_tier_counts(segment_col, segment_value):
    """
    Return (must_n, should_n, can_n) for a (segment dimension,
    segment value) pair. For VC_CAT, consults the explicit map
    above with fuzzy-key matching; for any other dimension, falls
    back to the default. `can_n` here is the May-Not-Have slot
    count (i.e. how many SKUs from the *bottom* of the ranking
    to surface as droppable).
    """
    if segment_col != COL_VC_CATEGORY:
        return DEFAULT_TIER_COUNTS
    target_key = _normalise_token(segment_value)
    for k, v in MUST_SHOULD_CAN_BY_VC_CAT.items():
        if _normalise_token(k) == target_key:
            return v
    return DEFAULT_TIER_COUNTS

# UI display-name overrides — column → label shown to the user.
# The underlying DataFrame column name stays the same; only the
# label rendered in widgets / report headings changes.
COL_DISPLAY_NAMES = {
    "PlipType": "PC Type"
}


def display_name(col):
    """Return the user-facing label for a column."""
    return COL_DISPLAY_NAMES.get(col, col)


def _normalise_token(s):
    """Lower-cased, whitespace + punctuation-stripped key for
    fuzzy matching. Strips comparison prefixes (`<`, `>`, `≤`,
    `≥`) AND common punctuation that varies between data
    sources (apostrophe, dash, dot, slash, comma, underscore)
    so that "Jan'25", "Jan 25", "jan-25", "Jan_25" all
    collapse to the same key. Also strips spaces.
    """
    raw = str(s).lower()
    for ch in (
        "<", ">", "≤", "≥", "=", "~",
        "'", "’", "`", "´",          # apostrophe variants
        "-", "_", ".", ",", "/", "\\",  # common separators
    ):
        raw = raw.replace(ch, "")
    return "".join(raw.split())


def _norm_series(s):
    """Vectorised version of _normalise_token for pandas Series.
    Used to make filter matching case + whitespace + light-
    spelling-variant insensitive on the data side.
    """
    out = s.astype(str).str.lower()
    for ch in (
        "<", ">", "≤", "≥", "=", "~",
        "'", "’", "`", "´",
        "-", "_", ".", ",", "/", "\\",
    ):
        out = out.str.replace(ch, "", regex=False)
    # Collapse all whitespace (incl. internal) — same as
    # "".join(s.split()) but vectorised.
    out = out.str.replace(r"\s+", "", regex=True)
    return out


def fuzzy_isin(series, values):
    """Case / whitespace / comparison-prefix / common-punctuation
    insensitive .isin(). Returns a boolean mask the same length
    as `series`. Both sides are normalised via _normalise_token
    before comparison so 'High End Grocer', 'HIGH END GROCER ',
    'highendgrocer', 'High-End-Grocer' all match.
    """
    if not values:
        return pd.Series(True, index=series.index)
    norm_values = {_normalise_token(v) for v in values}
    return _norm_series(series).isin(norm_values)


def order_segment_values(segment_col, values):
    """
    Return `values` sorted according to the custom ordering for
    `segment_col` (VC_CAT, Region, Region_Cat). Values not in
    the canonical list keep at the end in their natural sort
    order. Falls back to the natural sort of `values` when no
    custom ordering applies.
    """
    values = list(values)
    if segment_col == COL_VC_CATEGORY:
        canon = VC_CAT_ORDER
    elif segment_col in (COL_REGION, COL_REGION_CAT):
        canon = REGION_ORDER
    else:
        return sorted(values)

    # Build normalised-key → first-seen index so prefix variants
    # (e.g. "<20L" and "20L") that normalise to the same key share
    # an index and sort together in the order they first appear.
    canon_index = {}
    for i, c in enumerate(canon):
        key = _normalise_token(c)
        canon_index.setdefault(key, i)

    in_canon, leftovers = [], []
    for v in values:
        idx = canon_index.get(_normalise_token(v))
        if idx is None:
            leftovers.append(v)
        else:
            in_canon.append((idx, v))
    in_canon.sort(key=lambda t: t[0])
    return [v for _, v in in_canon] + sorted(leftovers)

# Time-period totals available in the source file.
# Each entry maps the column to the number of months it covers,
# which is used to normalise Throughput to a monthly average so
# the two sheets are directly comparable.
PERIOD_COLS = {
    "L3M":  ("L3M",  3),
    "L15M": ("L15M", 15)
}

# Monthly volume columns — used to compute "months active" per
# SKU so we can filter out seasonal / one-shot promo SKUs.
#
# These two lists are DYNAMIC: they're populated by
# `_detect_month_cols()` at load-time from whatever monthly
# columns the source file actually contains. The file structure
# stays the same — "MonYY_V" volume columns and "Mon'YY" MRP
# columns — but the *set* of months is discovered, not
# hardcoded. So a file covering Apr'24 → Jun'26 works without
# code changes, exactly the same way a Jan'25 → Mar'26 file
# does.
#
# They start empty and get filled in by `load_data` on the very
# first call (and on every subsequent call against a different
# file, since the lists are rebuilt from df.columns). Downstream
# code reads them as before — every reference is a runtime
# `MONTH_COLS` lookup, not a closure over the original list, so
# late population is fine.
MONTH_COLS = []   # e.g. ["Jan25_V", "Feb25_V", …]  in chronological order

# =========================================================
# MRP FILE  (uploaded — used by value-based smoothing only)
# =========================================================
# The MRP master is uploaded by the user via the sidebar (see the
# file_uploader below). Parsed ONCE per session via
# st.cache_resource keyed on the raw bytes — does not reload on
# every rerun.
#
# Expected schema (see screenshot):
#   col A : LINKNO
#   col B : Line Name
#   col C..: monthly MRP columns named "Jan'25", "Feb'25", … etc
#           (one column per month, in the same order as MONTH_COLS).
# Multiple LINKNOs can map to the same Line Name; we average MRP
# across them per month for the join.

# MRP-file column names paired 1:1 with MONTH_COLS above. Note the
# apostrophe — that's how they appear in the source file. Also
# dynamic; built alongside MONTH_COLS from the discovered set of
# (month, year) pairs.
MRP_MONTH_COLS = []   # e.g. ["Jan'25", "Feb'25", …]  in chronological order


# =========================================================
# DYNAMIC MONTH DISCOVERY
# =========================================================
# Scans a column-name iterable for monthly markers of the form
# "MonYY" (Jan25, FEB-25, Mar'25, Apr_25, may25, Jun 25, …) with
# an optional trailing "_V" / " V" / "V" suffix that flags the
# volume columns. Returns the discovered months in chronological
# order, formatted both ways:
#
#   ([volume column names like "Jan25_V"],
#    [MRP column names like "Jan'25"])
#
# Year handling: a 2-digit year < 50 is treated as 20YY, ≥ 50
# as 19YY (so "Jul25" → 2025, "Jul99" → 1999). 4-digit years
# pass through. This keeps the function future-proof past 2049
# without needing edits — but the dashboard's data is always
# this-decade so the boundary is academic.

_MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_MONTH_NUM = {name.lower(): i + 1 for i, name in enumerate(_MONTH_NAMES)}


def _detect_month_cols(columns):
    """
    Find every "MonYY" column in `columns` and return the two
    canonical lists used everywhere downstream:

        volume_cols  — "Jan25_V" style, one per discovered month
        mrp_cols     — "Jan'25"  style, paired 1:1

    Both lists are sorted chronologically by (year, month). The
    matcher is case / whitespace / punctuation insensitive — same
    rules as _normalise_token plus a trailing "v" stripper — so
    raw column names like "JAN-25 V", "jan_25", "Jan'25 V" all
    map to the canonical "Jan25_V".

    Pure function: returns the lists, doesn't mutate globals.
    The caller (`load_data`) decides when to publish them.
    """
    import re
    # Pattern: month abbreviation, optional separator, year
    # (2 or 4 digits). Anchored to the start of the normalised
    # key. We work on the normalised key (lowercased, separators
    # stripped) so "Jan-25", "Jan'25", "JAN25" all reduce to
    # "jan25" before the regex runs.
    rx = re.compile(
        r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"(\d{2}|\d{4})"
        r"v?$"   # optional trailing 'v' (volume marker)
    )

    seen = {}   # (year, month_num) → True; dedupes if a file lists
    #                                  the same month under two names
    for col in columns:
        key = _normalise_token(col)        # strips '-_./\, etc.
        m = rx.match(key)
        if not m:
            continue
        month_name = m.group(1)
        year_raw = m.group(2)
        month_num = _MONTH_NUM[month_name]
        if len(year_raw) == 2:
            yy = int(year_raw)
            year = 2000 + yy if yy < 50 else 1900 + yy
        else:
            year = int(year_raw)
        seen[(year, month_num)] = True

    if not seen:
        return [], []

    # Chronological order = sort the (year, month) tuples.
    ordered = sorted(seen.keys())
    volume_cols = []
    mrp_cols = []
    for year, month_num in ordered:
        mon = _MONTH_NAMES[month_num - 1]
        yy = year % 100
        volume_cols.append(f"{mon}{yy:02d}_V")
        mrp_cols.append(f"{mon}'{yy:02d}")
    return volume_cols, mrp_cols

# All categorical/dimension columns that the dashboard filters on
DIM_COLS = [
    COL_OUTLET,
    COL_ASM,
    COL_CHANNEL,
    COL_PCTYPE,
    COL_RD,
    COL_RE,
    COL_REGION,
    COL_REGION_CAT,
    COL_STATUS,
    COL_VC,
    COL_VC_CATEGORY,
    COL_SETTY,
    COL_BRAND,
    COL_SKU,
    COL_SKU_TIER
]

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    """
    Read the uploaded source file (parquet or csv) into a
    DataFrame. Cached on the raw bytes so repeated reruns
    don't re-parse the file.
    """
    bio = io.BytesIO(file_bytes)
    name_lower = (file_name or "").lower()

    if name_lower.endswith(".csv"):
        df = pd.read_csv(bio)
    else:
        # default to parquet for .parquet / .pq / unknown
        df = pd.read_parquet(bio)

    # Normalise monthly-volume column names so the rest of the
    # code can rely on a canonical "MonYY_V" form ("Jan25_V",
    # "Feb25_V", ...). The source file may name them in many
    # ways: "JAN25", "Jan25", "Jan 25", "Jan'25", "Jan-25",
    # "Jan_25", with or without a trailing "_V" / " V" / "V"
    # suffix, in any case. We match case/whitespace/punctuation-
    # insensitively and rename to the canonical form so downstream
    # code (MRP-join, divisor-join, throughput math, month-picker)
    # finds the columns regardless of source-file casing.
    #
    # DYNAMIC DISCOVERY: the *set* of months is read from the file
    # itself rather than being hardcoded. We populate the module-
    # level MONTH_COLS / MRP_MONTH_COLS globals here so every
    # downstream reference (which uses `MONTH_COLS` as a name, not
    # a snapshot) sees the discovered set. The lists are sorted
    # chronologically by (year, month).
    def _month_key(s):
        # Same idea as _normalise_token, but also strips a
        # trailing single "v" so "Jan25_V" and "JAN25" collapse
        # to the same key ("jan25").
        k = _normalise_token(s)
        if k.endswith("v"):
            k = k[:-1]
        return k

    # Step 1 — discover which months the file actually carries,
    # in chronological order, and build the canonical name pair
    # (volume + MRP).
    _vol_cols, _mrp_cols = _detect_month_cols(df.columns)

    # Step 2 — publish to module globals so every downstream
    # `MONTH_COLS` / `MRP_MONTH_COLS` reference sees the
    # discovered set. We MUTATE the existing list objects (clear
    # + extend) rather than rebinding the names, because some
    # downstream modules may already hold a reference to the
    # original list object.
    global MONTH_COLS, MRP_MONTH_COLS
    MONTH_COLS.clear()
    MONTH_COLS.extend(_vol_cols)
    MRP_MONTH_COLS.clear()
    MRP_MONTH_COLS.extend(_mrp_cols)

    # Step 3 — rename the file's actual columns to the canonical
    # "MonYY_V" form. Build the canonical lookup from the
    # discovered set (not a hardcoded list).
    _canon_by_key = {_month_key(m): m for m in MONTH_COLS}
    _rename_months = {}
    for actual_col in df.columns:
        key = _month_key(actual_col)
        canon = _canon_by_key.get(key)
        if canon is not None and actual_col != canon:
            _rename_months[actual_col] = canon
    if _rename_months:
        df = df.rename(columns=_rename_months)

    # Club VC_Category "<33L" into the "30-35L" bucket, and rename
    # all VC_Category labels from the legacy "<X L" form to the new
    # explicit range form so they read more clearly across the app.
    #   <20L   → 16-20L
    #   <35L   → 30-35L   (also catches the old <33L)
    #   <70L   → 45-70L
    #   <220L  → 110-220L
    #   <340L  → 280-340L
    #   <465L  → 380-465L
    #   >500L  → 700-1000L
    if COL_VC_CATEGORY in df.columns:
        df[COL_VC_CATEGORY] = (
            df[COL_VC_CATEGORY]
            .astype(str)
            .str.strip()
            .replace({
                "<33L":  "30-35L",
                "<20L":  "16-20L",
                "<35L":  "30-35L",
                "<70L":  "45-70L",
                "<220L": "110-220L",
                "<340L": "280-340L",
                "<465L": "380-465L",
                ">500L": "700-1000L",
            })
        )

    # Rename SKU tier labels from "Tier N" to explicit value bands
    # so the in-app display matches the business definition.
    #   Tier 1 → Greater than 200
    #   Tier 2 → 120-150
    #   Tier 3 → 80-110
    #   Tier 4 → 35-60
    #   Tier 5 → Less than 35
    if COL_SKU_TIER in df.columns:
        df[COL_SKU_TIER] = (
            df[COL_SKU_TIER]
            .astype(str)
            .str.strip()
            .replace({
                "Tier 1": "Greater than 200",
                "Tier 2": "120-150",
                "Tier 3": "80-110",
                "Tier 4": "35-60",
                "Tier 5": "Less than 35",
                "tier 1": "Greater than 200",
                "tier 2": "120-150",
                "tier 3": "80-110",
                "tier 4": "35-60",
                "tier 5": "Less than 35",
                "T1":     "Greater than 200",
                "T2":     "120-150",
                "T3":     "80-110",
                "T4":     "35-60",
                "T5":     "Less than 35",
            })
        )

    # SKU tier is OPTIONAL. If the source file has no "SKU tier"
    # column, materialise it with a single neutral placeholder so
    # the ~dozen downstream references (sidebar visual filter,
    # Auto-Insights tier restriction, Segment SKU Lists tier
    # filter, the SKU Priority Lister tier scope, the categorical
    # sheet) all keep working instead of raising KeyError. Also
    # normalise blank / NaN tiers to the placeholder so the column
    # never carries empty strings.
    if COL_SKU_TIER not in df.columns:
        df[COL_SKU_TIER] = SKU_TIER_PLACEHOLDER
    else:
        _tier_str = df[COL_SKU_TIER].astype(str).str.strip()
        df[COL_SKU_TIER] = _tier_str.where(
            (_tier_str != "")
            & (_tier_str.str.lower() != "nan")
            & (_tier_str.str.lower() != "none"),
            other=SKU_TIER_PLACEHOLDER,
        )

    for col, _months in PERIOD_COLS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    # Make sure L15M / L3M always exist for downstream code that
    # references them by name even if the user uploaded a file
    # missing one of them.
    for col, _ in PERIOD_COLS.values():
        if col not in df.columns:
            df[col] = 0.0

    # Coerce monthly volume columns numerically so smoothing
    # / trend logic works on uploaded files where they may have
    # come through as strings.
    for m in MONTH_COLS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

    # Normalise casing on text columns prone to spelling drift
    # (e.g. "HIGH END GROCER" vs "High end Grocer" in RE).
    for txt_col in [COL_RE, COL_CHANNEL, COL_PCTYPE]:
        if txt_col in df.columns:
            df[txt_col] = (
                df[txt_col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

    return df


def detect_outlet_name_col(df):
    """Return whichever outlet-name column exists in df, or None."""
    for c in OUTLET_NAME_CANDIDATES:
        if c in df.columns:
            return c
    return None


# =========================================================
# MULTI-FILE BASE LOADER  (concatenates up to N base files)
# =========================================================

@st.cache_data(show_spinner=False)
def load_and_merge_base_files(parts):
    """
    Load one or more base files and concatenate them row-wise
    into a single DataFrame, returning (merged_bytes, merged_name,
    report).

    Each `parts` entry is a (file_bytes, file_name) tuple. Empty
    or None entries are skipped, so the caller can pass three
    slots blindly even when only the first is filled.

    Loading goes through load_data() so each part is independently
    normalised (canonical MonYY_V column names, VC_CAT relabelling
    from "<X L" to "X-Y L", SKU-tier relabelling from "Tier N" to
    explicit value bands, text-column casing, numeric coercion of
    period totals and monthly volumes). After that the parts are
    concatenated with sort=False so columns from any part are kept;
    columns absent from a part fill with NaN for that part's rows.

    The merged frame is then serialised back to parquet bytes so
    we can hand it to the existing load_data_pieces() pipeline as
    if it were a single uploaded file. This means downstream code
    paths (divisor join, SKU-master join, MRP/GM/size joins, all
    the filters and tabs) need no changes whatsoever — they just
    see one bigger DataFrame.

    Cached on the tuple of file bytes + names, so adding/removing
    one of the optional uploads re-runs the merge; toggling other
    sidebar filters does not.

    report = {
        "ok":         bool,
        "msg":        human-readable summary string,
        "n_files":    how many files were merged,
        "row_counts": [n_rows per file, in upload order],
        "total_rows": int,
    }
    """
    # Filter out empty slots — accept a (bytes, name) tuple only
    # when bytes is non-empty.
    real_parts = [
        (b, n) for (b, n) in parts
        if b is not None and len(b) > 0
    ]
    if not real_parts:
        return None, None, {
            "ok": False,
            "msg": "No base file uploaded.",
            "n_files": 0,
            "row_counts": [],
            "total_rows": 0,
        }

    # Single-file fast path: skip the round-trip through parquet
    # bytes when there's nothing to merge. This keeps the original
    # single-file behaviour byte-for-byte identical to before this
    # multi-file feature was added.
    if len(real_parts) == 1:
        b, n = real_parts[0]
        # We still validate it can be parsed so the error surfaces
        # here rather than deep inside load_data_pieces().
        try:
            _check = load_data(b, n)
            n_rows = len(_check)
        except Exception as e:
            return None, None, {
                "ok": False,
                "msg": f"Could not read base file '{n}': {e}",
                "n_files": 0,
                "row_counts": [],
                "total_rows": 0,
            }
        return b, n, {
            "ok": True,
            "msg": f"Loaded 1 base file ({n_rows:,} rows).",
            "n_files": 1,
            "row_counts": [n_rows],
            "total_rows": n_rows,
        }

    # Multi-file path: load each via load_data() so per-file
    # normalisation runs, then concatenate.
    frames = []
    row_counts = []
    names = []
    for b, n in real_parts:
        try:
            part_df = load_data(b, n).copy()
        except Exception as e:
            return None, None, {
                "ok": False,
                "msg": f"Could not read base file '{n}': {e}",
                "n_files": 0,
                "row_counts": [],
                "total_rows": 0,
            }
        frames.append(part_df)
        row_counts.append(len(part_df))
        names.append(n)

    try:
        merged = pd.concat(frames, axis=0, ignore_index=True,
                           sort=False)
    except Exception as e:
        return None, None, {
            "ok": False,
            "msg": f"Could not concatenate base files: {e}",
            "n_files": 0,
            "row_counts": [],
            "total_rows": 0,
        }

    # Serialise back to parquet bytes so the downstream
    # load_data_pieces pipeline can consume it unchanged. We use
    # parquet (not csv) so column dtypes round-trip exactly and
    # the big monthly-volume matrix stays compact.
    out_bio = io.BytesIO()
    try:
        merged.to_parquet(out_bio, index=False)
    except Exception:
        # Fallback for environments without pyarrow / fastparquet:
        # round-trip via CSV. Slower and slightly lossier on dtypes
        # (numeric coercion in load_data will re-cast the monthly
        # columns anyway), but keeps the feature working.
        out_bio = io.BytesIO()
        merged.to_csv(out_bio, index=False)
        merged_name = "merged_base_files.csv"
    else:
        merged_name = "merged_base_files.parquet"

    merged_bytes = out_bio.getvalue()
    total_rows = int(sum(row_counts))
    msg_parts = ", ".join(
        f"{nm} ({rc:,} rows)"
        for nm, rc in zip(names, row_counts)
    )
    return merged_bytes, merged_name, {
        "ok": True,
        "msg": (
            f"Merged {len(real_parts)} base files into "
            f"{total_rows:,} rows ({msg_parts})."
        ),
        "n_files": len(real_parts),
        "row_counts": row_counts,
        "total_rows": total_rows,
    }


# =========================================================
# MRP LOADER  (local file, cached for the whole session)
# =========================================================

@st.cache_resource(show_spinner=False)
def load_mrp_table(mrp_bytes, mrp_name):
    """
    Parse the uploaded MRP master once per session and return a
    DataFrame indexed by lower-cased Line Name with one column per
    monthly MRP.

    Cached via @st.cache_resource (not cache_data) so the parsed
    table is held by reference — effectively free after the first
    call. The cache key is the raw uploaded bytes, so the table is
    re-parsed only if the user uploads a different MRP file.

    Returns (mrp_lookup, status_message). On any failure returns
    (None, error_string) so the caller can fall back to volume-
    based smoothing without crashing.
    """
    if mrp_bytes is None:
        return None, "No MRP file uploaded."

    bio = io.BytesIO(mrp_bytes)
    name_lower = (mrp_name or "").lower()

    try:
        if name_lower.endswith((".xlsx", ".xls", ".xlsm")):
            mrp_df = pd.read_excel(bio)
        elif name_lower.endswith(".csv"):
            mrp_df = pd.read_csv(bio)
        elif name_lower.endswith((".parquet", ".pq")):
            mrp_df = pd.read_parquet(bio)
        else:
            # Best-effort: try Excel first, then CSV.
            try:
                bio.seek(0)
                mrp_df = pd.read_excel(bio)
            except Exception:
                bio.seek(0)
                mrp_df = pd.read_csv(bio)
    except Exception as exc:
        return None, f"Could not read MRP file: {exc}"

    # Fuzzy-locate the SKU column (case/space/punct-insensitive)
    # so MRP files that label it "line name", "LINE_NAME",
    # "Line-Name", etc. still work.
    _mrp_col_keys = {
        _normalise_token(c): c for c in mrp_df.columns
    }
    sku_col_actual = _mrp_col_keys.get(_normalise_token(COL_SKU))
    if sku_col_actual is None:
        return None, (
            f"MRP file missing '{COL_SKU}' column. Found: "
            f"{list(mrp_df.columns)[:6]}…"
        )

    # Match month columns case/space/punct-insensitively against
    # the canonical MRP_MONTH_COLS list. This is forgiving of
    # files where someone typed "Jan 25", "jan-25", "JAN'25",
    # "Jan_25", etc. instead of the canonical "Jan'25". Without
    # this fuzzy step, the strict `c in mrp_df.columns` test
    # below would silently drop every month column and downstream
    # MRP values would all come back NaN — which surfaces as
    # "No SKUs have a positive TP × MRP × GM score" on the
    # Ranked SKU list. Mirrors the divisor loader's behaviour.
    month_map = {}     # canonical month → actual column in file
    missing = []
    for canon in MRP_MONTH_COLS:
        actual = _mrp_col_keys.get(_normalise_token(canon))
        if actual is not None:
            month_map[canon] = actual
        else:
            missing.append(canon)

    # Keep only the columns we need; rename to canonical form so
    # downstream lookups (`mrp_aligned[mcol]` with mcol = "Jan'25")
    # find them.
    keep_actual = [sku_col_actual] + list(month_map.values())
    mrp_df = mrp_df[keep_actual].copy()
    rename_map = {v: k for k, v in month_map.items()}
    rename_map[sku_col_actual] = COL_SKU
    mrp_df = mrp_df.rename(columns=rename_map)

    keep = [COL_SKU] + list(month_map.keys())
    for c in keep[1:]:
        mrp_df[c] = pd.to_numeric(mrp_df[c], errors="coerce")

    # Build the normalised join key FIRST (strip + lowercase), then
    # groupby on it. Doing it in this order is essential: if we
    # grouped on raw Line Name first and lowercased afterwards,
    # variants that differ only in case/whitespace (e.g. "FS 20 "
    # vs "FS 20") would survive groupby and collapse only at the
    # rename step — producing a non-unique index that breaks the
    # downstream .reindex / .loc lookup with
    # "cannot reindex on an axis with duplicate labels".
    mrp_df["_join_key"] = (
        mrp_df[COL_SKU].astype(str).str.strip().str.lower()
    )
    mrp_avg = (
        mrp_df.groupby("_join_key", as_index=True)[keep[1:]]
        .mean()
    )

    # Final safety net: collapse any remaining duplicates (e.g.
    # NaN keys) by averaging again. After this the index is
    # guaranteed unique, which .reindex requires.
    if not mrp_avg.index.is_unique:
        mrp_avg = mrp_avg.groupby(level=0).mean()

    msg = f"Loaded MRPs for {len(mrp_avg):,} SKUs."
    if missing:
        msg += f" (Missing month columns: {missing})"

    return mrp_avg, msg


# =========================================================
# MRP-DERIVED LINE NO → LINE NAME LOOKUP
# =========================================================
# The Final MRP Tracker file already carries both LINKNO and
# Line Name (plus monthly MRPs). We use that pairing as our
# canonical Line No → Line Name lookup so the user no longer
# needs to upload a separate SKU master file. Cached on the
# raw MRP bytes — parsed once per session.

@st.cache_resource(show_spinner=False)
def load_lineno_to_name_from_mrp(mrp_bytes, mrp_name):
    """
    Parse the MRP file and return (lineno_to_name, status_msg).

    lineno_to_name is a pandas Series indexed by normalised
    Line No (lower-cased + stripped) with values = canonical
    Line Name strings. Used by load_data_pieces to attach
    human-readable names to the base file's rows.

    Returns (None, error_msg) on any failure so the caller can
    decide whether to fall back to the base file's own Line
    Name column (if present).
    """
    if mrp_bytes is None:
        return None, "No MRP file uploaded."

    bio = io.BytesIO(mrp_bytes)
    name_lower = (mrp_name or "").lower()

    try:
        if name_lower.endswith((".xlsx", ".xls", ".xlsm")):
            mrp_df = pd.read_excel(bio)
        elif name_lower.endswith(".csv"):
            mrp_df = pd.read_csv(bio)
        elif name_lower.endswith((".parquet", ".pq")):
            mrp_df = pd.read_parquet(bio)
        else:
            try:
                bio.seek(0)
                mrp_df = pd.read_excel(bio)
            except Exception:
                bio.seek(0)
                mrp_df = pd.read_csv(bio)
    except Exception as exc:
        return None, f"Could not read MRP file: {exc}"

    lineno_col = _find_col(mrp_df.columns, LINE_NO_CANDIDATES)
    name_col = _find_col(
        mrp_df.columns,
        [COL_SKU, "LineName", "Line_Name", "Line Name"],
    )

    if lineno_col is None or name_col is None:
        return None, (
            "MRP file does not have both a Line No / LINKNO "
            "column and a Line Name column — cannot derive "
            "Line No → Line Name mapping."
        )

    keep = mrp_df[[lineno_col, name_col]].copy()
    keep.columns = [COL_LINE_NO, COL_SKU]

    keep[COL_LINE_NO] = keep[COL_LINE_NO].astype(str).str.strip()
    keep[COL_SKU] = keep[COL_SKU].astype(str).str.strip()

    keep = keep[
        keep[COL_LINE_NO].notna()
        & (keep[COL_LINE_NO] != "")
        & (keep[COL_LINE_NO].str.lower() != "nan")
        & keep[COL_SKU].notna()
        & (keep[COL_SKU] != "")
        & (keep[COL_SKU].str.lower() != "nan")
    ]

    if keep.empty:
        return None, (
            "MRP file has no valid (Line No, Line Name) rows."
        )

    keep["_join_key"] = (
        keep[COL_LINE_NO].astype(str).str.strip().str.lower()
    )
    keep = keep.drop_duplicates(subset="_join_key", keep="first")
    lineno_to_name = keep.set_index("_join_key")[COL_SKU]

    msg = (
        f"Derived Line No → Line Name mapping from MRP file: "
        f"{len(lineno_to_name):,} entries."
    )
    return lineno_to_name, msg


# =========================================================
# DIVISOR (SALES → PIECES CONVERSION) LOADER
# =========================================================
# The source file's monthly columns now hold SALES (₹), not
# pieces. To get pieces we divide each (SKU, month) cell by a
# per-SKU per-month divisor (typically the average price per
# piece for that SKU in that month). The user uploads this
# divisor file separately. Same schema as the MRP file:
#   col 1 : Line Name (SKU)
#   col 2+: one column per month named Jan'25, Feb'25, …
#           (apostrophe form — paired 1:1 with MONTH_COLS).
#
# SKUs missing from the divisor file are EXCLUDED from analysis
# (and the loader records their names for a sidebar warning).
# Cached with @st.cache_resource on the raw bytes — parsed once
# per session.

def _find_col(df_columns, candidates):
    """Return the actual column name in `df_columns` matching any
    of `candidates` (case/space/punct-insensitively). None if no
    match. Useful for source files that name columns slightly
    differently across uploads."""
    norm_to_actual = {
        _normalise_token(c): c for c in df_columns
    }
    for cand in candidates:
        actual = norm_to_actual.get(_normalise_token(cand))
        if actual is not None:
            return actual
    return None


# =========================================================
# SKU MASTER LOADER  (Line No → Line Name)
# =========================================================
# The base source file holds Line No only (stable SKU ID).
# Human-readable Line Name lives in a separate SKU master file
# that the user uploads. This loader parses the master and
# returns a Series indexed by normalised Line No → Line Name.
#
# The master file is OPTIONAL when the base file already
# carries Line Name; REQUIRED otherwise. The main flow handles
# that decision — this loader is just a parser.

@st.cache_resource(show_spinner=False)
def load_sku_master_table(master_bytes, master_name):
    """
    Parse the uploaded SKU master file.

    Schema (flexible column names):
      - Line No column: any of LINE_NO_CANDIDATES
      - Line Name column: COL_SKU (or case-variant of it)

    Returns (lineno_to_name, status_message) where
    lineno_to_name is a pandas Series indexed by normalised
    Line No (lower-cased + stripped of whitespace) with values
    = canonical Line Name strings.

    On any failure returns (None, error_string).
    """
    if master_bytes is None:
        return None, "No SKU master file uploaded."

    bio = io.BytesIO(master_bytes)
    name_lower = (master_name or "").lower()

    try:
        if name_lower.endswith((".xlsx", ".xls", ".xlsm")):
            m_df = pd.read_excel(bio)
        elif name_lower.endswith(".csv"):
            m_df = pd.read_csv(bio)
        elif name_lower.endswith((".parquet", ".pq")):
            m_df = pd.read_parquet(bio)
        else:
            try:
                bio.seek(0)
                m_df = pd.read_excel(bio)
            except Exception:
                bio.seek(0)
                m_df = pd.read_csv(bio)
    except Exception as exc:
        return None, f"Could not read SKU master file: {exc}"

    lineno_col = _find_col(m_df.columns, LINE_NO_CANDIDATES)
    name_col = _find_col(m_df.columns, [COL_SKU, "LineName", "Line_Name"])
    # Brand is OPTIONAL on the master — if found we'll attach it
    # as a side-channel on the returned Series so the caller can
    # use it when the base file doesn't carry Brand itself.
    brand_col = _find_col(
        m_df.columns,
        [COL_BRAND, "BRAND", "brand", "Brand Name", "BrandName"],
    )

    # If we couldn't find by canonical names, fall back to
    # positional: assume col 1 = Line No, col 2 = Line Name.
    if lineno_col is None and len(m_df.columns) >= 1:
        lineno_col = m_df.columns[0]
    if name_col is None and len(m_df.columns) >= 2:
        name_col = m_df.columns[1]

    if lineno_col is None or name_col is None:
        return None, (
            "SKU master file needs at least 2 columns "
            "(Line No, Line Name). "
            f"Found: {list(m_df.columns)[:6]}…"
        )

    # Build the lookup: normalised Line No → trimmed Line Name.
    # If a Brand column was found we carry it through too so we
    # can attach it as .attrs["brand_lookup"] on the returned Series.
    keep_cols = [lineno_col, name_col]
    if brand_col is not None and brand_col not in keep_cols:
        keep_cols.append(brand_col)
    keep = m_df[keep_cols].copy()
    # Rename the first two to canonical names; leave the brand
    # column (if any) under its original name and rename it next.
    rename_map = {lineno_col: COL_LINE_NO, name_col: COL_SKU}
    if brand_col is not None:
        rename_map[brand_col] = COL_BRAND
    keep = keep.rename(columns=rename_map)

    # Drop rows where the Line No is blank/NaN — they can't act
    # as a join key.
    keep[COL_LINE_NO] = keep[COL_LINE_NO].astype(str).str.strip()
    keep = keep[
        keep[COL_LINE_NO].notna()
        & (keep[COL_LINE_NO] != "")
        & (keep[COL_LINE_NO].str.lower() != "nan")
    ]

    # Drop rows where the Line Name is blank — we'd otherwise
    # attach an empty string back to the base, which gets
    # dropped downstream as "missing name" anyway. Better to
    # surface this loss at master-load time.
    keep[COL_SKU] = keep[COL_SKU].astype(str).str.strip()
    keep = keep[
        keep[COL_SKU].notna()
        & (keep[COL_SKU] != "")
        & (keep[COL_SKU].str.lower() != "nan")
    ]

    if keep.empty:
        return None, (
            "SKU master file has no valid (Line No, Line Name) "
            "rows after cleaning."
        )

    keep["_join_key"] = (
        keep[COL_LINE_NO].astype(str).str.strip().str.lower()
    )

    # If the same Line No appears more than once with different
    # names, keep the FIRST occurrence (the master file is
    # authoritative; later duplicates are silently dropped).
    keep = keep.drop_duplicates(subset="_join_key", keep="first")
    lineno_to_name = (
        keep.set_index("_join_key")[COL_SKU]
    )

    # If the master had a Brand column, build a parallel lookup
    # and attach it as .attrs["brand_lookup"] on the returned
    # Series. Callers that don't care about Brand ignore the
    # attr; callers that need it (load_data_pieces) read it.
    brand_msg = ""
    if COL_BRAND in keep.columns:
        brand_keep = keep[["_join_key", COL_BRAND]].copy()
        brand_keep[COL_BRAND] = brand_keep[COL_BRAND].astype(str).str.strip()
        brand_keep = brand_keep[
            (brand_keep[COL_BRAND] != "")
            & (brand_keep[COL_BRAND].str.lower() != "nan")
        ]
        if not brand_keep.empty:
            brand_lookup = brand_keep.set_index("_join_key")[COL_BRAND]
            # Series.attrs survives a .reindex() so the consumer
            # can pull it back out after the name-mapping step.
            lineno_to_name.attrs["brand_lookup"] = brand_lookup
            brand_msg = f", {len(brand_lookup):,} brands"

    msg = (
        f"Loaded SKU master: {len(lineno_to_name):,} Line No → "
        f"Line Name mappings{brand_msg}."
    )
    return lineno_to_name, msg


# =========================================================
# DIVISOR LOADER  (now keyed on Line No)
# =========================================================

@st.cache_resource(show_spinner=False)
def load_divisor_table(div_bytes, div_name):
    """
    Parse the uploaded Divisor master.

    Schema:
        col with Line No   (any LINE_NO_CANDIDATES name)
        col with Line Name (ignored for matching — present for
                            human readability when reviewing
                            the divisor file; the join uses
                            Line No only)
        one column per month: Jan'25, Feb'25, … (apostrophe
        form — matched case/space/punct insensitively)

    Returns (divisor_lookup, status_message) where
    divisor_lookup is a DataFrame indexed by NORMALISED Line
    No (lower-cased + stripped of whitespace) with one column
    per MRP_MONTH_COLS month. Cells with non-positive divisors
    are treated as missing (no conversion possible → SKU
    excluded for that month).

    On any failure returns (None, error_string).
    """
    if div_bytes is None:
        return None, "No divisor file uploaded."

    bio = io.BytesIO(div_bytes)
    name_lower = (div_name or "").lower()

    try:
        if name_lower.endswith((".xlsx", ".xls", ".xlsm")):
            div_df = pd.read_excel(bio)
        elif name_lower.endswith(".csv"):
            div_df = pd.read_csv(bio)
        elif name_lower.endswith((".parquet", ".pq")):
            div_df = pd.read_parquet(bio)
        else:
            try:
                bio.seek(0)
                div_df = pd.read_excel(bio)
            except Exception:
                bio.seek(0)
                div_df = pd.read_csv(bio)
    except Exception as exc:
        return None, f"Could not read divisor file: {exc}"

    # Locate the Line No column flexibly. If we can't find one
    # via the candidates, fall back to the first column.
    lineno_col = _find_col(div_df.columns, LINE_NO_CANDIDATES)
    if lineno_col is None:
        lineno_col = div_df.columns[0]

    # Match month columns case/space/punct-insensitively against
    # MRP_MONTH_COLS. This is forgiving of files where someone
    # typed "Jan 25" or "jan'25" instead of "Jan'25".
    div_col_keys = {
        _normalise_token(c): c for c in div_df.columns
    }
    month_map = {}  # canonical month → actual column in file
    missing_months = []
    for canon in MRP_MONTH_COLS:
        actual = div_col_keys.get(_normalise_token(canon))
        if actual is not None:
            month_map[canon] = actual
        else:
            missing_months.append(canon)

    if not month_map:
        _expected = (
            f"{MRP_MONTH_COLS[0]} … {MRP_MONTH_COLS[-1]}"
            if MRP_MONTH_COLS
            else "monthly columns"
        )
        return None, (
            f"Divisor file has no month columns matching "
            f"{_expected}. Found columns: "
            f"{list(div_df.columns)[:8]}…"
        )

    # Build the parsed table — rename Line No col to canonical
    # COL_LINE_NO and month cols to canonical MRP_MONTH_COLS names.
    keep = [lineno_col] + list(month_map.values())
    div_df = div_df[keep].copy()
    rename_map = {v: k for k, v in month_map.items()}
    rename_map[lineno_col] = COL_LINE_NO
    div_df = div_df.rename(columns=rename_map)

    for c in month_map.keys():
        div_df[c] = pd.to_numeric(div_df[c], errors="coerce")

    # Normalised join key on Line No.
    div_df["_join_key"] = (
        div_df[COL_LINE_NO].astype(str).str.strip().str.lower()
    )
    # Drop blank Line Nos — they'd collide on the empty-string key.
    div_df = div_df[
        (div_df["_join_key"] != "")
        & (div_df["_join_key"] != "nan")
    ]
    div_avg = (
        div_df.groupby("_join_key", as_index=True)[
            list(month_map.keys())
        ].mean()
    )
    if not div_avg.index.is_unique:
        div_avg = div_avg.groupby(level=0).mean()

    # Replace non-positive cells with NaN — a divisor of 0 or
    # negative makes no physical sense and would explode the
    # pieces computation.
    div_avg = div_avg.where(div_avg > 0)

    msg = f"Loaded divisors for {len(div_avg):,} SKUs."
    if missing_months:
        msg += f" (Missing month columns: {missing_months})"

    return div_avg, msg


@st.cache_data(show_spinner=False)
def load_gm_table(gm_bytes, gm_name):
    """
    Parse the optional Gross-Margin file. Lenient about column
    names — we take the first two columns and treat them as
    (SKU/Line Name, GM value). Returns a tuple (Series, message)
    where the Series is indexed by lower-cased / stripped SKU
    name and the values are float GM indices.

    On any parsing failure returns (None, error_message).
    """
    if gm_bytes is None:
        return None, "No GM file uploaded."

    bio = io.BytesIO(gm_bytes)
    name_lower = (gm_name or "").lower()

    try:
        if name_lower.endswith((".xlsx", ".xls", ".xlsm")):
            gm_df = pd.read_excel(bio)
        elif name_lower.endswith(".csv"):
            gm_df = pd.read_csv(bio)
        else:
            # Best-effort: Excel first, then CSV.
            try:
                bio.seek(0)
                gm_df = pd.read_excel(bio)
            except Exception:
                bio.seek(0)
                gm_df = pd.read_csv(bio)
    except Exception as exc:
        return None, f"Could not read GM file: {exc}"

    if gm_df.shape[1] < 2:
        return None, (
            "GM file needs at least 2 columns "
            "(SKU / Line Name, then Gross Margin)."
        )

    # Take the first two columns regardless of their exact names —
    # the user may label the GM column "GM", "GM Index", "Gross
    # Margin", "Gross Margin %", etc.
    sku_col = gm_df.columns[0]
    gm_col  = gm_df.columns[1]

    keep = gm_df[[sku_col, gm_col]].copy()
    keep.columns = [COL_SKU, "GM_index"]
    keep["GM_index"] = pd.to_numeric(
        keep["GM_index"], errors="coerce"
    )
    keep = keep.dropna(subset=[COL_SKU])
    keep["_join_key"] = (
        keep[COL_SKU].astype(str).str.strip().str.lower()
    )
    # Average if multiple rows for the same SKU
    gm_avg = (
        keep.groupby("_join_key")["GM_index"]
        .mean()
    )
    # Drop SKUs whose GM is missing — those are excluded from
    # the GM-based ranking per the user's spec.
    gm_avg = gm_avg.dropna()

    if gm_avg.empty:
        return None, (
            "GM file parsed but all gross-margin values are "
            "blank / non-numeric."
        )

    msg = (
        f"Loaded gross margins for {len(gm_avg):,} SKUs "
        f"(column used: '{gm_col}')."
    )
    return gm_avg, msg


def load_sku_size_table(sz_bytes, sz_name):
    """
    Parse the optional SKU-Size file used by the VC Planogram
    Builder tab. Schema:
        col 1 = SKU / Line Name,
        col 2 = Size category ("Large" / "Medium" / "Small"),
        col 3 = Brand   (OPTIONAL — used as a Brand source for
                         the whole dashboard when the base file
                         carries no Brand column).
    We're lenient about exact column names and just take the
    columns by position. Returns (Series, message) where the
    Series is indexed by lower-cased / stripped SKU name and the
    values are the normalised size strings ("Large" / "Medium" /
    "Small"). On failure returns (None, error_message).

    If a 3rd column is present, a parallel SKU-name → Brand
    lookup is attached to the returned Series as
    .attrs["brand_lookup"] (indexed by lower-cased / stripped
    SKU name). Callers that don't care about Brand ignore the
    attr; load_data_pieces reads it to fill a missing Brand
    column. The attr survives .reindex() so it can be pulled
    back out downstream.

    Size normalisation: case-insensitive match to one of the
    three buckets — "large", "big", "big/large" → "Large";
    "medium", "med", "mid" → "Medium"; "small", "countline" →
    "Small". Anything else is dropped with a count in the
    status message.
    """
    if sz_bytes is None:
        return None, "No SKU-Size file uploaded."

    bio = io.BytesIO(sz_bytes)
    name_lower = (sz_name or "").lower()

    try:
        if name_lower.endswith((".xlsx", ".xls", ".xlsm")):
            sz_df = pd.read_excel(bio)
        elif name_lower.endswith(".csv"):
            sz_df = pd.read_csv(bio)
        else:
            try:
                bio.seek(0)
                sz_df = pd.read_excel(bio)
            except Exception:
                bio.seek(0)
                sz_df = pd.read_csv(bio)
    except Exception as exc:
        return None, f"Could not read SKU-Size file: {exc}"

    if sz_df.shape[1] < 2:
        return None, (
            "SKU-Size file needs at least 2 columns "
            "(SKU / Line Name, then Size)."
        )

    sku_col = sz_df.columns[0]
    sz_col = sz_df.columns[1]
    # Optional 3rd column = Brand. Taken by position so the user
    # can label it anything ("Brand", "BRAND", "Brand Name"…).
    brand_col = sz_df.columns[2] if sz_df.shape[1] >= 3 else None

    _take_cols = [sku_col, sz_col]
    if brand_col is not None:
        _take_cols.append(brand_col)
    keep = sz_df[_take_cols].copy()
    keep.columns = (
        [COL_SKU, "Size_raw", COL_BRAND]
        if brand_col is not None
        else [COL_SKU, "Size_raw"]
    )
    keep = keep.dropna(subset=[COL_SKU, "Size_raw"])
    keep["_join_key"] = (
        keep[COL_SKU].astype(str).str.strip().str.lower()
    )

    # Build the SKU-name → Brand side-channel BEFORE size
    # normalisation drops rows with unrecognised sizes — a SKU
    # with a valid brand but an odd size string should still
    # contribute its brand. Keyed on lower-cased / stripped SKU
    # name to match how load_data_pieces joins brands by name.
    brand_lookup = None
    n_brands = 0
    if brand_col is not None and COL_BRAND in keep.columns:
        _bk = keep[["_join_key", COL_BRAND]].copy()
        _bk[COL_BRAND] = _bk[COL_BRAND].astype(str).str.strip()
        _bk = _bk[
            (_bk[COL_BRAND] != "")
            & (_bk[COL_BRAND].str.lower() != "nan")
        ]
        if not _bk.empty:
            brand_lookup = (
                _bk.drop_duplicates("_join_key", keep="first")
                .set_index("_join_key")[COL_BRAND]
            )
            n_brands = len(brand_lookup)

    def _norm_size(s):
        s = str(s).strip().lower()
        if s in ("large", "big", "big/large", "l"):
            return "Large"
        if s in ("medium", "med", "mid", "m"):
            return "Medium"
        if s in ("small", "countline", "count line", "s"):
            return "Small"
        return None

    keep["Size"] = keep["Size_raw"].apply(_norm_size)
    n_unknown = int(keep["Size"].isna().sum())
    keep = keep.dropna(subset=["Size"])

    if keep.empty:
        # No recognised sizes. If the file still carried a usable
        # Brand column, return an EMPTY size-series that carries
        # the brand side-channel so load_data_pieces can still use
        # the brands — the Planogram tab will just stay locked.
        if brand_lookup is not None:
            empty_sz = pd.Series(dtype="object", name="Size")
            empty_sz.attrs["brand_lookup"] = brand_lookup
            return empty_sz, (
                "SKU-Size file had no recognised sizes "
                "(Large / Medium / Small), but "
                f"{n_brands:,} brand(s) were read from column "
                f"'{brand_col}' and will be used for the Brand "
                "breakdown."
            )
        return None, (
            "SKU-Size file parsed but no rows had a recognised "
            "size (Large / Medium / Small)."
        )

    # If a SKU appears more than once, keep the first non-null
    # mapping.
    sz_series = (
        keep.drop_duplicates("_join_key")
        .set_index("_join_key")["Size"]
    )

    # Attach the brand side-channel (if any). Series.attrs
    # survives .reindex() so load_data_pieces can pull it back out.
    if brand_lookup is not None:
        sz_series.attrs["brand_lookup"] = brand_lookup

    msg = (
        f"Loaded sizes for {len(sz_series):,} SKUs "
        f"(column used: '{sz_col}')"
    )
    if n_brands:
        msg += f", {n_brands:,} brands (column used: '{brand_col}')"
    if n_unknown:
        msg += f", {n_unknown} row(s) skipped (unknown size)"
    msg += "."
    return sz_series, msg


@st.cache_data(show_spinner=False)
def load_vc_cat_table(vc_bytes, vc_name):
    """
    Parse the optional VC-Category mapping file. Schema (by
    position, lenient about exact column names):
        col 1 = VC Model name  (joins to the base file's VC_MODEL)
        col 2 = VC Category    (the band, e.g. "16-20L", "45-70L")

    Returns (Series, message) where the Series is indexed by the
    NORMALISED VC_MODEL key (via _normalise_token, so casing /
    whitespace / punctuation differences don't matter) and the
    values are the cleaned VC_CAT strings. On any failure returns
    (None, error_message).

    The VC_CAT values are passed through the same legacy→canonical
    relabelling the source loader applies (<20L → 16-20L, etc.) so
    an uploaded mapping using the old "<X L" form still lines up
    with the in-app ordering and tier-count maps.
    """
    if vc_bytes is None:
        return None, "No VC-Category file uploaded."

    bio = io.BytesIO(vc_bytes)
    name_lower = (vc_name or "").lower()

    try:
        if name_lower.endswith((".xlsx", ".xls", ".xlsm")):
            vc_df = pd.read_excel(bio)
        elif name_lower.endswith(".csv"):
            vc_df = pd.read_csv(bio)
        elif name_lower.endswith((".parquet", ".pq")):
            vc_df = pd.read_parquet(bio)
        else:
            try:
                bio.seek(0)
                vc_df = pd.read_excel(bio)
            except Exception:
                bio.seek(0)
                vc_df = pd.read_csv(bio)
    except Exception as exc:
        return None, f"Could not read VC-Category file: {exc}"

    if vc_df.shape[1] < 2:
        return None, (
            "VC-Category file needs at least 2 columns "
            "(VC Model name, then VC Category)."
        )

    # Take the first two columns by position — the user may label
    # them "VC_MODEL"/"VC Model"/"Model" and "VC_CAT"/"VC Category"
    # /"Band" etc.; we don't depend on exact names.
    model_col = vc_df.columns[0]
    cat_col = vc_df.columns[1]

    keep = vc_df[[model_col, cat_col]].copy()
    keep.columns = [COL_VC, COL_VC_CATEGORY]

    keep[COL_VC] = keep[COL_VC].astype(str).str.strip()
    keep[COL_VC_CATEGORY] = keep[COL_VC_CATEGORY].astype(str).str.strip()

    # Drop rows with a blank model or blank category — they can't
    # act as a mapping.
    keep = keep[
        (keep[COL_VC] != "")
        & (keep[COL_VC].str.lower() != "nan")
        & (keep[COL_VC_CATEGORY] != "")
        & (keep[COL_VC_CATEGORY].str.lower() != "nan")
    ]

    if keep.empty:
        return None, (
            "VC-Category file has no valid (VC Model, VC Category) "
            "rows after cleaning."
        )

    # Relabel legacy VC_CAT forms to the canonical range form so an
    # uploaded mapping lines up with VC_CAT_ORDER and the tier-count
    # map (same mapping the base loader applies in load_data()).
    keep[COL_VC_CATEGORY] = keep[COL_VC_CATEGORY].replace({
        "<33L":  "30-35L",
        "<20L":  "16-20L",
        "<35L":  "30-35L",
        "<70L":  "45-70L",
        "<220L": "110-220L",
        "<340L": "280-340L",
        "<465L": "380-465L",
        ">500L": "700-1000L",
    })

    # Normalised join key on VC_MODEL so casing / whitespace /
    # punctuation drift between the mapping file and the source
    # file's VC_MODEL doesn't break the join.
    keep["_join_key"] = keep[COL_VC].apply(_normalise_token)
    keep = keep[keep["_join_key"] != ""]

    if keep.empty:
        return None, (
            "VC-Category file rows had empty VC Model keys after "
            "normalisation."
        )

    # If a VC Model appears more than once, keep the FIRST mapping
    # (file is authoritative; later duplicates dropped).
    keep = keep.drop_duplicates(subset="_join_key", keep="first")
    vc_lookup = keep.set_index("_join_key")[COL_VC_CATEGORY]

    msg = (
        f"Loaded VC categories for {len(vc_lookup):,} VC models "
        f"(columns used: '{model_col}' → '{cat_col}')."
    )
    return vc_lookup, msg


@st.cache_data(show_spinner=False)
def compute_outlet_value_totals(
    file_bytes_token, file_name,
    div_bytes_token, div_name,
    master_bytes_token, master_name,
    mrp_bytes_token, mrp_name,
    already_pieces=False,
):
    """
    Compute Σ (monthly_pieces × monthly_MRP) per outlet across
    all SKUs, for the *unfiltered* base file (post sales→pieces
    conversion). Returned as a Series indexed by OUTLETUID.

    Cached on (base file bytes, divisor file bytes, SKU master
    bytes, MRP file bytes). Runs exactly once per unique
    combination — not per rerun and not per filter change.
    Filters are applied downstream just by sub-setting this
    series via outlet ids.
    """
    base_df, _ = load_data_pieces(
        file_bytes_token, file_name,
        div_bytes_token, div_name,
        master_bytes_token, master_name,
        None, None,
        None, None,
        already_pieces,
        mrp_bytes_token, mrp_name,
    )

    mrp_lookup, _msg = load_mrp_table(mrp_bytes_token, mrp_name)
    if mrp_lookup is None:
        return None

    # Vectorised computation: build a per-row sale value as
    #     Σ_m (volume_row[m] * mrp_for_line_name[m])
    # by joining on lower-cased Line Name.
    sku_keys = (
        base_df[COL_SKU].astype(str).str.strip().str.lower()
    )

    # Reindex MRP table to align with base_df row order; missing
    # SKUs get NaN MRPs which become 0 contribution.
    mrp_aligned = mrp_lookup.reindex(sku_keys.values).fillna(0)
    mrp_aligned.index = base_df.index

    # Pair each volume column with its MRP column and sum
    # contributions.
    row_value = pd.Series(0.0, index=base_df.index)
    for vol_col, mrp_col in zip(MONTH_COLS, MRP_MONTH_COLS):
        if vol_col not in base_df.columns:
            continue
        if mrp_col not in mrp_aligned.columns:
            continue
        row_value = row_value + (
            base_df[vol_col].astype(float).values
            * mrp_aligned[mrp_col].astype(float).values
        )

    # Roll up to outlet level.
    outlet_totals = (
        pd.DataFrame({
            COL_OUTLET: base_df[COL_OUTLET].values,
            "value": row_value.values
        })
        .groupby(COL_OUTLET)["value"]
        .sum()
    )

    return outlet_totals


# =========================================================
# SALES → PIECES CONVERSION  (cached, runs ONCE per session)
# =========================================================
# The source file's monthly columns hold SALES (₹). The user
# uploads a divisor file (one number per SKU per month). This
# wrapper:
#
#   1. Loads the raw source file (cached separately).
#   2. Loads the divisor lookup (cached separately).
#   3. For every monthly column present, divides sales by the
#      corresponding divisor — converting ₹ to pieces. This
#      happens IN-PLACE on a single DataFrame so all downstream
#      code that already reads MONTH_COLS now reads pieces
#      without any further changes.
#   4. Recomputes L3M and L15M aggregate columns from the
#      converted pieces so any legacy code that still references
#      them stays correct.
#   5. Drops every row whose SKU is missing from the divisor
#      file (so KPIs aren't contaminated by un-convertible
#      sales).
#
# Cached on (source bytes, divisor bytes) → runs once. Toggling
# any sidebar filter, the month-picker, smoothing, etc., does
# NOT re-run this conversion.

@st.cache_data(show_spinner=False)
def load_data_pieces(
    file_bytes, file_name,
    div_bytes, div_name,
    master_bytes, master_name,
    sku_size_bytes=None, sku_size_name=None,
    vc_cat_bytes=None, vc_cat_name=None,
    already_pieces=False,
    mrp_bytes_for_names=None, mrp_name_for_names=None,
):
    """
    Returns (pieces_df, report_dict).

    Pipeline:
        1. Read raw source file (cached separately).
        2. Locate the Line No column on the base (any variant
           name in LINE_NO_CANDIDATES) and rename to canonical
           COL_LINE_NO.
        3. Attach Line Name to the base by joining the Line No
           → Line Name lookup derived from the MRP (Final MRP
           Tracker) file. The MRP file's LINKNO + Line Name
           pairing is authoritative and replaces the legacy
           separate SKU master upload. If the MRP file isn't
           provided, the legacy SKU master path is used as a
           fallback. Required only if the base file doesn't
           already carry Line Name.
        4. Drop every row where Line Name is still missing or
           blank — per user spec: "if line name is missing
           then drop the sku, don't count it in analysis".
        5. (Sales mode only) Join the divisor lookup on Line
           No (lower-cased) and do per-cell sales→pieces
           conversion. In **Pieces mode** (`already_pieces=True`)
           this step is skipped entirely — the monthly cells
           are already pieces and are used as-is.
        6. (Sales mode only) Drop SKUs that have no divisor
           entry at all. In Pieces mode no SKUs are dropped
           on this basis.
        7. Recompute L3M / L15M aggregates from the monthly
           pieces columns.

    report_dict = {
        "ok": bool,
        "msg": str,
        "dropped_no_name": [Line No, …],  # base rows w/o name
        "dropped_skus": [Line Name, …],   # SKUs missing from divisor
        "months_used": [str, …],
    }

    If the divisor file is missing or unreadable, report_dict
    ["ok"] is False — the main flow halts the app in that case.
    """
    base_df = load_data(file_bytes, file_name).copy()

    # Defensive re-population of MONTH_COLS / MRP_MONTH_COLS. On a
    # cache HIT of load_data() the function body doesn't run, so
    # the globals it would have published may still be empty if
    # this is the first time we're hitting this DataFrame in the
    # session. _detect_month_cols is pure + cheap; rerunning it
    # here guarantees downstream code (load_divisor_table,
    # vol_to_mrp zip, etc.) sees the correct lists.
    _v, _m = _detect_month_cols(base_df.columns)
    MONTH_COLS.clear(); MONTH_COLS.extend(_v)
    MRP_MONTH_COLS.clear(); MRP_MONTH_COLS.extend(_m)

    # ---------- Step 2: locate / canonicalise Line No ----------
    base_lineno_col = _find_col(base_df.columns, LINE_NO_CANDIDATES)
    if base_lineno_col is None:
        return base_df, {
            "ok": False,
            "msg": (
                "Source file has no Line No column. Tried: "
                f"{LINE_NO_CANDIDATES[:5]}…"
            ),
            "dropped_no_name": [],
            "dropped_skus": [],
            "months_used": [],
        }
    if base_lineno_col != COL_LINE_NO:
        base_df = base_df.rename(columns={base_lineno_col: COL_LINE_NO})

    # Strip whitespace on Line No so the join key is stable.
    base_df[COL_LINE_NO] = (
        base_df[COL_LINE_NO].astype(str).str.strip()
    )

    # ---------- Step 3: attach Line Name from MRP file ----------
    # Primary source is the MRP (Final MRP Tracker) file, which
    # carries LINKNO + Line Name. The legacy separate SKU master
    # file is still accepted as a fallback for backward compat.
    has_name_in_base = COL_SKU in base_df.columns

    # Try MRP-derived lookup first.
    mrp_name_lookup, mrp_name_status = (None, "MRP file not provided.")
    if mrp_bytes_for_names is not None:
        mrp_name_lookup, mrp_name_status = load_lineno_to_name_from_mrp(
            mrp_bytes_for_names, mrp_name_for_names
        )

    # Fall back to the legacy SKU master path if the MRP file
    # didn't yield a usable lookup.
    master_lookup, master_status = (None, "")
    if mrp_name_lookup is None and master_bytes is not None:
        master_lookup, master_status = load_sku_master_table(
            master_bytes, master_name
        )

    name_lookup = mrp_name_lookup if mrp_name_lookup is not None else master_lookup

    if not has_name_in_base and name_lookup is None:
        # Base has no Line Name AND no source for the mapping → fatal.
        return base_df, {
            "ok": False,
            "msg": (
                "Source file has only Line No (no Line Name) and "
                "no MRP file with a Line Name column was uploaded. "
                "Upload the Final MRP Tracker file in the sidebar "
                "so the dashboard can attach human-readable SKU "
                f"names. (MRP-derived lookup said: {mrp_name_status})"
            ),
            "dropped_no_name": [],
            "dropped_skus": [],
            "months_used": [],
        }

    if name_lookup is not None:
        # MRP-derived (or legacy master) lookup is authoritative —
        # overwrites any Line Name already in the base.
        lineno_keys = (
            base_df[COL_LINE_NO].astype(str).str.strip().str.lower()
        )
        mapped_names = name_lookup.reindex(lineno_keys.values)
        mapped_names.index = base_df.index
        if has_name_in_base:
            # Fill mapped names; where lookup has nothing, fall
            # back to whatever the base already had.
            base_df[COL_SKU] = mapped_names.where(
                mapped_names.notna(),
                base_df[COL_SKU],
            )
        else:
            base_df[COL_SKU] = mapped_names

    # ---------- Step 4: drop rows with missing Line Name ----------
    name_series = base_df[COL_SKU].astype(str).str.strip()
    name_missing = (
        base_df[COL_SKU].isna()
        | (name_series == "")
        | (name_series.str.lower() == "nan")
    )
    dropped_no_name_linenos = sorted(
        base_df.loc[name_missing, COL_LINE_NO]
        .astype(str)
        .unique()
        .tolist()
    )
    base_df = base_df.loc[~name_missing].copy()

    if base_df.empty:
        return base_df, {
            "ok": False,
            "msg": (
                "Every base row had a missing Line Name after "
                "applying the SKU master. Nothing left to analyse."
            ),
            "dropped_no_name": dropped_no_name_linenos,
            "dropped_skus": [],
            "months_used": [],
        }

    # ---------- Step 4b: ensure Brand column exists ----------
    # Brand is referenced everywhere in the dashboard (groupbys,
    # filters, the planogram, the priority lister) but is OPTIONAL
    # in the source file. Resolution order (each later step only
    # fills rows still missing a Brand):
    #   (a) Brand already present on the base file → use it.
    #   (b) Brand provided on the SKU master file (side-channel
    #       via master_lookup.attrs["brand_lookup"], keyed on
    #       Line No) → join it in.
    #   (c) Brand provided on the SKU-Size file's 3rd column
    #       (side-channel via sku_size_lookup.attrs["brand_lookup"],
    #       keyed on Line Name) → join it in.
    #   (d) Derive Brand from the first token of Line Name
    #       (e.g. "Cadbury Dairy Milk 13g" → "Cadbury",
    #        "Silk Roast Almond 58g" → "Silk").
    # In case (d) we record `brand_derived = True` in the report
    # so the main flow can surface a sidebar notice.
    brand_derived = False
    brand_source = "base file"
    has_brand_in_base = COL_BRAND in base_df.columns
    if has_brand_in_base:
        # Tidy whatever is there: cast to str + strip, replace
        # empty/nan with NaN so the downstream fallbacks can kick
        # in row-by-row.
        bser = base_df[COL_BRAND].astype(str).str.strip()
        base_df[COL_BRAND] = bser.where(
            (bser != "") & (bser.str.lower() != "nan"),
            other=np.nan,
        )

    master_brand_lookup = None
    if master_lookup is not None:
        master_brand_lookup = master_lookup.attrs.get("brand_lookup")

    if master_brand_lookup is not None:
        lineno_keys = (
            base_df[COL_LINE_NO].astype(str).str.strip().str.lower()
        )
        mapped_brands = master_brand_lookup.reindex(lineno_keys.values)
        mapped_brands.index = base_df.index
        if has_brand_in_base:
            # Master fills only the gaps; existing base values win.
            base_df[COL_BRAND] = base_df[COL_BRAND].where(
                base_df[COL_BRAND].notna(),
                mapped_brands,
            )
        else:
            base_df[COL_BRAND] = mapped_brands
            has_brand_in_base = True
            brand_source = "SKU master"

    # ---- (c) SKU-Size file 3rd column → Brand (keyed on name) ----
    # Parse the optional SKU-Size file (if its bytes were threaded
    # in) and pull its brand side-channel. This is keyed on the
    # lower-cased / stripped Line Name (the SKU-Size file maps SKU
    # name → size/brand), unlike the master which is keyed on
    # Line No. It only fills rows that are still missing a Brand
    # after (a)/(b), so the base file and SKU master always win.
    size_brand_lookup = None
    if sku_size_bytes is not None:
        _sz_series_for_brand, _ = load_sku_size_table(
            sku_size_bytes, sku_size_name
        )
        if _sz_series_for_brand is not None:
            size_brand_lookup = (
                _sz_series_for_brand.attrs.get("brand_lookup")
            )

    if size_brand_lookup is not None:
        name_keys = (
            base_df[COL_SKU].astype(str).str.strip().str.lower()
        )
        mapped_size_brands = size_brand_lookup.reindex(name_keys.values)
        mapped_size_brands.index = base_df.index
        if COL_BRAND in base_df.columns:
            # Fill only the gaps; whatever (a)/(b) already set wins.
            _had_any_before = base_df[COL_BRAND].notna().any()
            base_df[COL_BRAND] = base_df[COL_BRAND].where(
                base_df[COL_BRAND].notna(),
                mapped_size_brands,
            )
            if not _had_any_before:
                has_brand_in_base = True
                brand_source = "SKU-Size file"
        else:
            base_df[COL_BRAND] = mapped_size_brands
            has_brand_in_base = True
            brand_source = "SKU-Size file"

    # Final fallback: derive Brand from Line Name (first token).
    # Catches any remaining NaN rows after the (a) / (b) / (c)
    # passes.
    if COL_BRAND not in base_df.columns:
        # Use object dtype, not float — otherwise the column is
        # float64-NaN and assigning strings into it fails on
        # newer pandas with LossySetitemError.
        base_df[COL_BRAND] = pd.Series(
            [np.nan] * len(base_df),
            index=base_df.index,
            dtype="object",
        )
    brand_missing_mask = base_df[COL_BRAND].isna()
    if brand_missing_mask.any():
        derived = (
            base_df.loc[brand_missing_mask, COL_SKU]
            .astype(str)
            .str.strip()
            .str.split(n=1)
            .str[0]
            .replace({"": "Unknown", "nan": "Unknown"})
            .fillna("Unknown")
        )
        base_df.loc[brand_missing_mask, COL_BRAND] = derived
        brand_derived = True
        if brand_source == "base file" and not has_brand_in_base:
            brand_source = "derived from Line Name"

    # ---------- Step 4c: apply VC_CAT mapping (optional) ----------
    # If a VC-Category mapping file was provided, set each row's
    # VC_CAT from it, joining on the (normalised) VC_MODEL. The
    # mapping is AUTHORITATIVE: where a VC Model is present in the
    # file, its VC_CAT overrides whatever the source carried; where
    # a VC Model is absent from the file, the row keeps its existing
    # VC_CAT (or stays blank if the source had none). Requires a
    # VC_MODEL column on the base file to join against.
    vc_cat_applied = 0
    vc_cat_source = None
    if vc_cat_bytes is not None:
        vc_lookup, _vc_status = load_vc_cat_table(
            vc_cat_bytes, vc_cat_name
        )
        if vc_lookup is not None and COL_VC in base_df.columns:
            model_keys = base_df[COL_VC].apply(_normalise_token)
            mapped_cats = vc_lookup.reindex(model_keys.values)
            mapped_cats.index = base_df.index
            matched_mask = mapped_cats.notna()
            vc_cat_applied = int(matched_mask.sum())
            if COL_VC_CATEGORY in base_df.columns:
                # Authoritative override where the file has a value;
                # keep the existing VC_CAT elsewhere.
                base_df[COL_VC_CATEGORY] = base_df[COL_VC_CATEGORY].astype(
                    "object"
                )
                base_df.loc[matched_mask, COL_VC_CATEGORY] = (
                    mapped_cats[matched_mask].values
                )
            else:
                # No VC_CAT column existed → create one from the map.
                base_df[COL_VC_CATEGORY] = mapped_cats
            vc_cat_source = "VC-Category file"

    # ---------- Step 5: load divisor + per-cell conversion ----------
    # PIECES MODE: skip the divisor join entirely — the monthly
    # cells on the base file are already pieces. We still need to
    # ensure the monthly columns are numeric (load_data already
    # coerces them, but a defensive re-coerce is cheap) and then
    # jump straight to the L3M / L15M recomputation in Step 7.
    if already_pieces:
        months_present = [
            m for m in MONTH_COLS if m in base_df.columns
        ]
        for vol_col in months_present:
            base_df[vol_col] = pd.to_numeric(
                base_df[vol_col], errors="coerce"
            ).fillna(0.0).clip(lower=0.0)

        if months_present:
            last3 = months_present[-3:]
            base_df["L3M"] = base_df[last3].sum(axis=1)
            base_df["L15M"] = base_df[months_present].sum(axis=1)
        else:
            base_df["L3M"] = 0.0
            base_df["L15M"] = 0.0

        n_rows = len(base_df)
        return base_df, {
            "ok": True,
            "msg": (
                f"Loaded {n_rows:,} rows in Pieces mode "
                f"(no divisor conversion applied; "
                f"{len(dropped_no_name_linenos):,} dropped earlier "
                f"for missing Line Name)."
            ),
            "dropped_no_name": dropped_no_name_linenos,
            "dropped_skus": [],
            "months_used": months_present,
            "brand_source": brand_source,
            "brand_derived": brand_derived,
            "vc_cat_applied": vc_cat_applied,
            "vc_cat_source": vc_cat_source,
        }

    # SALES MODE: load the divisor lookup and convert sales →
    # pieces cell-by-cell.
    div_lookup, div_status = load_divisor_table(div_bytes, div_name)
    if div_lookup is None:
        return base_df, {
            "ok": False,
            "msg": div_status,
            "dropped_no_name": dropped_no_name_linenos,
            "dropped_skus": [],
            "months_used": [],
        }

    months_present = [m for m in MONTH_COLS if m in base_df.columns]
    vol_to_mrp = dict(zip(MONTH_COLS, MRP_MONTH_COLS))

    # Build per-row divisor matrix aligned to base_df row order.
    # Match on Line No (the divisor lookup is keyed on it).
    lineno_keys = (
        base_df[COL_LINE_NO].astype(str).str.strip().str.lower()
    )
    div_aligned = div_lookup.reindex(lineno_keys.values)
    div_aligned.index = base_df.index

    # SKUs (by Line No) missing from divisor entirely → dropped.
    has_any_divisor = div_aligned.notna().any(axis=1)
    n_before = len(base_df)
    dropped_mask = ~has_any_divisor
    dropped_skus = sorted(
        base_df.loc[dropped_mask, COL_SKU]
        .astype(str)
        .unique()
        .tolist()
    )

    base_df = base_df.loc[has_any_divisor].copy()
    div_aligned = div_aligned.loc[has_any_divisor]

    # Per-cell sales→pieces.
    for vol_col in months_present:
        mrp_col = vol_to_mrp.get(vol_col)
        if mrp_col is None or mrp_col not in div_aligned.columns:
            base_df[vol_col] = 0.0
            continue
        sales = pd.to_numeric(
            base_df[vol_col], errors="coerce"
        ).fillna(0.0)
        divisor = div_aligned[mrp_col].astype(float)
        pieces = sales.values / divisor.values
        pieces = np.where(
            np.isfinite(pieces) & (pieces >= 0),
            pieces, 0.0
        )
        base_df[vol_col] = pieces

    # ---------- Step 7: recompute L3M / L15M from pieces ----------
    if months_present:
        last3 = months_present[-3:]
        base_df["L3M"] = base_df[last3].sum(axis=1)
        base_df["L15M"] = base_df[months_present].sum(axis=1)
    else:
        base_df["L3M"] = 0.0
        base_df["L15M"] = 0.0

    n_after = len(base_df)
    msg = (
        f"Converted sales → pieces for {n_after:,} rows "
        f"({n_before - n_after:,} dropped: missing from divisor; "
        f"{len(dropped_no_name_linenos):,} additionally dropped "
        f"earlier for missing Line Name)."
    )

    return base_df, {
        "ok": True,
        "msg": msg,
        "dropped_no_name": dropped_no_name_linenos,
        "dropped_skus": dropped_skus,
        "months_used": months_present,
        "brand_source": brand_source,
        "brand_derived": brand_derived,
        "vc_cat_applied": vc_cat_applied,
        "vc_cat_source": vc_cat_source,
    }


# =========================================================
# FILE UPLOADER  (replaces hardcoded DATA_PATH)
# =========================================================

st.sidebar.title("Data Source")

# Data mode selector — chooses how to interpret the monthly
# columns on the base file:
#   - "Sales (₹) — convert to pieces using divisor file":
#       The base file's monthly columns hold ₹ sales values.
#       A divisor file is REQUIRED; each (SKU, month) cell is
#       divided by the corresponding per-SKU per-month divisor
#       to produce pieces. This is the original V4 behaviour.
#   - "Pieces — already in pieces":
#       The base file's monthly columns already hold pieces.
#       NO divisor file is needed. The conversion step is
#       skipped entirely and the monthly cells are used as-is.
# The choice is persisted in session state via the radio's key
# and threaded into load_data_pieces() through `mode_pieces`.
data_mode = st.sidebar.radio(
    "Base file content",
    options=[
        "Sales (₹) — convert to pieces using divisor file",
        "Pieces — already in pieces",
    ],
    index=0,
    key="sb_data_mode",
    help=(
        "Pick how the monthly columns in your base file should "
        "be interpreted. In 'Sales' mode the dashboard divides "
        "each (SKU, month) cell by the divisor file to derive "
        "pieces. In 'Pieces' mode the cells are already pieces "
        "— no divisor file is needed and no division happens."
    ),
)
_already_pieces = data_mode.startswith("Pieces")

# Base file uploader — a single file. Earlier versions supported
# up to three files of the same schema that were concatenated;
# that multi-file path has been removed per user request — only
# one base file is accepted now. The file is threaded through
# the existing load_data_pieces pipeline. In 'Sales' mode its
# monthly cells are divided by the divisor file to derive
# pieces; in 'Pieces' mode the cells are used as-is.
uploaded_file = st.sidebar.file_uploader(
    "Upload source file (required)",
    type=["parquet", "pq", "csv"],
    key="sb_uploaded_file",
    help=(
        "Pick the base file from your PC. Expected columns "
        "include OUTLETUID, Brand, Line No, and the monthly "
        "columns (Jan25_V … Mar26_V). The base file may use "
        "Line No only; Line Name is attached via the SKU "
        "master file (below). In 'Sales' mode the monthly "
        "cells are converted to pieces at load-time using the "
        "divisor file; in 'Pieces' mode they are used as-is. "
        "Other dimension columns (ASM_AREA, CHANNEL, RD_CODE, "
        "RE, Region, VC_MODEL, etc.) are auto-detected — the "
        "dashboard works with whichever ones are present and "
        "ignores missing ones."
    )
)

# SKU master upload has been removed — the Line No → Line Name
# mapping is now derived directly from the Final MRP Tracker file
# (which already carries LINKNO + Line Name + monthly MRPs). We
# keep `uploaded_sku_master_file = None` as a stub so the rest of
# the dashboard's existing references continue to compile, but no
# uploader is shown in the sidebar.
uploaded_sku_master_file = None

# REQUIRED divisor file — used to convert each (SKU, month)
# sales cell into pieces. Loaded once per session via
# @st.cache_resource keyed on its raw bytes. New schema (v4.1):
#   col with Line No  (any of LINE_NO_CANDIDATES — match is
#                      case/space/punct-insensitive)
#   col with Line Name (kept for human readability; ignored
#                       for matching — Line No is the join key)
#   one column per month: Jan'25, Feb'25, …
uploaded_divisor_file = st.sidebar.file_uploader(
    "Upload Divisor file (sales → pieces)",
    type=["xlsx", "xls", "xlsm", "csv", "parquet", "pq"],
    key="sb_uploaded_divisor_file",
    help=(
        "Required ONLY in 'Sales' mode (ignored in 'Pieces' "
        "mode). Columns: Line No, Line Name (for reference; "
        "the join uses Line No), then one column per month "
        "named Jan'25, Feb'25, … Mar'26 (with the apostrophe). "
        "For each (SKU, month) cell the dashboard divides "
        "sales by this divisor to get pieces — the conversion "
        "runs ONCE at load-time and is cached, so changing "
        "filters never re-runs it. SKUs (by Line No) missing "
        "from this file are excluded from analysis."
    ),
    disabled=_already_pieces,
)

# Optional MRP master — only needed if you want value-based
# smoothing instead of pieces-based. Uploaded once per session
# and cached; flipping the smoothing-basis radio is instant
# afterwards.
uploaded_mrp_file = st.sidebar.file_uploader(
    "Upload MRP file (optional)",
    type=["xlsx", "xls", "xlsm", "csv", "parquet", "pq"],
    key="sb_uploaded_mrp_file",
    help=(
        "Optional. Expected columns: LINKNO, Line Name, then "
        "one column per month named Jan'25, Feb'25, … Mar'26 "
        "(with the apostrophe). When provided, you can switch "
        "the smoothing basis to 'Sale Value' to rank outlets "
        "by ₹ total instead of pieces. Cached for the session "
        "— uploaded once, not re-parsed on every rerun."
    )
)

# Optional Gross Margin master — fully optional. Unlocks the
# "SKU Priority Lister" top-level tab when provided. Expected
# schema: two columns — the first must be a SKU / Line Name
# column, the second must be the gross-margin value (typically
# a percentage or an index). We're lenient about exact column
# names and just take the first two columns.
uploaded_gm_file = st.sidebar.file_uploader(
    "Upload Gross Margin file (optional)",
    type=["xlsx", "xls", "xlsm", "csv"],
    key="sb_uploaded_gm_file",
    help=(
        "Optional. Two-column file: column 1 = Line Name (SKU "
        "name), column 2 = Gross Margin (GM index / %). "
        "Uploading this enables the 'SKU Priority Lister' tab "
        "where SKUs can be ranked by Throughput × MRP × GM. "
        "SKUs missing from this file are simply excluded from "
        "the GM-based ranking; everything else in the "
        "dashboard works exactly as before."
    )
)

# Optional SKU-Size file (Large / Medium / Small per SKU). Used
# by the VC Planogram Builder tab to decide which SKUs may be
# placed in which shelf-slot type, and (via an optional 3rd
# column) as a Brand source for the whole dashboard. Columns:
# col 1 = SKU / Line Name, col 2 = Size ("Large" / "Medium" /
# "Small"), col 3 = Brand (optional).
uploaded_sku_size_file = st.sidebar.file_uploader(
    "Upload SKU Size file (optional)",
    type=["xlsx", "xls", "xlsm", "csv"],
    key="sb_uploaded_sku_size_file",
    help=(
        "Optional. Column 1 = Line Name (SKU name), column 2 = "
        "Size category ('Large', 'Medium', or 'Small'), and an "
        "OPTIONAL column 3 = Brand. Uploading this enables the "
        "'VC Planogram Builder' tab where you can drop SKUs into "
        "a virtual cooler skeleton respecting size-fit rules. If "
        "a 3rd Brand column is present it is used to fill the "
        "Brand breakdown when the source file carries no Brand "
        "column. SKUs missing from this file cannot be placed in "
        "the builder; everything else works as before."
    )
)

# Optional VC-Category mapping file. Two columns: col 1 = VC
# Model name (joins to the base file's VC_MODEL), col 2 = VC
# Category (the band). When uploaded, it sets / overrides each
# row's VC_CAT from this mapping — useful when the source file
# has no VC_CAT column, or carries an out-of-date one.
uploaded_vc_cat_file = st.sidebar.file_uploader(
    "Upload VC Category file (optional)",
    type=["xlsx", "xls", "xlsm", "csv", "parquet", "pq"],
    key="sb_uploaded_vc_cat_file",
    help=(
        "Optional. Two-column file: column 1 = VC Model name "
        "(matched to the source file's VC_MODEL), column 2 = VC "
        "Category band (e.g. '16-20L', '45-70L'). When provided, "
        "the dashboard sets each row's VC_CAT from this mapping "
        "(authoritative — it overrides any VC_CAT already on the "
        "source file). VC Models not in this file keep whatever "
        "VC_CAT the source had, or blank if none. Matching is "
        "case / whitespace / punctuation-insensitive."
    )
)

if uploaded_file is None:
    st.title("🍫 Mondelez VC Chocolates Dashboard")
    st.info(
        "👈 Upload your source file (parquet or csv) in the "
        "sidebar to get started."
    )
    st.stop()

# Divisor file is required ONLY in Sales mode. In Pieces mode the
# base file's monthly cells are already pieces, so no divisor join
# happens and the file (if uploaded) is ignored.
if (not _already_pieces) and uploaded_divisor_file is None:
    st.title("🍫 Mondelez VC Chocolates Dashboard")
    st.info(
        "👈 You're in **Sales** mode — the source file's monthly "
        "columns hold SALES (₹). Upload a **Divisor file** in the "
        "sidebar so the dashboard can convert sales → pieces, OR "
        "switch the **Base file content** radio to **'Pieces — "
        "already in pieces'** if your file already holds pieces. "
        "Divisor schema: Line No, Line Name (for reference), then "
        "one column per month named Jan'25, Feb'25, … Mar'26."
    )
    st.stop()

# ----- Load the single base file ------------------------------
# Earlier versions of the dashboard supported up to three base
# files of the same schema that were concatenated row-wise. That
# multi-file path has been removed per user request — only one
# base file is accepted now. The file is threaded through the
# existing load_data_pieces() pipeline unchanged.
_merged_bytes = uploaded_file.getvalue()
_merged_name = uploaded_file.name

# Run the sales → pieces conversion ONCE per session (Sales mode)
# OR skip the conversion entirely (Pieces mode — base cells are
# already pieces). Cached on (source bytes, divisor bytes,
# sku-master bytes, mode flag) so filter changes never re-trigger
# it. The SKU master file is optional at the uploader level —
# load_data_pieces decides whether it's actually needed based on
# whether the base file already has Line Name; if it's missing
# AND needed, the conversion errors out with a clear message.
_master_bytes = (
    uploaded_sku_master_file.getvalue()
    if uploaded_sku_master_file is not None else None
)
_master_name = (
    uploaded_sku_master_file.name
    if uploaded_sku_master_file is not None else None
)

# SKU-Size file bytes are threaded into load_data_pieces too, so
# the optional 3rd "Brand" column there can act as a Brand source
# when the base file / SKU master don't carry Brand. Threading
# the bytes (not the parsed table) keeps the conversion's cache
# key complete — changing the SKU-Size file re-runs the join.
_sku_size_bytes_for_load = (
    uploaded_sku_size_file.getvalue()
    if uploaded_sku_size_file is not None else None
)
_sku_size_name_for_load = (
    uploaded_sku_size_file.name
    if uploaded_sku_size_file is not None else None
)

# VC-Category mapping bytes are threaded in too so the VC_MODEL →
# VC_CAT join runs inside the cached conversion (and re-runs when
# the mapping file changes).
_vc_cat_bytes_for_load = (
    uploaded_vc_cat_file.getvalue()
    if uploaded_vc_cat_file is not None else None
)
_vc_cat_name_for_load = (
    uploaded_vc_cat_file.name
    if uploaded_vc_cat_file is not None else None
)

# In Pieces mode the divisor file is ignored — pass None for its
# bytes/name so the cache key still varies with the mode flag.
_div_bytes_for_load = (
    None if _already_pieces
    else uploaded_divisor_file.getvalue()
)
_div_name_for_load = (
    None if _already_pieces
    else uploaded_divisor_file.name
)

# Read MRP file bytes early so load_data_pieces can derive the
# Line No → Line Name mapping from it (replacing the legacy
# SKU master upload).
_mrp_bytes_for_load = (
    uploaded_mrp_file.getvalue()
    if uploaded_mrp_file is not None else None
)
_mrp_name_for_load = (
    uploaded_mrp_file.name
    if uploaded_mrp_file is not None else None
)

df, _conv_report = load_data_pieces(
    _merged_bytes,
    _merged_name,
    _div_bytes_for_load,
    _div_name_for_load,
    _master_bytes,
    _master_name,
    _sku_size_bytes_for_load,
    _sku_size_name_for_load,
    _vc_cat_bytes_for_load,
    _vc_cat_name_for_load,
    _already_pieces,
    _mrp_bytes_for_load,
    _mrp_name_for_load,
)

# Re-publish MONTH_COLS / MRP_MONTH_COLS from the loaded df.
# load_data() populates these globals on a cache miss, but on a
# cache HIT the function body doesn't re-run and the lists would
# stay empty for the rest of the rerun. The detection is pure +
# cheap (just scans column names), so running it again here is
# free and guarantees the globals are populated on every rerun
# of the script.
_vol_cols_now, _mrp_cols_now = _detect_month_cols(df.columns)
MONTH_COLS.clear(); MONTH_COLS.extend(_vol_cols_now)
MRP_MONTH_COLS.clear(); MRP_MONTH_COLS.extend(_mrp_cols_now)

if not _conv_report["ok"]:
    st.title("🍫 Mondelez VC Chocolates Dashboard")
    st.error(
        f"Could not convert sales → pieces: {_conv_report['msg']}"
    )
    st.stop()

st.sidebar.success(f"✅ {_conv_report['msg']}")

# Warn about base rows dropped for missing Line Name (after the
# SKU master join). These are Line Nos with no name in the
# master AND no name in the base — they can't be analysed.
_no_name = _conv_report.get("dropped_no_name", [])
if _no_name:
    with st.sidebar.expander(
        f"⚠️ {len(_no_name)} Line No(s) dropped (missing Line Name)"
    ):
        st.write(
            "\n".join(
                f"• {s}" for s in _no_name[:200]
            )
        )
        if len(_no_name) > 200:
            st.caption(f"… and {len(_no_name) - 200} more")

# Warn about SKUs dropped for being absent from the divisor file.
if _conv_report["dropped_skus"]:
    with st.sidebar.expander(
        f"⚠️ {len(_conv_report['dropped_skus'])} SKU(s) excluded "
        "(missing from divisor file)"
    ):
        st.write(
            "\n".join(
                f"• {s}" for s in _conv_report["dropped_skus"][:200]
            )
        )
        if len(_conv_report["dropped_skus"]) > 200:
            st.caption(
                f"… and {len(_conv_report['dropped_skus']) - 200} more"
            )

# Notice when Brand wasn't supplied on the base file and we had
# to derive it from Line Name. Helps the user understand why a
# Brand-level breakdown might look unfamiliar.
if _conv_report.get("brand_derived"):
    _brand_src = _conv_report.get("brand_source", "")
    if _brand_src == "SKU-Size file":
        # Some/all brands came from the SKU-Size file's 3rd column,
        # but at least one row still had to fall back to Line Name.
        st.sidebar.info(
            "ℹ️ Brand was taken from the SKU-Size file's 3rd "
            "column where available; any SKUs missing there had "
            "their Brand derived from the first word of the Line "
            "Name."
        )
    else:
        st.sidebar.info(
            "ℹ️ No **Brand** column was found on the base file. "
            "Brand was derived from the first word of each Line "
            "Name (e.g. 'Cadbury Dairy Milk 13g' → 'Cadbury'). "
            "For a more accurate breakdown, add a Brand column to "
            "the base file, the SKU master file, or a 3rd column "
            "on the SKU-Size file."
        )
elif _conv_report.get("brand_source") == "SKU-Size file":
    # Brand fully resolved from the SKU-Size file's 3rd column.
    st.sidebar.info(
        "ℹ️ Brand was sourced from the SKU-Size file's 3rd column."
    )

# Notice about the VC-Category mapping file, if one was applied.
_vc_applied = _conv_report.get("vc_cat_applied", 0)
if uploaded_vc_cat_file is not None:
    if _vc_applied:
        st.sidebar.success(
            f"✅ VC_CAT set from the VC-Category file for "
            f"{_vc_applied:,} row(s) (matched on VC_MODEL)."
        )
    else:
        st.sidebar.warning(
            "⚠️ A VC-Category file was uploaded but none of its "
            "VC Models matched the source file's VC_MODEL values "
            "(or the source file has no VC_MODEL column). VC_CAT "
            "was left unchanged."
        )

# Resolve the optional outlet-name column once, used later by
# the field-team action list.
COL_OUTLET_NAME = detect_outlet_name_col(df)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.divider()

st.sidebar.title("Filters")

# Clear-all-filters button. Wipes every filter / curation /
# selector across the sidebar AND both period sheets.
if st.sidebar.button(
    "🧹  Clear All Filters",
    use_container_width=True,
    help=(
        "Resets sidebar filters, visualization filters, "
        "Auto-Insights curation, segment selectors, heatmap "
        "metric, and the Compare-Segments left/right inputs "
        "for the selected analysis period."
    )
):
    # Sidebar + visual filters (sb_*) — but keep the uploaders
    # and the month picker so the user doesn't have to re-upload
    # / re-select after wiping filters.
    _keep_keys = {
        "sb_uploaded_file",
        "sb_uploaded_divisor_file",
        "sb_uploaded_sku_master_file",
        "sb_uploaded_mrp_file",
        "sb_uploaded_gm_file",
        "sb_uploaded_sku_size_file",
        "sb_uploaded_vc_cat_file",
        "sb_data_mode",
        "sb_selected_months",
    }
    for k in list(st.session_state.keys()):
        if k in _keep_keys:
            continue
        if k.startswith("sb_"):
            del st.session_state[k]
            continue
        # Per-period curation, segment, heatmap, trend,
        # compare-segments and report state. "sel_" is the new
        # period_id; the legacy "l3_" / "l15_" prefixes are kept
        # to clean up state from older versions of the app that
        # may still be in the user's session.
        if any(k.startswith(p) for p in ("sel_", "l3_", "l15_")):
            del st.session_state[k]
    st.rerun()

st.sidebar.divider()


# ---------- SKU TYPE: Cooler vs Ambient ----------

sku_type_filter = st.sidebar.radio(
    "SKU Type",
    options=["All", "Cooler", "Ambient"],
    index=0,
    horizontal=True,
    key="sb_sku_type",
    help=(
        "**Display + ranking-scope filter.** Cooler = Silk + "
        "Bournville + Temptations. Ambient = rest of portfolio "
        "(CDM, 5 Star, Perk, Fuse, Nutties, Crispello, Milkinis, "
        "etc.). Picking Cooler or Ambient narrows charts and "
        "tables to that subset, and on the Segment SKU Lists tab "
        "the Critical / Important / Moderate tier boundaries are "
        "recomputed *within* that subset so the ranking is "
        "subset-relative. Per-SKU values (Throughput, "
        "Penetration %, Latest MRP, Rank Score) and the segment "
        "outlet universe stay unchanged from the All view."
    )
)


def create_sidebar_filter(column, label=None):
    """Build a sidebar multiselect for `column`. Returns [] if the
    column isn't in the loaded df, so callers can stay declarative
    and the rest of the app skips the filter naturally."""
    if column not in df.columns:
        return []
    return st.sidebar.multiselect(
        label or column,
        sorted(
            df[column]
            .dropna()
            .astype(str)
            .unique()
        ),
        key=f"sb_{column}"
    )

# Declarative list of (column, label) pairs the dashboard knows
# about. Any entry whose column is missing from the uploaded file
# is silently skipped by create_sidebar_filter — the rest of the
# app reads the resulting empty list as "no filter".
KNOWN_FILTERS = [
    (COL_ASM,          None),
    (COL_CHANNEL,      None),
    (COL_PCTYPE,       "PC Type"),
    (COL_RD,           None),
    (COL_RE,           None),
    (COL_REGION,       None),
    (COL_REGION_CAT,   None),
    (COL_STATUS,       None),
    (COL_VC,           None),
    (COL_VC_CATEGORY,  None),
    (COL_SETTY,        None),
]

# Build the filter values, keyed by canonical column name. Each
# call is a no-op (returns []) for columns the source file
# doesn't carry.
_sidebar_filters = {
    col: create_sidebar_filter(col, label) for col, label in KNOWN_FILTERS
}

# Back-compat aliases — keep the legacy variable names alive so
# the rest of the file doesn't need touching.
asm_filter         = _sidebar_filters[COL_ASM]
channel_filter     = _sidebar_filters[COL_CHANNEL]
pctype_filter      = _sidebar_filters[COL_PCTYPE]
rd_filter          = _sidebar_filters[COL_RD]
re_filter          = _sidebar_filters[COL_RE]
region_filter      = _sidebar_filters[COL_REGION]
region_cat_filter  = _sidebar_filters[COL_REGION_CAT]
status_filter      = _sidebar_filters[COL_STATUS]
vc_filter          = _sidebar_filters[COL_VC]
vc_category_filter = _sidebar_filters[COL_VC_CATEGORY]
setty_filter       = _sidebar_filters[COL_SETTY]

# Auto-discover extra categorical columns the file carries that
# we don't already know about. Treat any text/object column with
# a sensible number of distinct values (≤ 200) as a candidate
# filter dimension — the user might have added new geography,
# cluster, or programme tags. Numeric and monthly-sales columns
# are excluded; so are the SKU / Brand / outlet-name columns
# (handled by their own widgets) and the period aggregates.
_known_filter_cols = {col for col, _ in KNOWN_FILTERS}
_reserved_cols = (
    _known_filter_cols
    | {COL_OUTLET, COL_BRAND, COL_SKU, COL_SKU_TIER, COL_LINE_NO}
    | set(MONTH_COLS)
    | {"L3M", "L15M"}
    | set(OUTLET_NAME_CANDIDATES)
)
_extra_filter_cols = []
for c in df.columns:
    if c in _reserved_cols:
        continue
    if df[c].dtype != object and not pd.api.types.is_string_dtype(df[c]):
        continue
    nunique = df[c].nunique(dropna=True)
    if 1 < nunique <= 200:
        _extra_filter_cols.append(c)

extra_filters = {}
if _extra_filter_cols:
    with st.sidebar.expander(
        f"➕ Other filters ({len(_extra_filter_cols)} detected)",
        expanded=False,
    ):
        for c in _extra_filter_cols:
            extra_filters[c] = st.multiselect(
                c,
                sorted(df[c].dropna().astype(str).unique()),
                key=f"sb_extra_{c}",
            )
            # Mirror the selection so the unified filter map below
            # picks it up.
            _sidebar_filters[c] = extra_filters[c]

def _safe_visual_multiselect(label, column, key):
    """Brand / SKU / Tier visualization-only multiselects.
    Returns an empty list if the column is missing — the rest of
    the app treats that as 'no visual filter' which is the
    correct behavior."""
    if column not in df.columns:
        return []
    return st.sidebar.multiselect(
        label,
        sorted(df[column].dropna().astype(str).unique()),
        key=key,
    )

brand_visual_filter = _safe_visual_multiselect(
    "Brand (Visualization Only)", COL_BRAND, "sb_brand_visual"
)

sku_tier_visual_filter = _safe_visual_multiselect(
    "SKU_tier (Visualization Only)", COL_SKU_TIER, "sb_sku_tier_visual"
)

sku_visual_filter = _safe_visual_multiselect(
    "SKU name (Visualization Only)", COL_SKU, "sb_sku_visual"
)

exclude_skus = _safe_visual_multiselect(
    "Exclude SKU from Analysis", COL_SKU, "sb_exclude_skus"
)

# ---------- DATA SMOOTHING (trim outliers by overall sales) ----------

st.sidebar.divider()
st.sidebar.markdown("##### 📉 Smooth Data (trim outlier outlets)")

# Basis for the smoothing percentile — pieces (existing default,
# uses L15M volume column) or sale value (volume × MRP, looked up
# from the uploaded MRP master). The MRP file is parsed ONCE per
# session via @st.cache_resource keyed on its raw bytes, and the
# per-outlet value totals are cached too — so flipping this radio
# is instant after the first compute.
if uploaded_mrp_file is not None:
    _mrp_bytes = uploaded_mrp_file.getvalue()
    _mrp_name = uploaded_mrp_file.name
    _mrp_lookup, _mrp_status = load_mrp_table(_mrp_bytes, _mrp_name)
else:
    _mrp_bytes = None
    _mrp_name = None
    _mrp_lookup, _mrp_status = (
        None,
        "No MRP file uploaded — upload one in the sidebar to "
        "enable value-based smoothing.",
    )
_mrp_available = _mrp_lookup is not None

# ---- Load optional Gross-Margin file (separate from MRP) ----
if uploaded_gm_file is not None:
    _gm_bytes = uploaded_gm_file.getvalue()
    _gm_name = uploaded_gm_file.name
    _gm_lookup, _gm_status = load_gm_table(_gm_bytes, _gm_name)
else:
    _gm_bytes = None
    _gm_name = None
    _gm_lookup, _gm_status = (
        None,
        "No GM file uploaded — upload one to unlock the "
        "SKU Priority Lister tab.",
    )
_gm_available = _gm_lookup is not None and not _gm_lookup.empty

if _gm_available:
    st.sidebar.caption(f"📊 {_gm_status}")
else:
    st.sidebar.caption(f"ℹ️ {_gm_status}")

# ---- SKU Size table (optional, drives the VC Planogram Builder) ----
if uploaded_sku_size_file is not None:
    _sz_bytes = uploaded_sku_size_file.getvalue()
    _sz_name = uploaded_sku_size_file.name
    _sku_size_lookup, _sku_size_status = load_sku_size_table(
        _sz_bytes, _sz_name
    )
else:
    _sku_size_lookup, _sku_size_status = (
        None,
        "No SKU-Size file uploaded — upload one to unlock the "
        "VC Planogram Builder tab.",
    )
_sku_size_available = (
    _sku_size_lookup is not None and not _sku_size_lookup.empty
)
if _sku_size_available:
    st.sidebar.caption(f"📏 {_sku_size_status}")
else:
    st.sidebar.caption(f"ℹ️ {_sku_size_status}")

smoothing_basis = st.sidebar.radio(
    "Smoothing basis",
    options=["Pieces (L15M pieces)", "Sale Value (pieces × MRP)"],
    index=0,
    horizontal=False,
    key="sb_smoothing_basis",
    help=(
        "Pieces  — original behaviour: ranks outlets by total "
        "L15M units sold.\n\n"
        "Sale Value — multiplies each month's pieces by that "
        "month's MRP per SKU (from the uploaded MRP file), then "
        "ranks outlets by total ₹ sales across the 15 months. "
        "MRP file is parsed once per session and cached."
    ),
    disabled=not _mrp_available
)

# Always surface the MRP load status — previously this was only
# shown when the smoothing-basis radio was set to "Sale Value",
# which meant a successfully-loaded MRP file produced ZERO visual
# confirmation in the default "Pieces" mode. That made it easy to
# think the file was loaded when it wasn't (or vice-versa), and
# downstream tabs that silently need MRP — notably the SKU Priority
# Lister — would render all-zero values with no obvious cause.
if not _mrp_available:
    st.sidebar.caption(f"ℹ️ {_mrp_status}")
else:
    st.sidebar.caption(f"💰 {_mrp_status}")

smoothing_range = st.sidebar.slider(
    "Keep outlets in this percentile range of total sales",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=1,
    key="sb_smoothing",
    help=(
        "Computes the chosen total (pieces or value) per outlet, "
        "then keeps only outlets whose total falls within the "
        "selected percentile range. Trim the top to drop "
        "pieces outliers; trim the bottom to drop dormant / "
        "near-zero outlets. Default (0,100) = no trimming."
    )
)

# =========================================================
# APPLY SIDEBAR FILTERS  (returns the slice of df)
# =========================================================

# =========================================================
# APPLY SIDEBAR FILTERS  (returns the slice of df)
# =========================================================

# Unified filter map: known filters + any auto-discovered extras.
# Columns missing from df were already given an empty selection
# by create_sidebar_filter, so the loop below skips them.
filter_map = dict(_sidebar_filters)

filtered_df = df.copy()

for col, vals in filter_map.items():

    if not vals:
        continue
    if col not in filtered_df.columns:
        continue

    # Case + whitespace + comparison-prefix insensitive .isin().
    # Both the chosen values and the column are normalised
    # before comparison so 'High End Grocer', 'HIGH END GROCER',
    # 'highendgrocer' all match.
    filtered_df = filtered_df[
        fuzzy_isin(filtered_df[col], vals)
    ]

if exclude_skus and COL_SKU in filtered_df.columns:

    filtered_df = filtered_df[
        ~fuzzy_isin(filtered_df[COL_SKU], exclude_skus)
    ]

# NOTE: The Cooler / Ambient SKU Type radio is a
# VISUALIZATION-ONLY filter. It must NOT touch filtered_df,
# otherwise it would shrink the outlet universe and bias every
# downstream calculation (penetration denominators, segment
# throughputs, KPI tiles, etc.). The filter is instead applied
# alongside the other "(Visualization Only)" filters at each
# display layer — see the visual-filter blocks further down.

# ---------- DATA SMOOTHING ----------
# Trim outlets whose total falls outside the selected percentile
# range. Operates on the OVERALL outlet total (summing every SKU
# row for that outlet), so we drop outliers at the outlet level
# rather than the SKU level. Basis is either L15M pieces (default)
# or sale value = Σ (monthly_vol × monthly_MRP) — see the radio
# above.
sm_lo, sm_hi = smoothing_range
_use_value_basis = (
    smoothing_basis.startswith("Sale Value") and _mrp_available
)

if (sm_lo, sm_hi) != (0, 100) and not filtered_df.empty:

    if _use_value_basis:
        # Pull the cached per-outlet value totals (computed once
        # against the *unfiltered* df), then restrict to outlets
        # still in filtered_df so percentile cuts respect the
        # active sidebar filters.
        full_value_totals = compute_outlet_value_totals(
            uploaded_file.getvalue(),
            uploaded_file.name,
            _div_bytes_for_load,
            _div_name_for_load,
            _master_bytes,
            _master_name,
            _mrp_bytes,
            _mrp_name,
            _already_pieces,
        )
        if full_value_totals is None:
            outlet_totals = (
                filtered_df.groupby(COL_OUTLET)["L15M"].sum()
            )
            _basis_label = "L15M pieces (MRP unavailable)"
        else:
            outlets_in_scope = filtered_df[COL_OUTLET].unique()
            outlet_totals = full_value_totals.reindex(
                outlets_in_scope
            ).dropna()
            _basis_label = "L15M sale value (₹)"
    else:
        outlet_totals = (
            filtered_df.groupby(COL_OUTLET)["L15M"].sum()
        )
        _basis_label = "L15M pieces"

    if not outlet_totals.empty:
        lo_cut = (
            outlet_totals.quantile(sm_lo / 100.0)
            if sm_lo > 0 else outlet_totals.min() - 1
        )
        hi_cut = (
            outlet_totals.quantile(sm_hi / 100.0)
            if sm_hi < 100 else outlet_totals.max() + 1
        )

        keep_outlets = outlet_totals[
            (outlet_totals >= lo_cut) &
            (outlet_totals <= hi_cut)
        ].index

        filtered_df = filtered_df[
            filtered_df[COL_OUTLET].isin(keep_outlets)
        ]

        st.sidebar.caption(
            f"📉 Smoothing kept {len(keep_outlets):,} of "
            f"{len(outlet_totals):,} outlets "
            f"(P{sm_lo}–P{sm_hi} of {_basis_label})."
        )

# =========================================================
# MONTH PICKER  (drives the dynamic analysis period)
# =========================================================
# Replaces the old fixed L3M / L15M tabs. The user ticks which
# months to focus the analysis on; downstream code reads a
# single dynamic period column built from the selected months.
#
# Only months actually present in the source file are offered —
# files with fewer than 15 months still work, and an added
# month column (e.g. "Apr26_V") would also be picked up here.

st.sidebar.divider()
st.sidebar.markdown("##### 📅 Analysis Period")

_available_months_in_df = [m for m in MONTH_COLS if m in df.columns]
# Pretty labels for the picker (strip the "_V" suffix).
_month_pretty = {m: m.replace("_V", "") for m in _available_months_in_df}

selected_months = st.sidebar.multiselect(
    "Select months for analysis",
    options=_available_months_in_df,
    default=_available_months_in_df,  # default: all months
    format_func=lambda m: _month_pretty.get(m, m),
    key="sb_selected_months",
    help=(
        "Pick the months you want the dashboard to focus on. "
        "Throughput, penetration, KPIs, ranks and segment "
        "tables are computed from the sum of pieces across "
        "exactly these months. The 'prior period' used for "
        "Δ metrics is the *remaining* months in the data."
    ),
)

if not selected_months:
    st.sidebar.warning(
        "Pick at least one month to run the analysis."
    )
    st.stop()

# Build the dynamic period column on BOTH the unfiltered base
# (`df`) and the filtered slice (`filtered_df`) so every
# downstream call that references the period column finds it.
PERIOD_COL_NAME = "_SELECTED_PIECES"
PERIOD_MONTHS = len(selected_months)

# Pretty label for the period (e.g. "Jan'25 + Mar'25 + Apr'25"
# when ≤ 4 months, otherwise "5 months: Jan'25 … May'25"). This
# is what shows up in chart titles and report headings.
def _build_period_label(months):
    pretty = [_month_pretty.get(m, m) for m in months]
    if len(pretty) <= 4:
        return " + ".join(pretty)
    return f"{len(pretty)} months: {pretty[0]} … {pretty[-1]}"

PERIOD_LABEL = _build_period_label(selected_months)

df[PERIOD_COL_NAME] = df[selected_months].sum(axis=1)
filtered_df[PERIOD_COL_NAME] = (
    filtered_df[selected_months].sum(axis=1)
)

st.sidebar.caption(
    f"🗓️ Analysis window: **{PERIOD_LABEL}** "
    f"({PERIOD_MONTHS} month{'s' if PERIOD_MONTHS != 1 else ''})"
)

# =========================================================
# HELPERS
# =========================================================

def build_outlet_sku_df(source_df, pieces_col):
    """
    Aggregate to outlet x SKU level using the chosen period column
    as the pieces measure. Outlets that didn't sell that SKU in the
    period (pieces <= 0) are dropped.

    Only DIM_COLS columns that are actually present in source_df
    are used as groupby keys — this keeps the function robust to
    source files that omit one or more dimension columns.
    """

    work = source_df.copy()

    work = work[work[pieces_col] > 0]

    group_cols = [c for c in DIM_COLS if c in work.columns]

    return (
        work
        .groupby(group_cols, as_index=False)
        .agg(pieces_sold=(pieces_col, "sum"))
    )


def compute_matrix(segment_df, period_months=1):
    """
    Build the Brand x SKU matrix. Throughput is expressed as a
    MONTHLY AVERAGE (period total volume / outlets / months in
    period) so that L3M / L15M sheets are comparable.
    """

    if segment_df.empty:
        return pd.DataFrame()

    total_outlets = (
        segment_df[COL_OUTLET]
        .nunique()
    )

    matrix_df = (
        segment_df
        .groupby(
            [COL_BRAND, COL_SKU],
            as_index=False
        )
        .agg(
            outlet_count=(COL_OUTLET, "nunique"),
            total_volume=("pieces_sold", "sum")
        )
    )

    matrix_df["Penetration %"] = (
        matrix_df["outlet_count"] /
        total_outlets
    ) * 100

    matrix_df["Throughput"] = (
        matrix_df["total_volume"] /
        matrix_df["outlet_count"] /
        period_months
    )

    matrix_df["Throughput_Log"] = np.log10(
        matrix_df["Throughput"] + 1
    )

    matrix_df["Penetration %"] = matrix_df["Penetration %"].round(1)
    matrix_df["Throughput"] = matrix_df["Throughput"].round(1)

    return matrix_df


# =========================================================
# AUTO-INSIGHT ENGINE  (pure pandas, runs on every tab)
# =========================================================

def _safe_div(a, b):
    return np.where(b > 0, a / b, 0)


def compute_sku_change(source_df):
    """
    For each SKU, compare the **user-selected** months against
    the remaining months in the data (the "prior" window).

    Logic:
        • current monthly throughput =
              Σ pieces(selected months) ÷ outlets_active_current
              ÷ #selected_months
        • prior monthly throughput =
              Σ pieces(prior months) ÷ outlets_active_prior
              ÷ #prior_months
        • Δ Pen (pp)        = pen_current − pen_prior
        • Δ Throughput (mo) = thr_current − thr_prior

    The split between "current" and "prior" is driven by the
    sidebar month picker (`selected_months`). If the user
    selected every month in the file there's no prior window —
    Δ values are then reported as 0.
    """

    if source_df.empty:
        return pd.DataFrame(columns=[
            COL_BRAND, COL_SKU,
            "Δ Pen (pp)", "Δ Throughput (mo)"
        ])

    months_present = [m for m in MONTH_COLS if m in source_df.columns]
    sel_months = [m for m in selected_months if m in months_present]
    prior_months = [m for m in months_present if m not in sel_months]

    n_sel = len(sel_months)
    n_pri = len(prior_months)

    work = source_df.copy()
    work["_cur_pieces"] = (
        work[sel_months].sum(axis=1) if sel_months else 0.0
    )
    work["_prior_pieces"] = (
        work[prior_months].sum(axis=1) if prior_months else 0.0
    )

    total_outlets = work[COL_OUTLET].nunique()
    if total_outlets <= 0:
        return pd.DataFrame(columns=[
            COL_BRAND, COL_SKU,
            "Δ Pen (pp)", "Δ Throughput (mo)"
        ])

    # Current side
    cur_active = work[work["_cur_pieces"] > 0]
    cur_grp = (
        cur_active
        .groupby([COL_BRAND, COL_SKU])
        .agg(
            cur_vol=("_cur_pieces", "sum"),
            cur_outlets=(COL_OUTLET, "nunique")
        )
        .reset_index()
    )

    # Prior side
    pri_active = work[work["_prior_pieces"] > 0]
    pri_grp = (
        pri_active
        .groupby([COL_BRAND, COL_SKU])
        .agg(
            pri_vol=("_prior_pieces", "sum"),
            pri_outlets=(COL_OUTLET, "nunique")
        )
        .reset_index()
    )

    chg = cur_grp.merge(
        pri_grp, on=[COL_BRAND, COL_SKU], how="outer"
    ).fillna(0)

    chg["pen_cur"] = (chg["cur_outlets"] / total_outlets) * 100
    chg["pen_pri"] = (chg["pri_outlets"] / total_outlets) * 100

    chg["thr_cur"] = (
        _safe_div(chg["cur_vol"], chg["cur_outlets"]) / max(n_sel, 1)
    )
    chg["thr_pri"] = (
        _safe_div(chg["pri_vol"], chg["pri_outlets"]) / max(n_pri, 1)
    )

    # If there's no prior window (user picked every month), Δ
    # is meaningless — surface zero rather than NaN/garbage.
    if n_pri == 0:
        chg["Δ Pen (pp)"] = 0.0
        chg["Δ Throughput (mo)"] = 0.0
    else:
        chg["Δ Pen (pp)"] = (
            chg["pen_cur"] - chg["pen_pri"]
        ).round(1)
        chg["Δ Throughput (mo)"] = (
            chg["thr_cur"] - chg["thr_pri"]
        ).round(1)

    return chg[[
        COL_BRAND, COL_SKU,
        "Δ Pen (pp)", "Δ Throughput (mo)"
    ]]


def compute_sku_penetration_impact(source_df):
    """
    Per-SKU "Penetration Impact" — the slope `b` from a simple
    linear regression of monthly throughput on penetration,
    using rolling 3-month windows as the observations.

    Definition of penetration here (per user spec):
        A SKU is "penetrated" in a 3-month window if it is sold
        in any outlet during those 3 months. Penetration % in
        the window = (# outlets where that SKU's 3-month volume
        > 0) ÷ (total outlets in scope) × 100.

    Throughput in the window:
        (sum of SKU volume across 3 months)
          ÷ (# outlets active in those 3 months)
          ÷ 3       (to express per outlet, per month)

    We slide a 3-month window across the 15 monthly columns,
    yielding up to 13 (penetration, throughput) observations per
    SKU. We then fit y = a + b * x and return `b` as
    "Penetration Impact". Interpretation: extra units / outlet /
    month for each +1pp of penetration, holding everything else
    equal.

    Returns DataFrame[Brand, Line Name, Penetration Impact].
    """

    if source_df.empty:
        return pd.DataFrame(columns=[
            COL_BRAND, COL_SKU, "Penetration Impact"
        ])

    # Only use month columns that actually exist in the data.
    months_present = [m for m in MONTH_COLS if m in source_df.columns]
    if len(months_present) < 3:
        return pd.DataFrame(columns=[
            COL_BRAND, COL_SKU, "Penetration Impact"
        ])

    total_outlets = source_df[COL_OUTLET].nunique()
    if total_outlets <= 0:
        return pd.DataFrame(columns=[
            COL_BRAND, COL_SKU, "Penetration Impact"
        ])

    # Build rolling 3-month windows: (Jan,Feb,Mar), (Feb,Mar,Apr)…
    windows = [
        months_present[i : i + 3]
        for i in range(len(months_present) - 2)
    ]

    work = source_df[[COL_OUTLET, COL_BRAND, COL_SKU] + months_present].copy()

    # For each window, compute per-SKU (penetration, throughput).
    # We do it column-by-column to keep memory reasonable.
    rows = []
    for w_idx, w_cols in enumerate(windows):
        # Per-outlet × SKU volume across the 3 months in window
        w_vol = work[w_cols].sum(axis=1)
        tmp = pd.DataFrame({
            COL_BRAND: work[COL_BRAND].values,
            COL_SKU: work[COL_SKU].values,
            COL_OUTLET: work[COL_OUTLET].values,
            "w_vol": w_vol.values,
        })
        # Active = any positive volume in the 3 months
        active = tmp[tmp["w_vol"] > 0]
        if active.empty:
            continue
        grp = (
            active.groupby([COL_BRAND, COL_SKU], as_index=False)
            .agg(
                vol=("w_vol", "sum"),
                outlets=(COL_OUTLET, "nunique"),
            )
        )
        grp["pen_pct"] = (grp["outlets"] / total_outlets) * 100
        grp["throughput"] = grp["vol"] / grp["outlets"] / 3.0
        grp["window"] = w_idx
        rows.append(grp[[COL_BRAND, COL_SKU,
                         "pen_pct", "throughput", "window"]])

    if not rows:
        return pd.DataFrame(columns=[
            COL_BRAND, COL_SKU, "Penetration Impact"
        ])

    obs = pd.concat(rows, ignore_index=True)

    # Per-SKU OLS slope: cov(x,y) / var(x). Need at least 2
    # distinct penetration values to fit a line; otherwise NaN.
    def _slope(g):
        x = g["pen_pct"].to_numpy(dtype=float)
        y = g["throughput"].to_numpy(dtype=float)
        if len(x) < 2:
            return np.nan
        x_var = x.var()
        if x_var < 1e-12:
            return np.nan
        # np.cov returns 2x2 matrix; element [0,1] is cov(x,y)
        cov_xy = np.cov(x, y, ddof=0)[0, 1]
        return cov_xy / x_var

    impact = (
        obs.groupby([COL_BRAND, COL_SKU])
        .apply(_slope, include_groups=False)
        .reset_index()
        .rename(columns={0: "Penetration Impact"})
    )

    # Older pandas returns the column unnamed; handle both shapes
    if "Penetration Impact" not in impact.columns:
        last_col = impact.columns[-1]
        impact = impact.rename(
            columns={last_col: "Penetration Impact"}
        )

    impact["Penetration Impact"] = (
        impact["Penetration Impact"].astype(float).round(2)
    )
    return impact[[COL_BRAND, COL_SKU, "Penetration Impact"]]


# =========================================================
# REALISTIC INCREMENTAL THROUGHPUT (per new outlet added)
# =========================================================
#
# Background and motivation
# -------------------------
# "Penetration Impact" (the slope `b` above) is fundamentally a
# *historical* statistic: it summarises what happened in the past
# 15 months — when penetration of this SKU moved by ±1pp across
# rolling 3-month windows, monthly throughput typically moved by
# `b` units/outlet/month. It is *not* a forecast.
#
# The companion field-team question, "if I take this SKU to 30
# new stores, how much extra volume do I get?", needs a different
# number. Reading `b` directly off the slope and multiplying by N
# implicitly assumes every new store performs like the average
# existing store — which is wrong for two reasons:
#
#   1. Adding stores does not retroactively change the throughput
#      of the stores that already stock the SKU. Their volume is
#      what it is.
#   2. The next 30 stores you reach are, by construction, *not*
#      the best 30. The easy wins are taken. New marginal stores
#      typically convert at a fraction of the existing average
#      throughput — but they are still real stores in the same
#      segment that sells chocolates, so the discount has a
#      sensible floor.
#
# Design correction (v15) — DECOUPLED HEADROOM
# --------------------------------------------
# The earlier (v14 and prior) formula multiplied throughput by
# `headroom = (100 − P) / 100` directly. That conflated two
# distinct quantities:
#
#   (a) HOW MANY new outlets can I still add?   →  bounded by
#       headroom — at P = 98% there are very few outlets left
#       to reach.
#   (b) WHAT will each new outlet sell?          →  a function
#       of segment throughput, NOT of how saturated the SKU
#       happens to be in that segment.
#
# Conflating (a) and (b) crushed the per-outlet number toward
# zero for the strongest, most-penetrated SKUs — exactly the
# SKUs where a field rep adding ONE more store would realistic-
# ally sell tens of units, not <1. So v15 separates them:
#
#   • Headroom continues to gate the TOTAL prize when summed
#     across the long tail of unreached outlets — but that is
#     the universe N × headroom calculation downstream, not
#     this per-outlet number.
#   • Per-outlet expected throughput stays a stable fraction
#     of segment T, bounded below by a "saturation floor" so
#     that even at P → 100% the next store still sells a
#     non-trivial share of segment average. The 2% of stores
#     not yet stocking a near-ubiquitous SKU are weaker than
#     average — but they are still stores in that VC band, and
#     will sell well above zero.
#
# Definition (v15)
# ----------------
# Realistic Multiplier
#   = marginal_efficiency × saturation_drag × historical_tilt
#
#   marginal_efficiency  = 0.70               (a calibration knob)
#       "Even an unsaturated SKU placed in a fresh outlet
#       converts at ~70% of an existing store's average
#       throughput, before further saturation adjustments."
#       The new outlet is by construction not yet rotating at
#       steady-state, and shopper familiarity has not built up.
#
#   saturation_drag       = SAT_FLOOR + (1 − SAT_FLOOR) ×
#                           (1 − P/100)        ∈ [SAT_FLOOR, 1.0]
#       Linear drag from 1.0 at P=0 down to SAT_FLOOR at
#       P=100. Replaces the old `headroom` term. At P=98% the
#       drag is ~0.31 (not 0.02), reflecting that the few
#       remaining outlets are weaker — but still real outlets.
#       SAT_FLOOR = 0.30 calibrates the floor for fully-
#       saturated SKUs: a new outlet for an everywhere-SKU
#       still sells ~30% of segment average.
#
#   historical_tilt      = smooth bounded function of b / T
#       Nudges the multiplier up when the SKU has *historically*
#       expanded healthily (positive `b` means throughput rose
#       as distribution widened — a "good distribution story"),
#       and trims it down when `b` is negative ("the SKU thins
#       out as it spreads"). Uses tanh so the tilt asymptotes
#       smoothly to ±20% rather than hard-clipping — this preserves
#       SKU-to-SKU ranking differentiation at the strong end (hard
#       clipping flattens all strong SKUs onto the same ceiling)
#       and stays numerically stable for low-throughput SKUs.
#
# Realistic Incremental per New Outlet (units / month / outlet)
#   = Throughput × Realistic Multiplier
#
# Hard ceiling: the final multiplier is clipped to [0, 0.95] so
# the metric can never silently claim "a new store will sell more
# than your existing average store" — that's exactly the over-
# promise we're trying to avoid.
#
# Practical multiplier band (with current constants):
#   • Best case  (P=0,  tilt=1.2): 0.70 × 1.00 × 1.20 = 0.84
#   • Typical    (P=50, tilt=1.0): 0.70 × 0.65 × 1.00 = 0.46
#   • Saturated  (P=98, tilt=1.0): 0.70 × 0.31 × 1.00 = 0.22
#
# Reading the number
# ------------------
# Says directly: "On average, each NEW store I open for this SKU
# will sell about X units per month, given how saturated I
# already am and how the SKU has historically behaved as it
# spread." Multiply by the number of new stores you actually open
# to get the realistic extra monthly volume.

# Exposed as module-level constants — one place to tune.
# v15 recalibration:
#   MARGINAL_EFFICIENCY 0.50 → 0.70  (un-dampen the baseline)
#   SAT_FLOOR (new)         0.30    (floor at full saturation)
MARGINAL_EFFICIENCY = 0.70
SAT_FLOOR = 0.30


def compute_realistic_incremental_per_outlet(matrix_df):
    """
    Compute the "Realistic Multiplier" and "Incremental TP /
    New Outlet" columns and append them to a copy of `matrix_df`.

    `matrix_df` must already have columns:
        - "Penetration %"          (current penetration)
        - "Throughput"             (current monthly throughput)
        - "Penetration Impact"     (historical slope b; optional)

    The function tolerates a missing "Penetration Impact" column —
    in that case, historical_tilt collapses to 1.0 and the
    multiplier reduces to marginal_efficiency × saturation_drag.

    Returns a new DataFrame with two extra columns appended:
        "Realistic Multiplier"          (dimensionless, < 1)
        "Incremental TP / New Outlet"   (units / outlet / month)
    """
    if matrix_df is None or matrix_df.empty:
        return matrix_df

    out = matrix_df.copy()

    if ("Penetration %" not in out.columns
            or "Throughput" not in out.columns):
        return out

    pen = out["Penetration %"].astype(float).clip(lower=0, upper=100)
    thr = out["Throughput"].astype(float).clip(lower=0)

    # Saturation drag (v15): replaces the old `headroom` term.
    # Goes from 1.0 at P=0 to SAT_FLOOR at P=100, linearly. A new
    # outlet for a near-saturated SKU still sells SAT_FLOOR of
    # segment average — not zero. The few outlets that haven't
    # picked up a near-ubiquitous SKU are weaker than the median,
    # but they're still real stores in the same VC band.
    saturation_drag = SAT_FLOOR + (1.0 - SAT_FLOOR) * (
        (100.0 - pen) / 100.0
    )

    # Historical tilt: smooth bounded function of b / T. Centred
    # at 1.0, asymptotically bounded to (0.8, 1.2) by tanh so a
    # single noisy slope can't flip the multiplier sign or push it
    # past the 0.95 ceiling. Unlike hard clipping, tanh preserves
    # ranking differentiation between strong SKUs (clipping flattens
    # them all to the same +0.2 ceiling), and the b / max(T, 0.5)
    # denominator stabilises low-throughput SKUs where b / T would
    # otherwise explode.
    if "Penetration Impact" in out.columns:
        b = out["Penetration Impact"].astype(float).fillna(0.0)
        # tilt_raw = b / max(T, 0.5) → "what fraction of current
        # throughput does each +1pp of penetration historically
        # add?". The floor of 0.5 on the denominator prevents
        # tiny-throughput SKUs from producing absurd ratios.
        tilt_raw = b / np.maximum(thr, 0.5)
        # Map raw ratio onto (0.8, 1.2) via tanh. Factor of 5.0
        # inside tanh sets the slope at the origin so a slope
        # contributing ~10% of current throughput per pp moves the
        # multiplier ~9–10% — comparable in feel to the old hard-
        # clip rule at moderate values but smooth at the edges.
        tilt = 1.0 + 0.2 * np.tanh(5.0 * tilt_raw)
    else:
        tilt = 1.0

    raw_mult = MARGINAL_EFFICIENCY * saturation_drag * tilt
    # Hard ceiling at 0.95: realism guarantee — new stores never
    # outperform the existing average store in this metric.
    realistic_mult = np.clip(raw_mult, 0.0, 0.95)

    out["Realistic Multiplier"] = np.round(realistic_mult, 3)
    out["Incremental TP / New Outlet"] = np.round(
        thr * realistic_mult, 2
    )
    return out


def insight_hidden_gems(matrix_df, top_n=10):
    """
    SKUs whose Throughput is above the 60th percentile while their
    Penetration sits below the 40th percentile — relaxed band so
    something always surfaces unless the matrix is tiny.
    """
    if matrix_df.empty:
        return pd.DataFrame()

    pen_thr = matrix_df["Penetration %"].quantile(0.40)
    thr_thr = matrix_df["Throughput"].quantile(0.60)

    gems = matrix_df[
        (matrix_df["Penetration %"] <= pen_thr) &
        (matrix_df["Throughput"] >= thr_thr)
    ].copy()

    gems["Opportunity Score"] = (
        gems["Throughput"] *
        (100 - gems["Penetration %"]) / 100
    ).round(1)

    return gems.sort_values(
        "Opportunity Score", ascending=False
    ).head(top_n)[
        [COL_BRAND, COL_SKU, "Penetration %",
         "Throughput", "Opportunity Score"]
    ]


def insight_distribution_gaps(matrix_df, top_n=10):
    """
    SKUs whose Penetration is above the 60th percentile while their
    Throughput sits below the 40th percentile.
    """
    if matrix_df.empty:
        return pd.DataFrame()

    pen_thr = matrix_df["Penetration %"].quantile(0.60)
    thr_thr = matrix_df["Throughput"].quantile(0.40)

    gaps = matrix_df[
        (matrix_df["Penetration %"] >= pen_thr) &
        (matrix_df["Throughput"] <= thr_thr)
    ].copy()

    return gaps.sort_values(
        "Throughput", ascending=True
    ).head(top_n)[
        [COL_BRAND, COL_SKU, "Penetration %", "Throughput"]
    ]


def compute_sku_months_active(source_df):
    """
    Per-SKU count of months (out of 15) where total volume across
    the dataset was positive. Filters out seasonal / one-shot
    promo SKUs that only appear in a few months.
    """

    if source_df.empty:
        return pd.Series(dtype=int)

    sku_monthly = (
        source_df
        .groupby(COL_SKU)[MONTH_COLS]
        .sum()
    )

    return (sku_monthly > 0).sum(axis=1).rename("months_active")


def insight_momentum(source_df, min_outlets=100, top_n=10):
    """
    Compare each SKU's L3M monthly throughput against its L15M
    monthly throughput  (L15M total ÷ outlets ÷ 15).

    `min_outlets` is the floor on outlets-in-L15 to qualify, used
    to suppress tail noise.
    """
    if source_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    grp = (
        source_df
        .groupby([COL_BRAND, COL_SKU], as_index=False)
        .agg(
            L3_total=("L3M", "sum"),
            L15_total=("L15M", "sum")
        )
    )

    # Outlet counts (separate to avoid index-based lambda hacks)
    L3_outlets = (
        source_df[source_df["L3M"] > 0]
        .groupby([COL_BRAND, COL_SKU])[COL_OUTLET]
        .nunique()
        .rename("outlets_L3")
        .reset_index()
    )

    L15_outlets = (
        source_df[source_df["L15M"] > 0]
        .groupby([COL_BRAND, COL_SKU])[COL_OUTLET]
        .nunique()
        .rename("outlets_L15")
        .reset_index()
    )

    grp = grp.merge(L3_outlets, on=[COL_BRAND, COL_SKU], how="left")
    grp = grp.merge(L15_outlets, on=[COL_BRAND, COL_SKU], how="left")
    grp[["outlets_L3", "outlets_L15"]] = (
        grp[["outlets_L3", "outlets_L15"]].fillna(0)
    )

    grp["L3 Monthly Throughput"] = (
        _safe_div(grp["L3_total"], grp["outlets_L3"]) / 3
    ).round(1)

    grp["L15 Monthly Throughput"] = (
        _safe_div(grp["L15_total"], grp["outlets_L15"]) / 15
    ).round(1)

    grp["Momentum (L3 / L15)"] = (
        _safe_div(
            grp["L3 Monthly Throughput"],
            grp["L15 Monthly Throughput"]
        )
    ).round(2)

    # Suppress the tail
    grp = grp[grp["outlets_L15"] >= min_outlets]

    cols = [
        COL_BRAND, COL_SKU,
        "L3 Monthly Throughput",
        "L15 Monthly Throughput",
        "Momentum (L3 / L15)"
    ]

    growers = grp[
        (grp["Momentum (L3 / L15)"] >= 1.1) &
        (grp["L15 Monthly Throughput"] > 0)
    ].sort_values(
        "Momentum (L3 / L15)", ascending=False
    ).head(top_n)[cols]

    decliners = grp[
        (grp["Momentum (L3 / L15)"] <= 0.95) &
        (grp["L15 Monthly Throughput"] > 0)
    ].sort_values(
        "Momentum (L3 / L15)", ascending=True
    ).head(top_n)[cols]

    return growers, decliners


def build_seasonal_trend(source_df, sku_list, mode="total"):
    """
    Long-format frame for plotting monthly trends of selected
    SKUs across the 15-month window.

    mode = 'total'      → sum of volume across outlets
    mode = 'per_outlet' → volume ÷ outlets that sold the SKU that
                          month (cleaner seasonality signal,
                          normalised for distribution growth).
    """

    if not sku_list or source_df.empty:
        return pd.DataFrame()

    sub = source_df[
        source_df[COL_SKU].astype(str).isin(sku_list)
    ]

    if sub.empty:
        return pd.DataFrame()

    totals = (
        sub.groupby(COL_SKU)[MONTH_COLS]
        .sum()
    )

    if mode == "per_outlet":

        # Active-outlet count per (SKU, month)
        bool_block = (sub[MONTH_COLS] > 0).copy()
        bool_block[COL_SKU] = sub[COL_SKU]
        actives = bool_block.groupby(COL_SKU)[MONTH_COLS].sum()

        values = (totals / actives.replace(0, np.nan)).fillna(0)

    else:
        values = totals

    long = (
        values.reset_index()
        .melt(
            id_vars=COL_SKU,
            var_name="Month",
            value_name="Volume"
        )
    )

    # Pretty month labels in correct chronological order
    long["Month"] = long["Month"].str.replace(
        "_V", "", regex=False
    )

    month_order = [m.replace("_V", "") for m in MONTH_COLS]

    long["Month"] = pd.Categorical(
        long["Month"],
        categories=month_order,
        ordered=True
    )

    long = long.sort_values([COL_SKU, "Month"])
    long["Volume"] = long["Volume"].round(1)

    return long


# Heatmap metric registry — defines how to compute and render
# each available metric.
# Color convention: GREEN = high / good, RED = low / bad.
# • Diverging metrics (lifts, deltas) use RdYlGn so the midpoint
#   is yellow, above-midpoint is green, below-midpoint is red.
# • Sequential metrics (absolute throughput, penetration) use a
#   white→green scale where deeper green = higher value.
HEATMAP_METRICS = {
    "total_volume": {
        "label":      "Total Pieces (units sold in period)",
        "colorscale": [
            [0.0, "#F8D7DA"],   # very low → light red
            [0.25, "#FFF3CD"],  # low → cream
            [0.5, "#D4EDDA"],   # mid → pale green
            [1.0, "#1B5E20"]    # high → deep green
        ],
        "midpoint":   None,
        "fmt":        ",.0f",
        "caption":    "Total units sold per (segment, SKU) over "
                      "the selected period. Higher = greener. "
                      "Use this when you want to see absolute "
                      "scale rather than per-outlet rates.",
        "colorbar":   "Units (period total)"
    },
    "lift": {
        "label":      "Throughput Lift (segment ÷ national)",
        "colorscale": "RdYlGn",
        "midpoint":   1.0,
        "fmt":        ".2f",
        "caption":    "Segment monthly throughput ÷ national monthly "
                      "throughput. Centred at 1.0; green = over-index, "
                      "red = under-index. Hides absolute pieces.",
        "colorbar":   "Lift (×)"
    },
    "throughput": {
        "label":      "Absolute Throughput (monthly per outlet)",
        "colorscale": [
            [0.0, "#F8D7DA"],   # very low → light red
            [0.25, "#FFF3CD"],  # low → cream
            [0.5, "#D4EDDA"],   # mid → pale green
            [1.0, "#1B5E20"]    # high → deep green
        ],
        "midpoint":   None,
        "fmt":        ".1f",
        "caption":    "Raw monthly throughput per active outlet in "
                      "the segment. Higher = greener. Use this when "
                      "Lift hides the fact that a SKU is already "
                      "very high-pieces.",
        "colorbar":   "Units / outlet / month"
    },
    "penetration": {
        "label":      "Penetration % (distribution density)",
        "colorscale": [
            [0.0, "#F8D7DA"],
            [0.25, "#FFF3CD"],
            [0.5, "#D4EDDA"],
            [1.0, "#1B5E20"]
        ],
        "midpoint":   None,
        "fmt":        ".0f",
        "caption":    "% of segment outlets stocking the SKU. "
                      "Higher = greener. Shows distribution density "
                      "independent of throughput.",
        "colorbar":   "Penetration %"
    },
    "penetration_lift": {
        "label":      "Penetration Lift (segment ÷ national)",
        "colorscale": "RdYlGn",
        "midpoint":   1.0,
        "fmt":        ".2f",
        "caption":    "Segment penetration ÷ national penetration. "
                      ">1 (green) = the SKU is *stocked* more widely "
                      "here than typical, regardless of how it sells.",
        "colorbar":   "Pen Lift (×)"
    },
    "throughput_delta": {
        "label":      "Throughput Δ — L3 ÷ L15 within segment",
        "colorscale": "RdYlGn",
        "midpoint":   1.0,
        "fmt":        ".2f",
        "caption":    "Within each segment, L3 monthly throughput "
                      "÷ L15 monthly throughput. Catches segment-"
                      "level momentum the national table misses. "
                      "Green = accelerating, red = decelerating.",
        "colorbar":   "L3 ÷ L15 (×)"
    },
    "penetration_impact": {
        "label":      "Penetration Impact (segment-level slope, historical)",
        "colorscale": "RdYlGn",
        "midpoint":   0.0,
        "fmt":        ".2f",
        "caption":    "Historical view, not a forecast. Per-SKU "
                      "slope `b` from y = a + b·x where y = "
                      "monthly throughput and x = penetration %, "
                      "fitted across rolling 3-month windows of "
                      "the 15-month history *within each segment*. "
                      "Positive (green) = in this segment, "
                      "throughput rose alongside wider distribution "
                      "in the past; negative (red) = throughput "
                      "thinned out as the SKU spread. Reads as "
                      "'this has happened before', not 'this will "
                      "happen next time'.",
        "colorbar":   "Pen Impact (slope)"
    },
    "opportunity_value": {
        "label":      "Incremental Sales Lift (₹/month from +1pp, historical)",
        "colorscale": [
            [0.0, "#F8D7DA"],   # very low → light red
            [0.25, "#FFF3CD"],  # low → cream
            [0.5, "#D4EDDA"],   # mid → pale green
            [1.0, "#1B5E20"]    # high → deep green
        ],
        "midpoint":   None,
        "fmt":        ",.0f",
        "caption":    "Historical view, not a forecast. Per "
                      "(segment, SKU), this is the *incremental* "
                      "₹/month that the historical penetration "
                      "slope `b` translates into when applied to "
                      "+1pp of penetration in that segment. "
                      "Formula: N_seg × b / 100 × avg MRP, where "
                      "N_seg = outlets in the segment, b = "
                      "segment-level penetration slope (Δ "
                      "throughput per +1pp pen), MRP = average "
                      "over the selected period's months. Reads "
                      "as 'what +1pp would have meant historically "
                      "at today's outlet count', not as a promise "
                      "for the next 30 stores you open. Requires "
                      "an uploaded MRP file; falls back to "
                      "incremental *pieces* otherwise.",
        "colorbar":   "Incr. Sales (₹/month per +1pp)"
    },
    # NEW METRIC — the practical "per new outlet" counterpart to
    # `opportunity_value`. Where the historical metrics above
    # describe what already happened, this one is built to answer
    # the field-team question: "if I take this SKU to ONE new
    # store in this segment, how much extra monthly volume should
    # I expect from THAT store?". The multiplier is strictly < 1
    # and varies by saturation, so saturated cells go red and
    # under-penetrated cells with healthy historical lift stay
    # the greenest.
    "realistic_incremental": {
        "label":      "Realistic Incremental TP per New Outlet (units/month)",
        "colorscale": [
            [0.0, "#F8D7DA"],
            [0.25, "#FFF3CD"],
            [0.5, "#D4EDDA"],
            [1.0, "#1B5E20"]
        ],
        "midpoint":   None,
        "fmt":        ".2f",
        "caption":    "Practical, forward-looking. For each "
                      "(segment, SKU), the expected extra units / "
                      "month from adding ONE new outlet in that "
                      "segment, given how saturated the SKU "
                      "already is there and how it has historically "
                      "behaved as it spread. Formula: T_seg × "
                      "Realistic Multiplier, where the multiplier "
                      f"= {MARGINAL_EFFICIENCY} × saturation_drag × "
                      "(1 + bounded tilt from historical slope), "
                      "capped at 0.95. saturation_drag goes from "
                      f"1.0 at P=0% down to {SAT_FLOOR:.2f} at "
                      "P=100% — even fully-saturated SKUs sell a "
                      "meaningful share of segment average in a "
                      "new outlet (the un-reached stores are "
                      "weaker than median, but still real stores "
                      "in that VC band). Multiply by the number "
                      "of new stores you actually open to size "
                      "the prize. Does NOT claim adding stores "
                      "raises the throughput of stores that "
                      "already stock the SKU.",
        "colorbar":   "Units / new outlet / month"
    }
}


def insight_heatmap(source_df, segment_col, period_col,
                    period_months, top_skus=25,
                    min_seg_outlets=50, metric="lift",
                    sub_segment_col=None):
    """
    SKU × Segment matrix ready to feed into a Plotly heatmap.
    Cell value depends on the chosen `metric` — see HEATMAP_METRICS.

    Returns a wide DataFrame (rows = SKUs, columns = segment
    values). Empty DataFrame if the slice is too thin.

    When `sub_segment_col` is provided (and different from
    `segment_col`), each segment column is subdivided by the
    sub-segment dimension. The returned DataFrame has a 2-level
    MultiIndex on columns: (segment_value, sub_segment_value).
    The min_seg_outlets gate is applied per (segment, sub_segment)
    cell so thin slices get dropped, not painted with noise.
    """

    if source_df.empty or segment_col not in source_df.columns:
        return pd.DataFrame()

    # Validate sub-segment column. If it's missing, blank, or
    # identical to the primary, silently fall back to single-level
    # behaviour so callers can pass `None` / `""` freely.
    use_sub = bool(
        sub_segment_col
        and sub_segment_col != segment_col
        and sub_segment_col in source_df.columns
    )

    src = source_df.copy()

    # Build a composite segment key so the existing metric logic
    # (7 branches below) keeps working unchanged. After the pivot
    # we split the composite back into a (segment, sub_segment)
    # MultiIndex on columns. Separator chosen to never appear in
    # real data.
    _COMBO_SEP = "║"
    _COMBO_COL = "__seg_combo__"
    if use_sub:
        src[_COMBO_COL] = (
            src[segment_col].astype(str)
            + _COMBO_SEP
            + src[sub_segment_col].astype(str)
        )
        effective_seg_col = _COMBO_COL
    else:
        effective_seg_col = segment_col

    # From here on, all groupby/pivot work uses the *effective*
    # segment column. The original primary `segment_col` is only
    # needed at the very end to split the composite key back into
    # a 2-level MultiIndex when a sub-segment is in play.
    original_segment_col = segment_col
    segment_col = effective_seg_col

    # For throughput_delta we need both L3 and L15 — pick L15 as
    # the universe so the SKU set is consistent.
    base_period = "L15M" if metric == "throughput_delta" else period_col

    active = src[src[base_period] > 0]
    if active.empty:
        return pd.DataFrame()

    # Top SKUs by national volume of the base period
    nat_rank = (
        active.groupby(COL_SKU)
        .agg(
            nat_vol=(base_period, "sum"),
            nat_outlets=(COL_OUTLET, "nunique")
        )
    )
    # `keep_skus` is the canonical row order for the final pivot —
    # by descending national volume. We reindex the pivot to this
    # list at the end so a SKU that lost all its (seg, sub) cells
    # to the sparsity gate still shows up as a row (blank/NaN
    # cells), instead of silently disappearing.
    #
    # Build the universe from ALL SKUs in `src` (not just `active`)
    # so SKUs with zero base-period volume still appear as rows
    # in the heatmap — they paint as fully-NaN rows, which is the
    # honest representation ("we tracked this SKU, it has no
    # measurable throughput in this period"). Without this, any
    # SKU with literally zero sales in the base period was
    # silently dropped from `keep_skus`, then never restored by
    # the reindex at the bottom of the function — leading to
    # fewer rows than the top_skus cap when some SKUs are
    # dormant. Order: positive-volume SKUs first (by descending
    # volume), then zero-volume SKUs at the bottom in stable
    # alphabetical order.
    positive_skus = list(
        nat_rank.sort_values("nat_vol", ascending=False).index
    )
    # Pull the *full* SKU universe straight from `src` (post-filter to
    # the chosen segment/period). Preserve original dtype so downstream
    # `.isin(keep_skus)` matches without coercion surprises. Compare
    # via a string-keyed set just to detect "already counted" — the
    # values returned in `keep_skus` are the original-dtype values.
    positive_set = {str(s) for s in positive_skus}
    all_skus_in_src = (
        src[COL_SKU].dropna().drop_duplicates().tolist()
    )
    zero_vol_skus = sorted(
        [s for s in all_skus_in_src if str(s) not in positive_set],
        key=str,
    )
    keep_skus = (positive_skus + zero_vol_skus)[:top_skus]

    src = src[src[COL_SKU].isin(keep_skus)]
    active = active[active[COL_SKU].isin(keep_skus)]

    # When a secondary breakdown is active, each primary cell gets
    # subdivided into ~`n_sub` sub-cells. Applying the original
    # `min_seg_outlets` gate per sub-cell would be unfairly harsh —
    # it implicitly demands `min_seg_outlets × n_sub` outlets per
    # primary segment, far above the single-level requirement. Scale
    # the gate down proportionally, with a hard floor so we don't
    # paint pure-noise cells built on 2–3 outlets.
    if use_sub:
        n_sub = max(
            src[sub_segment_col].astype(str).nunique(), 1
        )
        min_seg_outlets = max(
            int(min_seg_outlets / n_sub),
            10
        )

    # Total outlets per segment value (denominator for penetration)
    seg_total_outlets = (
        src.groupby(segment_col)[COL_OUTLET]
        .nunique()
        .rename("seg_total_outlets")
    )

    # ----- Throughput Lift -----
    if metric == "lift":

        nat_thr = pd.Series(
            (
                _safe_div(nat_rank["nat_vol"], nat_rank["nat_outlets"])
                / period_months
            ),
            index=nat_rank.index, name="nat_thr"
        )

        seg = (
            active.groupby([segment_col, COL_SKU])
            .agg(
                seg_vol=(period_col, "sum"),
                seg_outlets=(COL_OUTLET, "nunique")
            )
            .reset_index()
        )
        seg = seg[seg["seg_outlets"] >= min_seg_outlets]
        if seg.empty:
            return pd.DataFrame()

        seg["seg_thr"] = (
            _safe_div(seg["seg_vol"], seg["seg_outlets"])
            / period_months
        )
        seg = seg.merge(
            nat_thr.reset_index(), on=COL_SKU, how="left"
        )
        seg["value"] = _safe_div(seg["seg_thr"], seg["nat_thr"])

    # ----- Absolute Throughput -----
    elif metric == "throughput":

        seg = (
            active.groupby([segment_col, COL_SKU])
            .agg(
                seg_vol=(period_col, "sum"),
                seg_outlets=(COL_OUTLET, "nunique")
            )
            .reset_index()
        )
        seg = seg[seg["seg_outlets"] >= min_seg_outlets]
        if seg.empty:
            return pd.DataFrame()

        seg["value"] = (
            _safe_div(seg["seg_vol"], seg["seg_outlets"])
            / period_months
        )

    # ----- Total Volume (raw units sold per segment, SKU) -----
    elif metric == "total_volume":

        seg = (
            active.groupby([segment_col, COL_SKU])
            .agg(
                seg_vol=(period_col, "sum"),
                seg_outlets=(COL_OUTLET, "nunique")
            )
            .reset_index()
        )
        seg = seg[seg["seg_outlets"] >= min_seg_outlets]
        if seg.empty:
            return pd.DataFrame()

        seg["value"] = seg["seg_vol"].round(0)

    # ----- Penetration % -----
    elif metric == "penetration":

        seg = (
            active.groupby([segment_col, COL_SKU])
            .agg(active_outlets=(COL_OUTLET, "nunique"))
            .reset_index()
        )
        seg = seg.merge(
            seg_total_outlets.reset_index(),
            on=segment_col, how="left"
        )
        seg = seg[seg["active_outlets"] >= min_seg_outlets]
        if seg.empty:
            return pd.DataFrame()

        seg["value"] = (
            _safe_div(seg["active_outlets"],
                      seg["seg_total_outlets"]) * 100
        )

    # ----- Penetration Lift -----
    elif metric == "penetration_lift":

        total_outlets = src[COL_OUTLET].nunique()
        nat_active = (
            active.groupby(COL_SKU)[COL_OUTLET]
            .nunique()
            .rename("nat_active")
        )
        nat_pen = (
            (nat_active / total_outlets * 100).rename("nat_pen")
        )

        seg = (
            active.groupby([segment_col, COL_SKU])
            .agg(active_outlets=(COL_OUTLET, "nunique"))
            .reset_index()
        )
        seg = seg.merge(
            seg_total_outlets.reset_index(),
            on=segment_col, how="left"
        )
        seg["seg_pen"] = (
            _safe_div(seg["active_outlets"],
                      seg["seg_total_outlets"]) * 100
        )
        seg = seg[seg["active_outlets"] >= min_seg_outlets]
        seg = seg.merge(nat_pen.reset_index(), on=COL_SKU, how="left")
        if seg.empty:
            return pd.DataFrame()

        seg["value"] = _safe_div(seg["seg_pen"], seg["nat_pen"])

    # ----- Throughput Δ (L3 ÷ L15) within segment -----
    elif metric == "throughput_delta":

        l3 = (
            src[src["L3M"] > 0]
            .groupby([segment_col, COL_SKU])
            .agg(
                l3_vol=("L3M", "sum"),
                l3_outlets=(COL_OUTLET, "nunique")
            )
            .reset_index()
        )
        l15 = (
            src[src["L15M"] > 0]
            .groupby([segment_col, COL_SKU])
            .agg(
                l15_vol=("L15M", "sum"),
                l15_outlets=(COL_OUTLET, "nunique")
            )
            .reset_index()
        )
        seg = l15.merge(
            l3, on=[segment_col, COL_SKU], how="left"
        )
        seg[["l3_vol", "l3_outlets"]] = (
            seg[["l3_vol", "l3_outlets"]].fillna(0)
        )
        seg = seg[seg["l15_outlets"] >= min_seg_outlets]
        if seg.empty:
            return pd.DataFrame()

        seg["l3_thr"]  = _safe_div(seg["l3_vol"],  seg["l3_outlets"])  / 3
        seg["l15_thr"] = _safe_div(seg["l15_vol"], seg["l15_outlets"]) / 15
        seg["value"]   = _safe_div(seg["l3_thr"], seg["l15_thr"])

    # ----- Penetration Impact (segment-level slope) -----
    # For each (segment value, SKU), fit y = a + b·x where
    # y = monthly throughput in a 3-month window, x = penetration %
    # in that window (share of segment outlets selling the SKU).
    # Rolling 3-month windows across the 15-month history → up to
    # 13 observations per (segment, SKU). Return slope `b`.
    elif metric == "penetration_impact":

        months_present = [
            m for m in MONTH_COLS if m in src.columns
        ]
        if len(months_present) < 3:
            return pd.DataFrame()

        windows = [
            months_present[i: i + 3]
            for i in range(len(months_present) - 2)
        ]

        # Segment outlet base (denominator for penetration). Use the
        # full src (not just `active`) so the denominator is stable
        # across windows.
        seg_total = (
            src.groupby(segment_col)[COL_OUTLET]
            .nunique()
            .rename("seg_total_outlets")
            .reset_index()
        )

        # Filter to segments meeting the min_seg_outlets gate
        seg_total = seg_total[
            seg_total["seg_total_outlets"] >= min_seg_outlets
        ]
        if seg_total.empty:
            return pd.DataFrame()
        valid_segs = set(seg_total[segment_col].astype(str))

        # Restrict src to those segments
        seg_src_local = src[
            src[segment_col].astype(str).isin(valid_segs)
        ]

        # Build (seg, sku, window) → (pen_pct, throughput) obs
        rows = []
        for w_idx, w_cols in enumerate(windows):
            w_vol = seg_src_local[w_cols].sum(axis=1)
            tmp = pd.DataFrame({
                segment_col:  seg_src_local[segment_col].values,
                COL_SKU:      seg_src_local[COL_SKU].values,
                COL_OUTLET:   seg_src_local[COL_OUTLET].values,
                "w_vol":      w_vol.values,
            })
            active_w = tmp[tmp["w_vol"] > 0]
            if active_w.empty:
                continue
            grp = (
                active_w.groupby(
                    [segment_col, COL_SKU], as_index=False
                ).agg(
                    vol=("w_vol", "sum"),
                    outlets=(COL_OUTLET, "nunique"),
                )
            )
            grp = grp.merge(seg_total, on=segment_col, how="left")
            grp["pen_pct"] = (
                grp["outlets"] / grp["seg_total_outlets"] * 100
            )
            grp["throughput"] = grp["vol"] / grp["outlets"] / 3.0
            grp["window"] = w_idx
            rows.append(grp[[
                segment_col, COL_SKU,
                "pen_pct", "throughput", "window"
            ]])

        if not rows:
            return pd.DataFrame()

        obs = pd.concat(rows, ignore_index=True)

        # Per (segment, SKU) OLS slope
        def _slope(g):
            x = g["pen_pct"].to_numpy(dtype=float)
            y = g["throughput"].to_numpy(dtype=float)
            if len(x) < 2:
                return np.nan
            x_var = x.var()
            if x_var < 1e-12:
                return np.nan
            cov_xy = np.cov(x, y, ddof=0)[0, 1]
            return cov_xy / x_var

        seg = (
            obs.groupby([segment_col, COL_SKU])
            .apply(_slope, include_groups=False)
            .reset_index()
            .rename(columns={0: "value"})
        )
        # Older pandas returns the slope column unnamed
        if "value" not in seg.columns:
            last_col = seg.columns[-1]
            seg = seg.rename(columns={last_col: "value"})

        seg = seg.dropna(subset=["value"])
        if seg.empty:
            return pd.DataFrame()

    # ----- Incremental Sales Lift (₹/month from +1pp) -----
    # Per (segment, SKU) *delta* from lifting penetration by +1pp:
    #     Incr Volume = N_seg × b / 100
    #     Incr Value  = Incr Volume × avg MRP for the SKU
    # where the per-segment slope `b` is fitted on rolling 3-month
    # windows of penetration % vs monthly throughput (same engine
    # as the penetration_impact metric). This is the *additional*
    # volume/revenue, not the post-lift total. Falls back to
    # incremental *volume* (units/month) when no MRP file is
    # available.
    elif metric == "opportunity_value":

        months_present = [
            m for m in MONTH_COLS if m in src.columns
        ]
        if len(months_present) < 3:
            return pd.DataFrame()

        windows = [
            months_present[i: i + 3]
            for i in range(len(months_present) - 2)
        ]

        # Segment outlet base — N_seg (and slope denominator)
        seg_total = (
            src.groupby(segment_col)[COL_OUTLET]
            .nunique()
            .rename("seg_total_outlets")
            .reset_index()
        )
        seg_total = seg_total[
            seg_total["seg_total_outlets"] >= min_seg_outlets
        ]
        if seg_total.empty:
            return pd.DataFrame()
        valid_segs = set(seg_total[segment_col].astype(str))

        seg_src_local = src[
            src[segment_col].astype(str).isin(valid_segs)
        ]

        # Build per-(seg, sku, window) (pen%, throughput) observations
        rows = []
        for w_idx, w_cols in enumerate(windows):
            w_vol = seg_src_local[w_cols].sum(axis=1)
            tmp = pd.DataFrame({
                segment_col:  seg_src_local[segment_col].values,
                COL_SKU:      seg_src_local[COL_SKU].values,
                COL_OUTLET:   seg_src_local[COL_OUTLET].values,
                "w_vol":      w_vol.values,
            })
            active_w = tmp[tmp["w_vol"] > 0]
            if active_w.empty:
                continue
            grp = (
                active_w.groupby(
                    [segment_col, COL_SKU], as_index=False
                ).agg(
                    vol=("w_vol", "sum"),
                    outlets=(COL_OUTLET, "nunique"),
                )
            )
            grp = grp.merge(seg_total, on=segment_col, how="left")
            grp["pen_pct"] = (
                grp["outlets"] / grp["seg_total_outlets"] * 100
            )
            grp["throughput"] = grp["vol"] / grp["outlets"] / 3.0
            grp["window"] = w_idx
            rows.append(grp[[
                segment_col, COL_SKU,
                "pen_pct", "throughput", "window"
            ]])

        if not rows:
            return pd.DataFrame()

        obs = pd.concat(rows, ignore_index=True)

        # Per-(segment, SKU) OLS slope of throughput on pen_pct.
        def _slope_b(g):
            x = g["pen_pct"].to_numpy(dtype=float)
            y = g["throughput"].to_numpy(dtype=float)
            if len(x) < 2:
                return np.nan
            x_var = x.var()
            if x_var < 1e-12:
                return np.nan
            cov_xy = np.cov(x, y, ddof=0)[0, 1]
            return cov_xy / x_var

        slope_df = (
            obs.groupby([segment_col, COL_SKU])
            .apply(_slope_b, include_groups=False)
            .reset_index()
            .rename(columns={0: "slope_b"})
        )
        if "slope_b" not in slope_df.columns:
            last_col = slope_df.columns[-1]
            slope_df = slope_df.rename(
                columns={last_col: "slope_b"}
            )
        slope_df = slope_df.dropna(subset=["slope_b"])
        if slope_df.empty:
            return pd.DataFrame()

        # Current-period per-(segment, SKU) penetration % and
        # throughput from the selected period (period_col, e.g.
        # L3M / L6M / L15M).
        cur = (
            active.groupby([segment_col, COL_SKU])
            .agg(
                cur_vol=(period_col, "sum"),
                cur_outlets=(COL_OUTLET, "nunique"),
            )
            .reset_index()
        )
        cur = cur.merge(seg_total, on=segment_col, how="left")
        cur = cur.dropna(subset=["seg_total_outlets"])
        cur = cur[cur["cur_outlets"] >= min_seg_outlets]
        if cur.empty:
            return pd.DataFrame()

        cur["P"] = (
            cur["cur_outlets"] / cur["seg_total_outlets"] * 100
        )
        cur["T"] = (
            _safe_div(cur["cur_vol"], cur["cur_outlets"])
            / period_months
        )

        seg = cur.merge(
            slope_df, on=[segment_col, COL_SKU], how="left"
        )
        # SKUs without a fitted slope (too few obs) get b=0, which
        # collapses Opportunity to pure throughput × N_seg / 100.
        # That's the right "no-slope-evidence" prior.
        seg["slope_b"] = seg["slope_b"].fillna(0.0)

        # Incremental Opportunity Volume (units/month) per
        # (segment, SKU) from lifting penetration by +1pp.
        # Model: throughput = a + b · pen_pct, so the marginal
        # monthly volume from one extra pp of penetration is
        # N_seg × b / 100. This is the *delta*, not the post-
        # lift total — i.e. the additional units you'd sell,
        # which is what "incremental" should mean.
        seg["opp_volume"] = (
            seg["seg_total_outlets"] * seg["slope_b"] / 100.0
        )
        # Floor at zero — a negative slope means "thinning out
        # as it spreads", i.e. no ₹ prize from pushing further.
        # Render that as zero opportunity rather than a negative
        # bill of materials.
        seg["opp_volume"] = seg["opp_volume"].clip(lower=0)

        # Look up average MRP per SKU across the selected period's
        # months. Falls back to opportunity *volume* if no MRP file
        # is loaded (so the metric still renders something useful).
        try:
            mrp_table = _mrp_lookup  # module-level, set in sidebar
        except NameError:
            mrp_table = None

        if mrp_table is not None and not mrp_table.empty:
            # Pick MRP columns aligned to the selected period.
            # period_months months at the tail of the 15-month
            # window — same recency convention as L3/L6/L15 use
            # in the volume table.
            n_months = max(int(period_months), 1)
            mrp_cols_sel = [
                c for c in MRP_MONTH_COLS[-n_months:]
                if c in mrp_table.columns
            ]
            if mrp_cols_sel:
                avg_mrp = (
                    mrp_table[mrp_cols_sel]
                    .mean(axis=1)
                    .rename("avg_mrp")
                )
                sku_keys = (
                    seg[COL_SKU].astype(str).str.strip().str.lower()
                )
                seg["avg_mrp"] = (
                    avg_mrp.reindex(sku_keys.values)
                    .fillna(0)
                    .values
                )
                seg["value"] = (
                    seg["opp_volume"] * seg["avg_mrp"]
                ).round(0)
            else:
                # MRP file loaded but no usable month columns →
                # fall back to opportunity volume.
                seg["value"] = seg["opp_volume"].round(0)
        else:
            # No MRP file — render opportunity *volume* so the
            # tile still tells a story instead of being blank.
            seg["value"] = seg["opp_volume"].round(0)

    # ----- Realistic Incremental Throughput per New Outlet -----
    # Per (segment, SKU), the practical "if I add one new outlet
    # in this segment, what extra monthly volume should I expect
    # from THAT outlet?". Mirrors the Overview-tab metric of the
    # same name, but computed *per segment* so saturation is
    # measured locally (a SKU may be saturated in one VC band and
    # still have plenty of headroom in another).
    #
    # Formula v15 (matching compute_realistic_incremental_per_outlet):
    #     T_seg       = monthly throughput in this (segment, SKU)
    #     P_seg       = penetration % in this (segment, SKU)
    #     b_seg       = historical slope of T on P in this segment
    #     sat_drag    = SAT_FLOOR + (1 − SAT_FLOOR) × (1 − P_seg/100)
    #                                              ∈ [SAT_FLOOR, 1.0]
    #     tilt_raw    = b_seg / max(T_seg, 0.5)
    #     tilt        = 1 + 0.2 · tanh(5 · tilt_raw)  ∈ (0.8, 1.2)
    #     mult        = clip(MARGINAL_EFFICIENCY · sat_drag · tilt,
    #                        0, 0.95)
    #     value       = T_seg × mult
    #
    # Why no raw `headroom` term any more: the old formula
    # multiplied T directly by (1 − P/100), which collapsed the
    # per-outlet number to ~0 for highly-penetrated SKUs even
    # though the segment was still selling at T_seg/outlet on
    # average. Headroom rightly bounds *how many* new outlets
    # exist (a downstream N×… calculation), but should NOT crush
    # *what each one sells*. The saturation_drag term keeps the
    # "marginal stores are weaker than the median" intuition
    # without crashing to zero.
    #
    # When b_seg is missing (too few obs) → tilt = 1.0 (neutral).
    # When T_seg = 0 → tilt = 1.0 (b/T undefined) and value = 0.
    elif metric == "realistic_incremental":

        months_present = [
            m for m in MONTH_COLS if m in src.columns
        ]
        if len(months_present) < 3:
            return pd.DataFrame()

        windows = [
            months_present[i: i + 3]
            for i in range(len(months_present) - 2)
        ]

        # Segment outlet base (denominator for penetration).
        seg_total = (
            src.groupby(segment_col)[COL_OUTLET]
            .nunique()
            .rename("seg_total_outlets")
            .reset_index()
        )
        seg_total = seg_total[
            seg_total["seg_total_outlets"] >= min_seg_outlets
        ]
        if seg_total.empty:
            return pd.DataFrame()
        valid_segs = set(seg_total[segment_col].astype(str))

        seg_src_local = src[
            src[segment_col].astype(str).isin(valid_segs)
        ]

        # Build per-(seg, sku, window) (pen%, throughput) observations
        # — same engine as opportunity_value. Needed for slope b_seg.
        rows = []
        for w_idx, w_cols in enumerate(windows):
            w_vol = seg_src_local[w_cols].sum(axis=1)
            tmp = pd.DataFrame({
                segment_col:  seg_src_local[segment_col].values,
                COL_SKU:      seg_src_local[COL_SKU].values,
                COL_OUTLET:   seg_src_local[COL_OUTLET].values,
                "w_vol":      w_vol.values,
            })
            active_w = tmp[tmp["w_vol"] > 0]
            if active_w.empty:
                continue
            grp = (
                active_w.groupby(
                    [segment_col, COL_SKU], as_index=False
                ).agg(
                    vol=("w_vol", "sum"),
                    outlets=(COL_OUTLET, "nunique"),
                )
            )
            grp = grp.merge(seg_total, on=segment_col, how="left")
            grp["pen_pct"] = (
                grp["outlets"] / grp["seg_total_outlets"] * 100
            )
            grp["throughput"] = grp["vol"] / grp["outlets"] / 3.0
            grp["window"] = w_idx
            rows.append(grp[[
                segment_col, COL_SKU,
                "pen_pct", "throughput", "window"
            ]])

        if not rows:
            return pd.DataFrame()

        obs = pd.concat(rows, ignore_index=True)

        def _slope_b(g):
            x = g["pen_pct"].to_numpy(dtype=float)
            y = g["throughput"].to_numpy(dtype=float)
            if len(x) < 2:
                return np.nan
            x_var = x.var()
            if x_var < 1e-12:
                return np.nan
            cov_xy = np.cov(x, y, ddof=0)[0, 1]
            return cov_xy / x_var

        slope_df = (
            obs.groupby([segment_col, COL_SKU])
            .apply(_slope_b, include_groups=False)
            .reset_index()
            .rename(columns={0: "slope_b"})
        )
        if "slope_b" not in slope_df.columns:
            last_col = slope_df.columns[-1]
            slope_df = slope_df.rename(
                columns={last_col: "slope_b"}
            )

        # Current-period (segment, SKU) penetration % and throughput
        cur = (
            active.groupby([segment_col, COL_SKU])
            .agg(
                cur_vol=(period_col, "sum"),
                cur_outlets=(COL_OUTLET, "nunique"),
            )
            .reset_index()
        )
        cur = cur.merge(seg_total, on=segment_col, how="left")
        cur = cur.dropna(subset=["seg_total_outlets"])
        cur = cur[cur["cur_outlets"] >= min_seg_outlets]
        if cur.empty:
            return pd.DataFrame()

        cur["P"] = (
            cur["cur_outlets"] / cur["seg_total_outlets"] * 100
        )
        cur["T"] = (
            _safe_div(cur["cur_vol"], cur["cur_outlets"])
            / period_months
        )

        seg = cur.merge(
            slope_df, on=[segment_col, COL_SKU], how="left"
        )
        seg["slope_b"] = seg["slope_b"].fillna(0.0)

        # Build the multiplier exactly as in
        # compute_realistic_incremental_per_outlet — same shape,
        # same constants, same clipping. Keeps the Overview-tab
        # and the per-segment heatmap reading on a comparable scale.
        #
        # v15: decoupled headroom — saturation drag goes from 1.0
        # at P=0 to SAT_FLOOR at P=100 (linearly), instead of
        # crushing the per-outlet number to ~0 when the SKU is
        # highly penetrated. The few outlets still un-reached for
        # a near-ubiquitous SKU are weaker than average, but they
        # are real stores in the same VC band — they will sell a
        # meaningful share of segment throughput.
        seg["saturation_drag"] = SAT_FLOOR + (1.0 - SAT_FLOOR) * (
            (100.0 - seg["P"].clip(lower=0, upper=100)) / 100.0
        )
        # Smooth tanh tilt with stabilised denominator — preserves
        # SKU-to-SKU differentiation that hard clipping would
        # flatten, and keeps low-throughput cells stable.
        tilt_raw = (
            seg["slope_b"].astype(float)
            / np.maximum(seg["T"].astype(float), 0.5)
        )
        seg["tilt"] = 1.0 + 0.2 * np.tanh(5.0 * tilt_raw)
        seg["mult"] = (
            MARGINAL_EFFICIENCY * seg["saturation_drag"] * seg["tilt"]
        ).clip(lower=0.0, upper=0.95)
        seg["value"] = (
            seg["T"].clip(lower=0) * seg["mult"]
        ).round(2)

    else:
        return pd.DataFrame()

    pivot = seg.pivot(
        index=COL_SKU,
        columns=segment_col,
        values="value"
    ).round(2)

    # Guarantee every SKU in the top-N national-volume list shows
    # up as a row, even if every single one of its (segment, sub)
    # cells got dropped by the sparsity gate. Without this reindex,
    # a thinly-distributed SKU silently disappears from the heatmap
    # — and once the secondary breakdown is on (which multiplies
    # the gate by 1/n_sub but still loses some), several rows can
    # vanish, making the chart look like fewer SKUs than the user
    # asked for. Order matches `keep_skus` (descending national
    # volume); the row-greenness sort downstream will reorder them
    # for display, but missing SKUs survive as NaN rows either way.
    pivot = pivot.reindex(keep_skus)

    if use_sub:
        # Split the composite "primary║secondary" key back into a
        # 2-level MultiIndex. Order primary by canonical business
        # order, and secondary by canonical business order *within*
        # each primary group. NaN-only columns (no data after the
        # min_seg_outlets gate) are dropped so empty sub-cells
        # don't clutter the heatmap.
        split = [str(c).split(_COMBO_SEP, 1) for c in pivot.columns]
        primary_vals = [s[0] if len(s) > 0 else "" for s in split]
        sub_vals     = [s[1] if len(s) > 1 else "" for s in split]
        pivot.columns = pd.MultiIndex.from_arrays(
            [primary_vals, sub_vals],
            names=[original_segment_col, sub_segment_col]
        )
        # Drop fully-empty columns (no SKU has any data in that
        # sub-cell). Rows are kept regardless — that's the whole
        # point of the reindex above.
        pivot = pivot.dropna(axis=1, how="all")
        if pivot.empty or pivot.shape[1] == 0:
            return pivot
        # Build canonical ordering for each level
        primary_order = order_segment_values(
            original_segment_col,
            list(dict.fromkeys(pivot.columns.get_level_values(0)))
        )
        sub_order = order_segment_values(
            sub_segment_col,
            list(dict.fromkeys(pivot.columns.get_level_values(1)))
        )
        # Sort columns: primary first (outer), sub-segment within
        sub_rank = {v: i for i, v in enumerate(sub_order)}
        primary_rank = {v: i for i, v in enumerate(primary_order)}
        col_keys = sorted(
            pivot.columns,
            key=lambda t: (
                primary_rank.get(t[0], len(primary_rank)),
                sub_rank.get(t[1], len(sub_rank)),
            )
        )
        return pivot[col_keys]

    # Apply canonical column ordering for VC_CAT / Region / Region_Cat
    # so the heatmap reads left-to-right in business order
    # (e.g. <20L → >500L for VC_CAT; Ultra Premium → Aspirational
    # for Region). Other segment dimensions keep natural order.
    ordered_cols = order_segment_values(
        segment_col, list(pivot.columns)
    )
    return pivot[ordered_cols]


def insight_distribution_health(source_df, min_outlets=100, top_n=10):
    """
    Penetration-side mirror of insight_momentum.

    For each SKU we compute:
        Active Rate = (outlets with L3 sales) / (outlets with L15 sales)

    Active Rate measures *distribution health*:
      • close to 1.00 → almost every outlet that ever stocked
        the SKU is still selling it ⇒ healthy, embedded SKU.
      • below 0.85   → meaningful share of L15 buyers have stopped
        stocking ⇒ distribution is eroding even if throughput
        looks fine in surviving outlets.

    Returns two frames:
      • stable    — Active Rate ≥ 0.98  (your distribution-locked-in SKUs)
      • eroding   — Active Rate ≤ 0.85  (distribution-loss watchlist)

    Pair this with momentum:
        eroding + L3 throughput up   →  concentrating to fewer outlets
        eroding + L3 throughput down →  delist queue
        stable  + L3 throughput up   →  strongest growth signal
    """

    if source_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    L3_outlets = (
        source_df[source_df["L3M"] > 0]
        .groupby([COL_BRAND, COL_SKU])[COL_OUTLET]
        .nunique()
        .rename("outlets_L3")
    )

    L15_outlets = (
        source_df[source_df["L15M"] > 0]
        .groupby([COL_BRAND, COL_SKU])[COL_OUTLET]
        .nunique()
        .rename("outlets_L15")
    )

    grp = pd.concat(
        [L3_outlets, L15_outlets], axis=1
    ).fillna(0).reset_index()

    grp = grp[grp["outlets_L15"] >= min_outlets]

    if grp.empty:
        return pd.DataFrame(), pd.DataFrame()

    grp["Active Rate"] = (
        grp["outlets_L3"] / grp["outlets_L15"]
    ).round(2)

    grp["Outlets Lost"] = (
        grp["outlets_L15"] - grp["outlets_L3"]
    ).astype(int)

    grp["outlets_L15"] = grp["outlets_L15"].astype(int)
    grp["outlets_L3"]  = grp["outlets_L3"].astype(int)

    cols = [
        COL_BRAND, COL_SKU,
        "outlets_L15", "outlets_L3",
        "Outlets Lost", "Active Rate"
    ]

    eroding = grp[grp["Active Rate"] <= 0.85].sort_values(
        "Active Rate", ascending=True
    ).head(top_n)[cols]

    stable = grp[grp["Active Rate"] >= 0.98].sort_values(
        "outlets_L15", ascending=False
    ).head(top_n)[cols]

    return stable, eroding


def insight_segment_overindex(source_df, segment_col,
                              period_col, period_months, top_n=12):
    """
    For each (segment value, SKU) pair, compute monthly throughput
    inside the segment vs national. Return the rows where the
    segment ratio is highest — i.e. SKUs that disproportionately
    win in particular cooler-size / region / channel buckets.

    This is exactly the "in rich-area outlets these SKUs over-index"
    pattern.
    """
    if source_df.empty or segment_col not in source_df.columns:
        return pd.DataFrame()

    src = source_df[source_df[period_col] > 0]

    if src.empty:
        return pd.DataFrame()

    # National throughput per SKU (monthly)
    nat = (
        src.groupby([COL_BRAND, COL_SKU])
        .agg(
            nat_vol=(period_col, "sum"),
            nat_outlets=(COL_OUTLET, "nunique")
        )
    )
    nat["Nat Throughput"] = (
        _safe_div(nat["nat_vol"], nat["nat_outlets"]) / period_months
    ).round(1)

    # Segment throughput
    seg = (
        src.groupby([segment_col, COL_BRAND, COL_SKU])
        .agg(
            seg_vol=(period_col, "sum"),
            seg_outlets=(COL_OUTLET, "nunique")
        )
        .reset_index()
    )
    seg = seg[seg["seg_outlets"] >= 25]

    if seg.empty:
        return pd.DataFrame()

    seg["Seg Throughput"] = (
        _safe_div(seg["seg_vol"], seg["seg_outlets"]) / period_months
    ).round(1)

    seg = seg.merge(
        nat[["Nat Throughput"]].reset_index(),
        on=[COL_BRAND, COL_SKU],
        how="left"
    )

    seg["Lift (×)"] = (
        _safe_div(seg["Seg Throughput"], seg["Nat Throughput"])
    ).round(2)

    seg = seg[seg["Lift (×)"] >= 1.5]

    return seg.sort_values(
        "Lift (×)", ascending=False
    ).head(top_n)[
        [
            segment_col, COL_BRAND, COL_SKU,
            "seg_outlets", "Seg Throughput",
            "Nat Throughput", "Lift (×)"
        ]
    ].rename(columns={"seg_outlets": "Outlets in Seg"})


def insight_segment_underindex(source_df, segment_col,
                               period_col, period_months,
                               top_n=12, max_lift=0.7,
                               min_seg_outlets=25):
    """
    Inverse of segment_overindex: (segment, SKU) pairs where the
    SKU disproportionately UNDER-performs vs national.
    Lift ≤ max_lift is shown — these are weak-fit combinations
    where extra distribution effort is not paying off.
    """
    if source_df.empty or segment_col not in source_df.columns:
        return pd.DataFrame()

    src = source_df[source_df[period_col] > 0]

    if src.empty:
        return pd.DataFrame()

    nat = (
        src.groupby([COL_BRAND, COL_SKU])
        .agg(
            nat_vol=(period_col, "sum"),
            nat_outlets=(COL_OUTLET, "nunique")
        )
    )
    nat["Nat Throughput"] = (
        _safe_div(nat["nat_vol"], nat["nat_outlets"]) / period_months
    ).round(1)

    seg = (
        src.groupby([segment_col, COL_BRAND, COL_SKU])
        .agg(
            seg_vol=(period_col, "sum"),
            seg_outlets=(COL_OUTLET, "nunique")
        )
        .reset_index()
    )
    seg = seg[seg["seg_outlets"] >= min_seg_outlets]

    if seg.empty:
        return pd.DataFrame()

    seg["Seg Throughput"] = (
        _safe_div(seg["seg_vol"], seg["seg_outlets"]) / period_months
    ).round(1)

    seg = seg.merge(
        nat[["Nat Throughput"]].reset_index(),
        on=[COL_BRAND, COL_SKU],
        how="left"
    )

    seg["Lift (×)"] = (
        _safe_div(seg["Seg Throughput"], seg["Nat Throughput"])
    ).round(2)

    # Underperformers — guard against /0 false positives
    seg = seg[
        (seg["Lift (×)"] <= max_lift) &
        (seg["Nat Throughput"] > 0)
    ]

    return seg.sort_values(
        "Lift (×)", ascending=True
    ).head(top_n)[
        [
            segment_col, COL_BRAND, COL_SKU,
            "seg_outlets", "Seg Throughput",
            "Nat Throughput", "Lift (×)"
        ]
    ].rename(columns={"seg_outlets": "Outlets in Seg"})



# =========================================================
# ZSM REPORT  (downloadable PDF)
# =========================================================
# Uses reportlab + kaleido. Install once with:
#     pip install reportlab kaleido
# Falls back to HTML if reportlab is missing; skips embedded
# heatmap images if kaleido is missing.
# =========================================================

def _tier_sort_key(tier_str):
    """
    Map a tier string to a sortable integer. The canonical order
    is "best/largest SKU tier first":
        Tier 1 (best) → Tier 2 → Tier 3 → Tier 4 → Tier 5 → unknown

    Supports three label flavours and produces the same ordering
    for all of them:
        1. Legacy numeric: "Tier 1", "T1", "1", "tier 2", ...
        2. New explicit value bands (the post-load canonical
           form set by load_data):
             "Greater than 200" → Tier 1
             "120-150"          → Tier 2
             "80-110"           → Tier 3
             "35-60"            → Tier 4
             "Less than 35"     → Tier 5
        3. Anything else → falls back to "first integer in the
           string" with a sentinel for missing values, so unknown
           tier labels still sort sensibly to the bottom.
    """
    if tier_str is None:
        return (10_000, "")
    s = str(tier_str).strip()
    if not s or s.lower() in ("nan", "none"):
        return (10_000, "")

    # ----- New explicit band labels -----
    # Use a normalised key (lowercase, whitespace-stripped) so
    # variants like "greater than 200", "Greater than  200" all
    # match. The integer assigned here is the tier rank, so the
    # sort comes out Tier 1 → Tier 5 same as the legacy form.
    _NEW_TIER_RANK = {
        "greaterthan200": 1,
        "120-150":        2,
        "80-110":         3,
        "35-60":          4,
        "lessthan35":     5,
    }
    _norm = "".join(s.lower().split())
    if _norm in _NEW_TIER_RANK:
        return (_NEW_TIER_RANK[_norm], s.lower())

    # ----- Legacy numeric tier labels -----
    # First digit-run anywhere in the string. Works for "Tier 1",
    # "T1", "1", "tier 2", etc.
    import re
    m = re.search(r"\d+", s)
    if m:
        try:
            return (int(m.group(0)), s.lower())
        except ValueError:
            pass
    # No digits — sort lexicographically AFTER numbered tiers
    return (9_999, s.lower())


def _build_sku_tier_map(seg_src):
    """
    Return a dict mapping SKU (Line Name) → tier string, taking
    the most-common tier value seen for each SKU in `seg_src`.
    SKUs absent from the frame, or with all-null tiers, map to
    None and will sort to the bottom via _tier_sort_key.
    """
    if (seg_src is None or seg_src.empty
            or COL_SKU not in seg_src.columns
            or COL_SKU_TIER not in seg_src.columns):
        return {}

    pair = (
        seg_src[[COL_SKU, COL_SKU_TIER]]
        .dropna(subset=[COL_SKU])
        .astype({COL_SKU: str})
    )
    pair[COL_SKU_TIER] = pair[COL_SKU_TIER].astype(str)
    # Mode per SKU — the tier value the SKU is most often tagged
    # with. groupby + value_counts + idxmax is robust to ties.
    out = {}
    for sku, sub in pair.groupby(COL_SKU):
        vc = sub[COL_SKU_TIER].value_counts()
        if vc.empty:
            out[sku] = None
        else:
            out[sku] = vc.index[0]
    return out


def _build_canonical_sku_order(seg_src, top_skus):
    """
    Build the canonical SKU ordering used across all heatmaps
    in a single report:
        1. Group by SKU tier (Tier 1 → Tier 2 → … → unknown).
        2. Within each tier, sort by descending L15M national
           volume (the same metric used to pick the top-N).
    Returns a list of SKU names (length ≤ top_skus). SKUs not
    present in seg_src are silently dropped by callers via the
    .intersection with the heatmap's actual SKU index.
    """
    if (seg_src is None or seg_src.empty
            or COL_SKU not in seg_src.columns
            or "L15M" not in seg_src.columns):
        return []

    active = seg_src[seg_src["L15M"] > 0]
    if active.empty:
        return []

    nat_vol = (
        active.groupby(COL_SKU)["L15M"]
        .sum()
        .sort_values(ascending=False)
        .head(top_skus)
    )
    sku_list = nat_vol.index.astype(str).tolist()

    tier_map = _build_sku_tier_map(seg_src)

    def _sort_key(sku):
        return _tier_sort_key(tier_map.get(sku))

    # Stable sort by tier; within a tier the original L15M-desc
    # order is preserved (Python's sorted is stable).
    return sorted(sku_list, key=_sort_key)


def _sort_heatmap_rows_by_greenness(heat_df, metric_key,
                                    sku_tier_map=None,
                                    fixed_order=None):
    """
    Reorder heatmap rows (SKUs). Three modes, in priority order:

      1. fixed_order given  → reindex to exactly that order
         (intersected with rows actually present). Used by the
         multi-heatmap PDF report so every panel shares the same
         SKU positions.

      2. sku_tier_map given → group rows by tier
         (Tier 1 → Tier 2 → … → unknown), and within each tier
         sort by descending row median (greenest at top, reddest
         at bottom). Used by the on-screen UI.

      3. Neither given      → original behaviour: pure greenness
         sort across all rows.

    For both diverging palettes (RdYlGn, centred at a midpoint)
    and the sequential white→green palettes, higher cell value
    = greener. NaN cells are ignored. Rows with no data fall to
    the bottom of their group.
    """
    if heat_df.empty:
        return heat_df

    # ----- Mode 1: fixed order (cross-metric alignment) -----
    # Used by the multi-heatmap PDF report: every panel should
    # render with the SAME SKU in the SAME row position. We do
    # this in two steps:
    #   a) reindex to `fixed_order` *exactly* — SKUs not in this
    #      metric's output become NaN rows (Plotly renders these
    #      as blank/grey cells, which preserves vertical
    #      position for every other SKU). Without this pad,
    #      panel A missing a SKU that panel B has would shift
    #      every later row by one and break the "row N = SKU X
    #      in every panel" guarantee.
    #   b) any SKUs in heat_df.index but missing from
    #      fixed_order (shouldn't happen with the canonical
    #      builder, but defensive) get appended to the end.
    if fixed_order:
        extras = [s for s in heat_df.index if s not in fixed_order]
        full_order = list(fixed_order) + extras
        return heat_df.reindex(full_order)

    # Row score = median value across segments (robust to one
    # dark column dragging a row).
    row_scores = heat_df.apply(
        lambda r: pd.to_numeric(r, errors="coerce").median(),
        axis=1
    )

    # ----- Mode 2: tier-grouped, then greenness within tier -----
    if sku_tier_map:
        tier_keys = [
            _tier_sort_key(sku_tier_map.get(str(idx)))
            for idx in heat_df.index
        ]
        order_df = pd.DataFrame({
            "sku":   heat_df.index,
            "tier":  tier_keys,
            "score": row_scores.values,
        })
        # Sort by tier ASC (so Tier 1 first), score DESC (greenest
        # at the top of each tier block), NaN scores to the bottom
        # of their tier block.
        order_df = order_df.sort_values(
            by=["tier", "score"],
            ascending=[True, False],
            na_position="last",
            kind="mergesort"   # stable
        )
        return heat_df.loc[order_df["sku"].tolist()]

    # ----- Mode 3: pure greenness (legacy) -----
    ordered_idx = row_scores.sort_values(
        ascending=False, na_position="last"
    ).index
    return heat_df.loc[ordered_idx]


# Metrics that are *additive* across SKUs — for these the bottom
# TOTAL row is a column-wise sum. Everything else gets a column-
# wise mean (a per-SKU average is the most defensible cross-row
# summary for ratio/per-outlet metrics like Lift, Throughput,
# Penetration %, slopes, etc.).
_ADDITIVE_HEATMAP_METRICS = {
    "total_volume",
    "opportunity_value",      # ₹/month (incremental) — additive
    "realistic_incremental",  # units/month per new outlet — additive
}


def _append_heatmap_total_row(heat_df, metric_key):
    """
    Append a bottom 'TOTAL' row to the heatmap pivot. For additive
    metrics (raw volumes, ₹ opportunity, etc.) the row contains
    column-wise sums; for ratio/per-outlet metrics it contains
    column-wise means. NaN cells are ignored so a partially-empty
    column still gets a totals value computed over the cells that
    do have data.
    """
    if heat_df is None or heat_df.empty:
        return heat_df

    is_additive = metric_key in _ADDITIVE_HEATMAP_METRICS

    if is_additive:
        totals = heat_df.sum(axis=0, skipna=True, min_count=1)
    else:
        totals = heat_df.mean(axis=0, skipna=True)

    total_label = "TOTAL"
    # Guarantee uniqueness — if a SKU is somehow literally named
    # "TOTAL" already we bump our label so .loc still works.
    while total_label in heat_df.index:
        total_label = total_label + " "

    # Build a one-row DataFrame with matching columns (handles both
    # plain Index and MultiIndex column layouts), then concat at
    # the bottom. round to keep the cell text the same width as
    # the SKU rows.
    total_df = pd.DataFrame(
        [totals.values],
        index=[total_label],
        columns=heat_df.columns
    )
    out = pd.concat([heat_df, total_df], axis=0)
    return out


def _render_heatmap_png(seg_src, segment_col, period_col,
                        period_months, metric_key,
                        top_skus=25, min_seg_outlets=50,
                        width=1000, height=600,
                        sku_order=None, sku_tier_map=None):
    """
    Render one Auto-Insights heatmap as PNG bytes for PDF
    embedding. Returns None if kaleido isn't installed or the
    slice is too thin.

    Row ordering (priority):
      • sku_order      — exact list; reindexed verbatim. Used by
                         the multi-heatmap PDF report so every
                         panel keeps each SKU in the same row.
      • sku_tier_map   — group by tier, greenness within tier.
                         Used when the caller wants tier grouping
                         but per-metric within-tier sorting.
      • neither        — legacy pure-greenness sort.
    """
    metric_meta = HEATMAP_METRICS[metric_key]

    heat_df = insight_heatmap(
        seg_src, segment_col, period_col, period_months,
        top_skus=top_skus,
        min_seg_outlets=min_seg_outlets,
        metric=metric_key
    )

    if heat_df.empty:
        return None

    # Greenest SKU on top, reddest on bottom — matches the
    # on-screen heatmap ordering so the PDF and UI stay aligned.
    heat_df = _sort_heatmap_rows_by_greenness(
        heat_df, metric_key,
        sku_tier_map=sku_tier_map,
        fixed_order=sku_order
    )

    # Append a bottom TOTAL row (sum for additive metrics, mean
    # otherwise) — mirrors the on-screen heatmap so the PDF report
    # carries the same summary footer.
    heat_df = _append_heatmap_total_row(heat_df, metric_key)

    seg_counts = (
        seg_src.groupby(segment_col)[COL_OUTLET]
        .nunique()
        .to_dict()
    )
    heat_df = heat_df.rename(columns={
        c: f"{c}<br>(n={seg_counts.get(c, 0):,})"
        for c in heat_df.columns
    })

    imshow_kwargs = dict(
        color_continuous_scale=metric_meta["colorscale"],
        aspect="auto",
        labels=dict(
            x=segment_col, y="SKU",
            color=metric_meta["colorbar"]
        ),
        text_auto=metric_meta["fmt"]
    )

    # Colour range so the bottom of the scale is reachable —
    # mirrors the on-screen heatmap.
    heat_vals = heat_df.to_numpy(dtype=float).ravel()
    heat_vals = heat_vals[~np.isnan(heat_vals)]
    if heat_vals.size > 0:
        v_min = float(heat_vals.min())
        v_max = float(heat_vals.max())
        p90 = float(np.quantile(heat_vals, 0.90))
        if metric_meta["midpoint"] is not None:
            mid = float(metric_meta["midpoint"])
            hi = max(p90, mid + 1e-6)
            if v_min >= mid:
                lo = mid - max((hi - mid) * 0.6, 0.2)
            else:
                lo = v_min
            imshow_kwargs["range_color"] = (lo, hi)
            imshow_kwargs["color_continuous_midpoint"] = mid
        else:
            lo = v_min
            hi = p90
            if hi - lo < 1e-6:
                hi = v_max + 1e-6
            imshow_kwargs["range_color"] = (lo, hi)
    elif metric_meta["midpoint"] is not None:
        imshow_kwargs["color_continuous_midpoint"] = (
            metric_meta["midpoint"]
        )

    fig = px.imshow(heat_df, **imshow_kwargs)

    # White background for print, brand-purple title, black body
    # font for readability. The green→red colour scale on the
    # cells themselves is intentionally untouched.
    n_rows = len(heat_df)
    fig.update_layout(
        height=max(420, 26 * n_rows),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#000000", size=12),
        coloraxis_colorbar=dict(
            title=metric_meta["colorbar"]
        ),
        margin=dict(l=140, r=30, t=150, b=40),
        title=dict(
            text=metric_meta["label"],
            font=dict(size=16, color="#2D006B"),
            y=0.98,
            yanchor="top"
        )
    )

    # Column labels at the TOP of the matrix (not bottom).
    fig.update_xaxes(side="top")

    try:
        return fig.to_image(
            format="png",
            width=width,
            height=max(height, 26 * n_rows + 150),
            scale=2
        )
    except Exception:
        return None


def build_segment_action_excel(left_df, right_df,
                               left_matrix, right_matrix,
                               diff_df,
                               left_label="Left",
                               right_label="Right",
                               pen_gap_threshold=10.0,
                               thr_gap_threshold=0.5,
                               trigger_mode="Penetration",
                               outlet_dim_cols=None,
                               outlet_name_col=None):
    """
    Identify SKUs with a material gap between the two segments
    and produce a per-outlet action list.

    `trigger_mode` controls which gap qualifies a SKU:
        • "Penetration"           → only Pen Δ ≥ pen_gap_threshold
        • "Throughput"            → only Thr Δ ≥ thr_gap_threshold
        • "Penetration or Throughput" → either trips the threshold

    Each outlet row carries:
        - RD Code, Outlet Name (if available), Channel
        - Problem  — what's wrong (low penetration / low throughput
                     for the SKU vs the leading segment)
        - Work needed — concrete next step the field rep should
                        take

    Returns Excel bytes (xlsx).
    """

    if outlet_dim_cols is None:
        outlet_dim_cols = [
            COL_ASM, COL_RD, COL_RE, COL_REGION,
            COL_REGION_CAT, COL_CHANNEL, COL_VC,
            COL_VC_CATEGORY
        ]
        if outlet_name_col and outlet_name_col not in outlet_dim_cols:
            outlet_dim_cols = [outlet_name_col] + outlet_dim_cols

    # --- Summary: which SKUs have meaningful gaps ---
    if diff_df is None or diff_df.empty:
        gaps = pd.DataFrame()
    else:
        gaps = diff_df.copy()
        gaps["Abs Pen Δ"] = gaps["Penetration Δ (L−R)"].abs()
        gaps["Abs Thr Δ"] = gaps["Throughput Δ (L−R)"].abs()

        # Mode-aware filtering
        if trigger_mode == "Penetration":
            gaps = gaps[gaps["Abs Pen Δ"] >= pen_gap_threshold]
        elif trigger_mode == "Throughput":
            gaps = gaps[gaps["Abs Thr Δ"] >= thr_gap_threshold]
        else:  # "Penetration or Throughput"
            gaps = gaps[
                (gaps["Abs Pen Δ"] >= pen_gap_threshold) |
                (gaps["Abs Thr Δ"] >= thr_gap_threshold)
            ]

        if not gaps.empty:
            # Decide lagging side based on the chosen trigger
            if trigger_mode == "Throughput":
                # Pure throughput trigger
                lagging_is_left = (
                    gaps["Throughput Δ (L−R)"] < 0
                ).values
            else:
                # Penetration-led (default + combined)
                pen_picks_left = gaps["Penetration Δ (L−R)"] < 0
                pen_active     = gaps["Abs Pen Δ"] >= pen_gap_threshold
                thr_picks_left = gaps["Throughput Δ (L−R)"] < 0
                lagging_is_left = np.where(
                    pen_active, pen_picks_left, thr_picks_left
                )

            gaps["Lagging Segment"]      = np.where(lagging_is_left, left_label,  right_label)
            gaps["Leading Segment"]      = np.where(lagging_is_left, right_label, left_label)
            gaps["Lagging Penetration %"] = np.where(lagging_is_left, gaps["Left Penetration %"], gaps["Right Penetration %"])
            gaps["Leading Penetration %"] = np.where(lagging_is_left, gaps["Right Penetration %"], gaps["Left Penetration %"])
            gaps["Lagging Throughput"]    = np.where(lagging_is_left, gaps["Left Throughput"],  gaps["Right Throughput"])
            gaps["Leading Throughput"]    = np.where(lagging_is_left, gaps["Right Throughput"], gaps["Left Throughput"])

            gaps["Trigger"] = np.where(
                (gaps["Abs Pen Δ"] >= pen_gap_threshold) &
                (gaps["Abs Thr Δ"] >= thr_gap_threshold),
                "Penetration + Throughput",
                np.where(
                    gaps["Abs Pen Δ"] >= pen_gap_threshold,
                    "Penetration only",
                    "Throughput only"
                )
            )

            gaps = gaps.sort_values(
                ["Abs Pen Δ", "Abs Thr Δ"], ascending=[False, False]
            )

    # --- Per-outlet action list ---
    actions = []

    if not gaps.empty:

        lagging_pool = {
            left_label:  left_df,
            right_label: right_df
        }

        for _, row in gaps.iterrows():
            sku       = row[COL_SKU]
            lagging   = row["Lagging Segment"]
            leading   = row["Leading Segment"]
            lag_pen   = row["Lagging Penetration %"]
            lead_pen  = row["Leading Penetration %"]
            lag_thr   = row["Lagging Throughput"]
            lead_thr  = row["Leading Throughput"]
            trigger_text = row["Trigger"]

            seg_df = lagging_pool[lagging]
            if seg_df is None or seg_df.empty:
                continue

            # Outlets that already stock this SKU in the lagging
            # segment (we still surface them when the trigger is
            # Throughput, since the problem there is "stocking it
            # but not selling enough"). For pure Penetration triggers
            # we list only the outlets MISSING the SKU.
            sku_rows = seg_df[seg_df[COL_SKU].astype(str) == str(sku)]
            sku_outlets = set(sku_rows[COL_OUTLET])

            all_outlets = (
                seg_df[[COL_OUTLET] + [
                    c for c in outlet_dim_cols
                    if c in seg_df.columns
                ]]
                .drop_duplicates(subset=[COL_OUTLET])
            )

            # Determine candidate outlets based on the trigger
            pen_triggered = row["Abs Pen Δ"] >= pen_gap_threshold
            thr_triggered = row["Abs Thr Δ"] >= thr_gap_threshold

            # Outlets MISSING the SKU (penetration problem)
            missing = all_outlets[
                ~all_outlets[COL_OUTLET].isin(sku_outlets)
            ].copy()

            if pen_triggered and not missing.empty:
                missing.insert(0, "SKU to Push", sku)
                missing.insert(1, "Lagging Segment", lagging)
                missing.insert(2, "Leading Segment", leading)
                missing.insert(3, "Trigger", trigger_text)
                missing.insert(
                    4, "Problem",
                    f"Not stocked here · Lagging Pen {lag_pen:.1f}% "
                    f"vs Leading {lead_pen:.1f}%"
                )
                missing.insert(
                    5, "Work Needed",
                    f"Place {sku} on shelf and confirm reorder. "
                    f"Expected ~{lead_thr:.1f} units/mo."
                )
                missing.insert(6, "Lagging Pen %", round(float(lag_pen), 1))
                missing.insert(7, "Leading Pen %", round(float(lead_pen), 1))
                missing.insert(8, "Lagging Throughput", round(float(lag_thr), 2))
                missing.insert(9, "Leading Throughput", round(float(lead_thr), 2))
                actions.append(missing)

            # Outlets STOCKING the SKU but underperforming
            # (throughput problem)
            if thr_triggered and lead_thr > 0 and not sku_rows.empty:
                # Per-outlet pieces sold for this SKU in the lagging
                # segment; compare against the leading benchmark.
                stocking = sku_rows[
                    [COL_OUTLET] + [
                        c for c in outlet_dim_cols
                        if c in sku_rows.columns
                    ] + ["pieces_sold"]
                ].drop_duplicates(subset=[COL_OUTLET]).copy()

                # Anything below the leading throughput benchmark is
                # surfaced. We don't know months precisely from a
                # single 'pieces_sold' aggregate, so we rank by
                # ascending pieces_sold and let the field team
                # focus on the weakest outlets.
                stocking = stocking.sort_values("pieces_sold")

                stocking.insert(0, "SKU to Push", sku)
                stocking.insert(1, "Lagging Segment", lagging)
                stocking.insert(2, "Leading Segment", leading)
                stocking.insert(3, "Trigger", trigger_text)
                stocking.insert(
                    4, "Problem",
                    f"Stocked but slow-moving · Lagging Thr "
                    f"{lag_thr:.1f} vs Leading {lead_thr:.1f}"
                )
                stocking.insert(
                    5, "Work Needed",
                    f"Audit visibility & shelf placement of {sku}; "
                    f"check facings, secondary placements, and "
                    f"freshness. Target ~{lead_thr:.1f} units/mo."
                )
                stocking.insert(6, "Lagging Pen %", round(float(lag_pen), 1))
                stocking.insert(7, "Leading Pen %", round(float(lead_pen), 1))
                stocking.insert(8, "Lagging Throughput", round(float(lag_thr), 2))
                stocking.insert(9, "Leading Throughput", round(float(lead_thr), 2))
                stocking = stocking.drop(columns=["pieces_sold"])
                actions.append(stocking)

    actions_df = (
        pd.concat(actions, ignore_index=True)
        if actions else pd.DataFrame()
    )

    # Re-order action columns: identifiers first, then context.
    if not actions_df.empty:
        front = [
            "SKU to Push",
            COL_OUTLET,
        ]
        if outlet_name_col and outlet_name_col in actions_df.columns:
            front.append(outlet_name_col)
        if COL_RD in actions_df.columns:
            front.append(COL_RD)
        if COL_CHANNEL in actions_df.columns:
            front.append(COL_CHANNEL)
        front += [
            "Problem", "Work Needed",
            "Lagging Segment", "Leading Segment", "Trigger",
            "Lagging Pen %", "Leading Pen %",
            "Lagging Throughput", "Leading Throughput"
        ]
        rest = [c for c in actions_df.columns if c not in front]
        ordered_cols = [c for c in front if c in actions_df.columns] + rest
        actions_df = actions_df[ordered_cols]

    summary_df = (
        gaps[[
            COL_SKU, "Trigger",
            "Lagging Segment", "Leading Segment",
            "Lagging Penetration %", "Leading Penetration %",
            "Penetration Δ (L−R)", "Abs Pen Δ",
            "Lagging Throughput", "Leading Throughput",
            "Throughput Δ (L−R)", "Abs Thr Δ"
        ]].rename(
            columns={
                "Penetration Δ (L−R)": "Pen Δ (L−R)",
                "Throughput Δ (L−R)":  "Thr Δ (L−R)"
            }
        ) if not gaps.empty else pd.DataFrame()
    )

    # Add per-SKU outlet counts
    if not summary_df.empty and not actions_df.empty:
        counts = (
            actions_df.groupby("SKU to Push")[COL_OUTLET]
            .nunique()
            .rename("Outlets to Action")
            .reset_index()
            .rename(columns={"SKU to Push": COL_SKU})
        )
        summary_df = summary_df.merge(
            counts, on=COL_SKU, how="left"
        )
        summary_df["Outlets to Action"] = (
            summary_df["Outlets to Action"].fillna(0).astype(int)
        )

    # --- Write to xlsx in memory ---
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        if summary_df.empty:
            pd.DataFrame({
                "Note": [
                    f"No SKUs cross the gap threshold "
                    f"(mode = '{trigger_mode}') for the two "
                    "selected segments."
                ]
            }).to_excel(writer, sheet_name="Summary", index=False)
        else:
            summary_df.to_excel(
                writer, sheet_name="Summary", index=False
            )

        if not actions_df.empty:
            actions_df.to_excel(
                writer, sheet_name="Outlet Action List", index=False
            )

        # Methodology
        pd.DataFrame({
            "Field": [
                "Trigger Mode", "Pen Gap Threshold",
                "Thr Gap Threshold",
                "Lagging Segment",
                "Problem", "Work Needed"
            ],
            "Description": [
                trigger_mode,
                f"{pen_gap_threshold} percentage points",
                f"{thr_gap_threshold} units / outlet / month",
                "Segment with the lower metric on this SKU",
                "What's wrong at this outlet for this SKU",
                "Concrete next step for the field rep"
            ]
        }).to_excel(
            writer, sheet_name="Methodology", index=False
        )

    return out.getvalue()


def _compute_pdf_quadrants(matrix_df, top_n=5):
    """Split matrix into 4 quadrants by median, return dict
    of named DataFrames for the PDF.

    Insertion order is deliberate — the PDF renders the dict
    pairwise into a 2-column grid (top row first, then bottom),
    so the order below produces a true Cartesian layout:

        Penetration on the X axis (low ← → high)
        Throughput  on the Y axis (low ↓ ↑ high)

        ┌───────────────────────┬───────────────────────┐
        │ Top-left              │ Top-right             │
        │ Hidden Gems           │ Core                  │
        │ Low Pen + High Thr    │ High Pen + High Thr   │
        ├───────────────────────┼───────────────────────┤
        │ Bottom-left           │ Bottom-right          │
        │ Tail                  │ Distribution Gaps     │
        │ Low Pen + Low Thr     │ High Pen + Low Thr    │
        └───────────────────────┴───────────────────────┘

    This matches how the same matrix is plotted on a scatter
    chart and how merchandisers read it intuitively.
    """
    if matrix_df.empty:
        return {}

    x_mid = matrix_df["Penetration %"].median()
    y_mid = matrix_df["Throughput_Log"].median()

    base_cols = [COL_BRAND, COL_SKU, "Penetration %", "Throughput"]
    extra_cols = [
        c for c in ["Penetration Impact", "Opportunity Pieces"]
        if c in matrix_df.columns
    ]
    cols = base_cols + extra_cols

    return {
        # ----- TOP ROW (High Throughput) -----
        # Top-left: Low Pen + High Thr → Hidden Gems
        "Hidden Gems (Low Pen + High Thr) — Expand": matrix_df[
            (matrix_df["Penetration %"] < x_mid) &
            (matrix_df["Throughput_Log"] >= y_mid)
        ].sort_values("Throughput", ascending=False).head(top_n)[cols],

        # Top-right: High Pen + High Thr → Core
        "Core (High Pen + High Thr) — Protect": matrix_df[
            (matrix_df["Penetration %"] >= x_mid) &
            (matrix_df["Throughput_Log"] >= y_mid)
        ].sort_values("Throughput", ascending=False).head(top_n)[cols],

        # ----- BOTTOM ROW (Low Throughput) -----
        # Bottom-left: Low Pen + Low Thr → Tail
        "Tail (Low Pen + Low Thr) — Triage": matrix_df[
            (matrix_df["Penetration %"] < x_mid) &
            (matrix_df["Throughput_Log"] < y_mid)
        ].sort_values("Throughput", ascending=False).head(top_n)[cols],

        # Bottom-right: High Pen + Low Thr → Distribution Gaps
        "Distribution Gaps (High Pen + Low Thr) — Review": matrix_df[
            (matrix_df["Penetration %"] >= x_mid) &
            (matrix_df["Throughput_Log"] < y_mid)
        ].sort_values("Penetration %", ascending=False).head(top_n)[cols],
    }


def _build_one_pager_pdf(period_label,
                         universal_filters_str,
                         local_filters_str,
                         subset_metrics,
                         overview_metrics,
                         full_matrix,
                         heatmap_pngs,
                         categorical_block=None):
    """
    Build the comprehensive PDF report.

    Parameters:
        universal_filters_str  — sidebar (universal) filters in
                                 effect for this run.
        local_filters_str      — Auto-Insights tab (local) curation
                                 in effect for this run.
        categorical_block      — optional dict produced by
                                 build_categorical_sku_block(): the
                                 SKU × selected-segment heatmap data
                                 the user explicitly asked for.

    Returns PDF bytes (or None if reportlab is missing).
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors as rl
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, KeepTogether, PageBreak, Image as RLImage
        )
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=landscape(A4),
        leftMargin=12 * mm, rightMargin=12 * mm,
        topMargin=10 * mm, bottomMargin=10 * mm,
        title="VC SKU Action Brief"
    )

    styles = getSampleStyleSheet()

    # ---------- BRAND PALETTE ----------
    # Mondelez-friendly palette per user request:
    #   • #2D006B  deep purple — primary brand (headers, titles)
    #   • #9C7E46  amber-bronze — secondary accent (meta/captions)
    #   • #CBB386  warm tan — light accent (borders, alt rows)
    #   • #FEFEFE  near-white — backgrounds and text on dark
    #   • #000000  black — body text for maximum readability
    BRAND_PRIMARY    = rl.HexColor("#2D006B")
    BRAND_SECONDARY  = rl.HexColor("#9C7E46")
    BRAND_LIGHT      = rl.HexColor("#CBB386")
    BRAND_BG_LIGHT   = rl.HexColor("#FEFEFE")
    BRAND_BLACK      = rl.HexColor("#000000")
    # Soft tint of CBB386 (≈25% mix with white) for alternating
    # row bands — keeps the alt stripe within the brand family
    # without crushing readability.
    BRAND_ALT_TINT   = rl.HexColor("#F0EADC")

    title_st = ParagraphStyle(
        "TitleX", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=20,
        textColor=BRAND_PRIMARY,
        spaceAfter=6, leading=24
    )
    sub_st = ParagraphStyle(
        "SubX", parent=styles["Normal"],
        fontSize=10, textColor=BRAND_SECONDARY,
        spaceAfter=8, leading=13
    )
    h2_st = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13,
        textColor=BRAND_PRIMARY,
        spaceBefore=12, spaceAfter=4, leading=16
    )
    cap_st = ParagraphStyle(
        "Cap", parent=styles["Normal"],
        fontSize=9.5, textColor=BRAND_SECONDARY,
        spaceAfter=4, leading=12
    )
    foot_st = ParagraphStyle(
        "Foot", parent=styles["Normal"],
        fontSize=9, textColor=BRAND_BLACK,
        leading=12
    )

    elems = []

    elems.append(Paragraph("VC SKU Action Brief", title_st))
    elems.append(Paragraph(
        f"Period: <b>{period_label}</b>  ·  "
        f"Generated: {dt.datetime.now().strftime('%d %b %Y, %H:%M')}  ·  "
        f"Subset: <b>{subset_metrics}</b>",
        sub_st
    ))
    elems.append(Paragraph(
        f"<b>Universal filters (sidebar):</b> "
        f"{universal_filters_str or '(none)'}",
        sub_st
    ))
    elems.append(Paragraph(
        f"<b>Local curation (Auto-Insights tab):</b> "
        f"{local_filters_str or '(none)'}",
        sub_st
    ))

    # ---------- OVERVIEW METRICS ----------

    if overview_metrics:
        elems.append(Paragraph("0. Overview", h2_st))

        ov_data = [[
            "Total Base", "Selling Outlets",
            "Penetration %", "Throughput (Monthly Avg)"
        ], [
            f"{overview_metrics.get('total_base', 0):,}",
            f"{overview_metrics.get('selling_outlets', 0):,}",
            f"{overview_metrics.get('penetration_pct', 0):.1f}%",
            f"{overview_metrics.get('throughput', 0):.2f}"
        ]]

        page_w = (297 - 24) * mm
        ov_tbl = Table(ov_data, colWidths=[page_w / 4] * 4)
        ov_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), BRAND_PRIMARY),
            ("TEXTCOLOR",   (0, 0), (-1, 0), BRAND_LIGHT),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 10.5),
            ("FONTNAME",    (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 1), (-1, 1), 20),
            ("TEXTCOLOR",   (0, 1), (-1, 1), BRAND_BLACK),
            ("BACKGROUND",  (0, 1), (-1, 1), BRAND_BG_LIGHT),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",  (0, 1), (-1, 1), 12),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
            ("LINEBELOW",   (0, 0), (-1, 0), 0.5, BRAND_LIGHT),
            ("BOX",         (0, 0), (-1, -1), 0.5, BRAND_LIGHT),
        ]))
        elems.append(ov_tbl)
        elems.append(Spacer(1, 6))

    # ---------- QUADRANT ANALYSIS ----------

    quads = _compute_pdf_quadrants(full_matrix) if full_matrix is not None else {}

    if quads:
        elems.append(Paragraph("1. SKU Quadrant Analysis", h2_st))
        elems.append(Paragraph(
            "Top 5 SKUs in each quadrant of the Penetration × "
            "Throughput matrix.", cap_st
        ))

        page_w = (297 - 24) * mm
        cell_w = page_w / 2 - 4

        quad_items = list(quads.items())
        for i in range(0, len(quad_items), 2):
            row_blocks = []
            for j in range(2):
                if i + j >= len(quad_items):
                    row_blocks.append("")
                    continue
                qname, qdf = quad_items[i + j]

                if qdf.empty:
                    inner = [Paragraph(
                        f"<b>{qname}</b>", cap_st
                    ), Paragraph(
                        "<i>(empty)</i>", cap_st
                    )]
                else:
                    qsub = qdf.copy()
                    for c in qsub.columns:
                        if qsub[c].dtype.kind == "f":
                            qsub[c] = qsub[c].round(2)
                    qdata = [list(qsub.columns)] + qsub.astype(str).values.tolist()
                    n_cols = len(qsub.columns)
                    qtbl = Table(
                        qdata,
                        colWidths=[cell_w / n_cols] * n_cols,
                        repeatRows=1
                    )
                    qtbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
                        ("TEXTCOLOR",  (0, 0), (-1, 0), BRAND_BG_LIGHT),
                        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE",   (0, 0), (-1, 0), 9),
                        ("FONTSIZE",   (0, 1), (-1, -1), 8.5),
                        ("TEXTCOLOR",  (0, 1), (-1, -1), BRAND_BLACK),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                            [BRAND_BG_LIGHT, BRAND_ALT_TINT]),
                        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING",   (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
                        ("GRID",         (0, 0), (-1, -1), 0.25, BRAND_LIGHT),
                    ]))
                    inner = [
                        Paragraph(f"<b>{qname}</b>", cap_st),
                        qtbl
                    ]
                row_blocks.append(inner)

            outer = Table(
                [[row_blocks[0], row_blocks[1]]],
                colWidths=[cell_w + 4, cell_w + 4]
            )
            outer.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            elems.append(outer)
            elems.append(Spacer(1, 4))

    # ---------- SKU PERFORMANCE TABLE ----------

    if full_matrix is not None and not full_matrix.empty:
        elems.append(PageBreak())
        n_skus_pdf = len(full_matrix)
        elems.append(Paragraph(
            f"2. SKU Performance Table  (all {n_skus_pdf} SKUs in the selected filter)",
            h2_st
        ))
        elems.append(Paragraph(
            "All curated SKUs, sorted by monthly throughput. "
            "Penetration Impact (historical) = slope of "
            "throughput on penetration across rolling 3-month "
            "windows — describes what already happened, not a "
            "forecast.",
            cap_st
        ))

        # Per user spec: keep Penetration %, Throughput,
        # Penetration Impact, and Opportunity Volume (when
        # available). Drop outlet_count, total_volume, Δ and
        # velocity columns.
        perf_cols = [
            COL_BRAND, COL_SKU,
            "Penetration %", "Throughput"
        ]
        if "Penetration Impact" in full_matrix.columns:
            perf_cols.append("Penetration Impact")
        if "Opportunity Pieces" in full_matrix.columns:
            perf_cols.append("Opportunity Pieces")
        if "Realistic Multiplier" in full_matrix.columns:
            perf_cols.append("Realistic Multiplier")
        if "Incremental TP / New Outlet" in full_matrix.columns:
            perf_cols.append("Incremental TP / New Outlet")

        # Include ALL SKUs in the selected filter — no head() cap
        perf = (
            full_matrix[perf_cols]
            .sort_values("Throughput", ascending=False)
            .copy()
        )
        perf["Penetration %"] = perf["Penetration %"].round(1)
        perf["Throughput"] = perf["Throughput"].round(2)
        if "Penetration Impact" in perf.columns:
            perf["Penetration Impact"] = (
                perf["Penetration Impact"].round(2)
            )
        if "Opportunity Pieces" in perf.columns:
            perf["Opportunity Pieces"] = (
                perf["Opportunity Pieces"].round(0)
            )
        if "Realistic Multiplier" in perf.columns:
            perf["Realistic Multiplier"] = (
                perf["Realistic Multiplier"].round(3)
            )
        if "Incremental TP / New Outlet" in perf.columns:
            perf["Incremental TP / New Outlet"] = (
                perf["Incremental TP / New Outlet"].round(2)
            )

        pdata = [list(perf.columns)] + perf.astype(str).values.tolist()
        page_w = (297 - 24) * mm
        ptbl = Table(
            pdata,
            colWidths=[page_w / len(perf_cols)] * len(perf_cols),
            repeatRows=1
        )
        ptbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
            ("TEXTCOLOR",  (0, 0), (-1, 0), BRAND_BG_LIGHT),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 9.5),
            ("FONTSIZE",   (0, 1), (-1, -1), 9),
            ("TEXTCOLOR",  (0, 1), (-1, -1), BRAND_BLACK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                [BRAND_BG_LIGHT, BRAND_ALT_TINT]),
            ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("GRID",         (0, 0), (-1, -1), 0.25, BRAND_LIGHT),
        ]))
        elems.append(ptbl)

    # ---------- HEATMAP IMAGES ----------

    if heatmap_pngs:
        page_w = (297 - 24) * mm
        for metric_label, png_bytes in heatmap_pngs:
            if not png_bytes:
                continue
            elems.append(PageBreak())
            elems.append(Paragraph(
                f"3. Heatmap — {metric_label}", h2_st
            ))
            img = RLImage(io.BytesIO(png_bytes), width=page_w, height=page_w * 0.55)
            elems.append(img)
    else:
        # Note when heatmaps couldn't be rendered (kaleido missing)
        elems.append(Spacer(1, 6))
        elems.append(Paragraph(
            "<i>Heatmap images not embedded — install "
            "<b>kaleido</b> (<code>pip install kaleido</code>) "
            "to include them in the PDF.</i>",
            cap_st
        ))

    # ---------- CATEGORICAL SKU BLOCK (per user request) ----------

    if (categorical_block is not None
            and isinstance(categorical_block, dict)
            and not categorical_block.get("table",
                                          pd.DataFrame()).empty):

        elems.append(PageBreak())
        elems.append(Paragraph(
            f"4. Categorical Throughput & Penetration — "
            f"by {display_name(categorical_block.get('segment_col', 'segment'))}",
            h2_st
        ))
        elems.append(Paragraph(
            categorical_block.get(
                "caption",
                "SKU × segment metric grid for the SKUs picked in "
                "the Categorical Sheet."
            ),
            cap_st
        ))

        cat_df = categorical_block["table"].copy()

        # Identify the numeric segment-value columns (the ones that
        # carry throughput / penetration values per segment). We
        # colour these per-row; the SKU and Pen% Range columns get
        # no background.
        cat_num_cols = [
            c for c in cat_df.columns
            if c not in [COL_SKU, "Pen% Range"]
        ]

        # Round numeric cols for display
        for c in cat_num_cols:
            cat_df[c] = pd.to_numeric(
                cat_df[c], errors="coerce"
            ).round(2)

        # ----- Build per-cell background colours (same logic as UI) -----
        # For each row: light→deep green for cells ≥ median, light→deep
        # red for cells < median. Intensity scales with distance from
        # median, clamped to P10/P90 so outliers don't wash the rest.
        GREEN_LIGHT = (220, 245, 220)
        GREEN_DEEP  = (102, 187, 106)
        RED_LIGHT   = (252, 224, 224)
        RED_DEEP    = (229, 115, 115)

        def _blend(c_light, c_deep, t):
            t = max(0.0, min(1.0, t))
            return tuple(
                int(round(c_light[i] + (c_deep[i] - c_light[i]) * t))
                for i in range(3)
            )

        col_idx = {c: i for i, c in enumerate(cat_df.columns)}
        # Build a list of (col, row_data_idx, bg_color_hex) entries
        cell_bg_styles = []
        for ridx, (_, rowvals) in enumerate(cat_df.iterrows()):
            num_series = pd.to_numeric(
                rowvals[cat_num_cols], errors="coerce"
            ).dropna()
            if num_series.empty:
                continue
            median_v = float(num_series.median())
            p10 = float(num_series.quantile(0.10))
            p90 = float(num_series.quantile(0.90))
            low_span = max(median_v - p10, 1e-9)
            high_span = max(p90 - median_v, 1e-9)

            for c in cat_num_cols:
                v = rowvals[c]
                if pd.isna(v):
                    continue
                v_f = float(v)
                if v_f >= median_v:
                    t = (v_f - median_v) / high_span
                    r, g, b = _blend(GREEN_LIGHT, GREEN_DEEP, t)
                else:
                    t = (median_v - v_f) / low_span
                    r, g, b = _blend(RED_LIGHT, RED_DEEP, t)
                # +1 because row 0 in the table is the header
                cell_bg_styles.append((
                    col_idx[c], ridx + 1,
                    rl.Color(r / 255.0, g / 255.0, b / 255.0)
                ))

        # Stringify for display (NaN → blank so coloured cells stay
        # clean rather than showing "nan").
        def _fmt_cell(v):
            if pd.isna(v):
                return ""
            return str(v)

        cdata = [list(cat_df.columns)]
        for _, rowvals in cat_df.iterrows():
            cdata.append([_fmt_cell(rowvals[c]) for c in cat_df.columns])

        n_cols = len(cat_df.columns)
        page_w = (297 - 24) * mm
        ctbl = Table(
            cdata,
            colWidths=[page_w / n_cols] * n_cols,
            repeatRows=1
        )

        # Base table style (header + padding + alternating rows for
        # the non-coloured columns). Coloured cells (green→red heat)
        # keep the same scheme — the user wants only the *frame*
        # palette (purple/tan/white/black) updated.
        base_style = [
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
            ("TEXTCOLOR",  (0, 0), (-1, 0), BRAND_BG_LIGHT),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 9),
            ("FONTSIZE",   (0, 1), (-1, -1), 8.5),
            ("TEXTCOLOR",  (0, 1), (-1, -1), BRAND_BLACK),
            ("LEFTPADDING",  (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("ALIGN",       (0, 1), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",        (0, 0), (-1, -1), 0.25, BRAND_LIGHT),
        ]
        # Alternating row backgrounds for the SKU column only
        sku_col_idx = col_idx.get(COL_SKU, 0)
        for ridx in range(1, len(cdata)):
            shade = (
                BRAND_BG_LIGHT if (ridx - 1) % 2 == 0
                else BRAND_ALT_TINT
            )
            base_style.append((
                "BACKGROUND",
                (sku_col_idx, ridx),
                (sku_col_idx, ridx),
                shade
            ))
            # Pen% Range column gets the same neutral alternating shade
            if "Pen% Range" in col_idx:
                pr_idx = col_idx["Pen% Range"]
                base_style.append((
                    "BACKGROUND",
                    (pr_idx, ridx),
                    (pr_idx, ridx),
                    shade
                ))
            # SKU column bold for readability
            base_style.append((
                "FONTNAME",
                (sku_col_idx, ridx),
                (sku_col_idx, ridx),
                "Helvetica-Bold"
            ))

        # Now overlay the per-cell heat colours for numeric columns
        for cidx, ridx, bg_color in cell_bg_styles:
            base_style.append((
                "BACKGROUND",
                (cidx, ridx),
                (cidx, ridx),
                bg_color
            ))

        ctbl.setStyle(TableStyle(base_style))
        elems.append(ctbl)

    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        "<b>Methodology.</b> Throughput = total pieces ÷ outlets ÷ "
        "months in period (always per-month). Penetration % = outlets "
        "stocking the SKU ÷ total outlets in the slice. Penetration "
        "Impact = slope `b` from y = a + b·x, where y is monthly "
        "throughput and x is penetration %, fitted across rolling "
        "3-month windows of the 15-month history. Lift = segment "
        "monthly throughput ÷ national monthly throughput. Heatmap "
        "colours: green = high / good, red = low / bad.",
        foot_st
    ))

    doc.build(elems)
    return buf.getvalue()


def _build_one_pager_html(period_label,
                          universal_filters_str,
                          local_filters_str,
                          subset_metrics,
                          overview_metrics,
                          full_matrix,
                          categorical_block=None):
    """Fallback — same content as the PDF, rendered as a single HTML page."""

    css = """
    <style>
    body { font-family: -apple-system, Segoe UI, Calibri, sans-serif;
           color: #000000; margin: 28px; font-size: 13px;
           background: #FEFEFE; }
    h1   { color: #2D006B; margin-bottom: 4px; font-size: 26px; }
    .meta { color: #9C7E46; font-size: 12.5px; margin-bottom: 18px;
            line-height: 1.5; }
    h2   { color: #2D006B; font-size: 16px; margin-top: 22px; }
    .cap { color: #9C7E46; font-size: 12px; margin-bottom: 6px; }
    table.rep { width: 100%; border-collapse: collapse; font-size: 12px;
                color: #000000; }
    table.rep th { background: #2D006B; color: #FEFEFE;
                   text-align: left; padding: 7px 9px;
                   font-weight: 700; }
    table.rep td { padding: 6px 9px; border-bottom: 1px solid #CBB386;
                   color: #000000; }
    table.rep tr:nth-child(even) td { background: #F0EADC; }
    .empty { color: #9C7E46; font-style: italic; font-size: 12px; }
    .foot  { color: #000000; font-size: 11px; margin-top: 30px;
             border-top: 1px solid #CBB386; padding-top: 10px;
             line-height: 1.5; }
    @media print { body { margin: 12mm; } h2 { page-break-after: avoid; } }
    </style>
    """

    overview_html = ""
    if overview_metrics:
        overview_html = (
            "<h2>0. Overview</h2>"
            "<table class='rep'><tr>"
            "<th>Total Base</th><th>Selling Outlets</th>"
            "<th>Penetration %</th><th>Throughput (Monthly Avg)</th>"
            "</tr><tr>"
            f"<td><b>{overview_metrics.get('total_base', 0):,}</b></td>"
            f"<td><b>{overview_metrics.get('selling_outlets', 0):,}</b></td>"
            f"<td><b>{overview_metrics.get('penetration_pct', 0):.1f}%</b></td>"
            f"<td><b>{overview_metrics.get('throughput', 0):.2f}</b></td>"
            "</tr></table>"
        )

    perf_html = ""
    if full_matrix is not None and not full_matrix.empty:
        perf_cols = [
            COL_BRAND, COL_SKU,
            "Penetration %", "Throughput"
        ]
        if "Penetration Impact" in full_matrix.columns:
            perf_cols.append("Penetration Impact")
        if "Opportunity Pieces" in full_matrix.columns:
            perf_cols.append("Opportunity Pieces")
        if "Realistic Multiplier" in full_matrix.columns:
            perf_cols.append("Realistic Multiplier")
        if "Incremental TP / New Outlet" in full_matrix.columns:
            perf_cols.append("Incremental TP / New Outlet")

        # Include ALL SKUs in the selected filter — no head() cap
        perf = (
            full_matrix[perf_cols]
            .sort_values("Throughput", ascending=False)
            .copy()
        )
        n_skus_html = len(perf)
        perf["Penetration %"] = perf["Penetration %"].round(1)
        perf["Throughput"] = perf["Throughput"].round(2)
        if "Penetration Impact" in perf.columns:
            perf["Penetration Impact"] = (
                perf["Penetration Impact"].round(2)
            )
        if "Opportunity Pieces" in perf.columns:
            perf["Opportunity Pieces"] = (
                perf["Opportunity Pieces"].round(0)
            )
        if "Realistic Multiplier" in perf.columns:
            perf["Realistic Multiplier"] = (
                perf["Realistic Multiplier"].round(3)
            )
        if "Incremental TP / New Outlet" in perf.columns:
            perf["Incremental TP / New Outlet"] = (
                perf["Incremental TP / New Outlet"].round(2)
            )
        perf_html = (
            f"<h2>2. SKU Performance Table (all {n_skus_html} SKUs in the selected filter)</h2>"
            + perf.to_html(index=False, classes="rep", border=0)
        )

    cat_html = ""
    if (categorical_block is not None
            and isinstance(categorical_block, dict)
            and not categorical_block.get("table",
                                          pd.DataFrame()).empty):
        seg_col = categorical_block.get("segment_col", "segment")
        cat_df = categorical_block["table"].copy()
        for c in cat_df.columns:
            if cat_df[c].dtype.kind == "f":
                cat_df[c] = cat_df[c].round(2)
        cat_html = (
            f"<h2>4. Categorical Throughput & Penetration — by {display_name(seg_col)}</h2>"
            f"<p class='cap'>"
            f"{categorical_block.get('caption', '')}</p>"
            + cat_df.to_html(index=False, classes="rep", border=0)
        )

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>VC SKU Action Brief</title>", css, "</head><body>",
        "<h1>VC SKU Action Brief</h1>",
        f"<div class='meta'>Period: <b>{period_label}</b> · "
        f"Generated {dt.datetime.now().strftime('%d %b %Y, %H:%M')} · "
        f"Subset: <b>{subset_metrics}</b><br>"
        f"Universal filters (sidebar): "
        f"{universal_filters_str or '(none)'}<br>"
        f"Local curation (Auto-Insights tab): "
        f"{local_filters_str or '(none)'}</div>",

        overview_html,
        perf_html,
        cat_html,

        "<div class='foot'>Methodology — Throughput is monthly average. "
        "Lift = segment monthly throughput ÷ national monthly throughput. "
        "Heatmap colours: green = high / good, red = low / bad. "
        "Heatmap images are not embedded in HTML output — for a complete "
        "PDF report including heatmaps, install <code>reportlab</code> and "
        "<code>kaleido</code>.</div>",
        "</body></html>"
    ]
    return "\n".join(parts).encode("utf-8")


# =========================================================
# CATEGORICAL SKU SHEET BUILDER
# =========================================================
# Per the screenshot the user shared: for each selected SKU, show
# the chosen metric across the values of a chosen segment column,
# plus an overall penetration-range column.
# =========================================================

def build_categorical_sku_block(source_df, segment_col,
                                sku_list, period_col,
                                period_months,
                                metric="throughput"):
    """
    Returns a dict with:
        table        — DataFrame (rows = SKU, cols = segment values
                       + Pen% Range)
        segment_col  — the chosen segment dimension name
        metric       — 'throughput' or 'penetration'
        caption      — descriptive caption

    For each SKU and each value of `segment_col`, computes monthly
    throughput (or penetration%, depending on `metric`). Adds a
    "Pen% Range" column showing min–max penetration of that SKU
    across the segment values, mirroring the user's screenshot.
    """

    if source_df.empty or not sku_list or segment_col not in source_df.columns:
        return {"table": pd.DataFrame(), "segment_col": segment_col,
                "metric": metric, "caption": ""}

    src = source_df[
        source_df[COL_SKU].astype(str).isin(sku_list)
    ].copy()
    if src.empty:
        return {"table": pd.DataFrame(), "segment_col": segment_col,
                "metric": metric, "caption": ""}

    # Total outlets per segment value (denominator for penetration)
    seg_totals = (
        src.groupby(segment_col)[COL_OUTLET]
        .nunique()
        .rename("seg_total_outlets")
    )

    active = src[src[period_col] > 0]

    # Aggregations
    grp = (
        active.groupby([COL_SKU, segment_col])
        .agg(
            seg_vol=(period_col, "sum"),
            seg_active_outlets=(COL_OUTLET, "nunique")
        )
        .reset_index()
    )
    grp = grp.merge(
        seg_totals.reset_index(), on=segment_col, how="left"
    )

    grp["throughput"] = (
        _safe_div(grp["seg_vol"], grp["seg_active_outlets"])
        / period_months
    )
    grp["penetration"] = (
        _safe_div(grp["seg_active_outlets"],
                  grp["seg_total_outlets"]) * 100
    )

    # Pivot to wide: rows = SKU, cols = segment values
    val_col = "throughput" if metric == "throughput" else "penetration"

    table = (
        grp.pivot(index=COL_SKU, columns=segment_col, values=val_col)
        .round(2)
    )

    # Apply custom column ordering for VC_CAT and Region/Region_Cat
    ordered_cols = order_segment_values(
        segment_col, list(table.columns)
    )
    table = table[ordered_cols]

    # Penetration-range column (always computed, regardless of metric)
    pen_pivot = grp.pivot(
        index=COL_SKU, columns=segment_col, values="penetration"
    )
    pen_min = pen_pivot.min(axis=1)
    pen_max = pen_pivot.max(axis=1)
    table["Pen% Range"] = [
        f"{int(round(lo))}–{int(round(hi))}%"
        if pd.notna(lo) and pd.notna(hi) else ""
        for lo, hi in zip(pen_min, pen_max)
    ]

    # Reset index so SKU is a column for export
    table = table.reset_index()

    metric_label = (
        "monthly throughput per active outlet"
        if metric == "throughput" else "penetration %"
    )
    seg_label = display_name(segment_col)
    caption = (
        f"Cell value = {metric_label} of the SKU within each value "
        f"of `{seg_label}`. The Pen% Range column shows the SKU's "
        f"min–max penetration across the segment values."
    )

    return {
        "table": table,
        "segment_col": segment_col,
        "metric": metric,
        "caption": caption
    }


def render_matrix(plot_df, title, key=None):

    if plot_df.empty:
        st.info("No data available.")
        return

    x_mid = plot_df["Penetration %"].median()
    y_mid = plot_df["Throughput_Log"].median()

    fig = px.scatter(
        plot_df,
        x="Penetration %",
        y="Throughput_Log",
        color=COL_BRAND,
        text=COL_SKU,
        hover_data=[
            "Penetration %",
            "Throughput",
            "outlet_count",
            "total_volume"
        ]
    )

    fig.update_traces(
        marker=dict(size=10),
        textposition="top center"
    )

    # ----- Dotted best-fit line -----
    # Fit log10(throughput+1) on penetration%. Because the y-axis
    # is the log-transformed throughput, the line is straight in
    # log space — but if you mentally read the y-axis as raw
    # throughput it traces a smooth curve, which is what we want.
    # Skip when there are fewer than 2 distinct penetration values
    # (the slope would be undefined).
    try:
        fit_x_raw = plot_df["Penetration %"].to_numpy(dtype=float)
        fit_y_raw = plot_df["Throughput_Log"].to_numpy(dtype=float)
        mask = np.isfinite(fit_x_raw) & np.isfinite(fit_y_raw)
        fit_x_raw = fit_x_raw[mask]
        fit_y_raw = fit_y_raw[mask]
        if len(fit_x_raw) >= 2 and fit_x_raw.var() > 1e-12:
            slope, intercept = np.polyfit(fit_x_raw, fit_y_raw, 1)
            line_x = np.linspace(
                fit_x_raw.min(), fit_x_raw.max(), 100
            )
            line_y = slope * line_x + intercept
            fig.add_scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(
                    color="#FFD24C",
                    width=2,
                    dash="dot"
                ),
                name="Best fit",
                hoverinfo="skip",
                showlegend=True
            )
    except Exception:
        pass

    fig.add_vline(
        x=x_mid,
        line_dash="dash",
        line_color="white"
    )

    fig.add_hline(
        y=y_mid,
        line_dash="dash",
        line_color="white"
    )

    fig.update_layout(
        title=title,
        height=850,
        plot_bgcolor="#050816",
        paper_bgcolor="#050816",
        font=dict(color="white"),
        yaxis_title="Throughput (Log Scale)"
    )

    st.plotly_chart(fig, use_container_width=True, key=key)


def render_quadrants(plot_df):

    if plot_df.empty:
        return

    x_mid = plot_df["Penetration %"].median()
    y_mid = plot_df["Throughput_Log"].median()

    # Quadrants
    q_hi_pen_hi_thr = plot_df[
        (plot_df["Penetration %"] >= x_mid) &
        (plot_df["Throughput_Log"] >= y_mid)
    ]
    q_lo_pen_hi_thr = plot_df[
        (plot_df["Penetration %"] < x_mid) &
        (plot_df["Throughput_Log"] >= y_mid)
    ]
    q_hi_pen_lo_thr = plot_df[
        (plot_df["Penetration %"] >= x_mid) &
        (plot_df["Throughput_Log"] < y_mid)
    ]
    q_lo_pen_lo_thr = plot_df[
        (plot_df["Penetration %"] < x_mid) &
        (plot_df["Throughput_Log"] < y_mid)
    ]

    # Common display columns — include Penetration Impact and
    # Opportunity Volume when available so each quadrant row
    # carries both the slope and the absolute opportunity in units.
    base_cols = [COL_BRAND, COL_SKU, "Penetration %", "Throughput"]
    extras = [
        c for c in ["Penetration Impact", "Opportunity Pieces"]
        if c in plot_df.columns
    ]
    show_cols = base_cols + extras

    # Layout per spec:
    #   Top-Left  = 🔵 Low Pen + High TP
    #   Top-Right = 🟢 High Pen + High TP
    #   Bot-Left  = 🔴 Low Pen + Low TP
    #   Bot-Right = 🟡 High Pen + Low TP

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("## 🔵 Low Penetration + High Throughput")
        st.dataframe(
            q_lo_pen_hi_thr.sort_values(
                "Throughput", ascending=False
            ).head(5)[show_cols],
            use_container_width=True,
            hide_index=True
        )

    with c2:
        st.markdown("## 🟢 High Penetration + High Throughput")
        st.dataframe(
            q_hi_pen_hi_thr.sort_values(
                "Throughput", ascending=False
            ).head(5)[show_cols],
            use_container_width=True,
            hide_index=True
        )

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("## 🔴 Low Penetration + Low Throughput")
        st.dataframe(
            q_lo_pen_lo_thr.sort_values(
                "Throughput", ascending=False
            ).head(5)[show_cols],
            use_container_width=True,
            hide_index=True
        )

    with c4:
        st.markdown("## 🟡 High Penetration + Low Throughput")
        st.dataframe(
            q_hi_pen_lo_thr.sort_values(
                "Penetration %", ascending=False
            ).head(5)[show_cols],
            use_container_width=True,
            hide_index=True
        )


def apply_filters(base_df, fmap, exclude_sku_list):

    temp_df = base_df.copy()

    for col, vals in fmap.items():

        if not vals:
            continue
        if col not in temp_df.columns:
            continue

        temp_df = temp_df[fuzzy_isin(temp_df[col], vals)]

    if exclude_sku_list and COL_SKU in temp_df.columns:

        temp_df = temp_df[
            ~fuzzy_isin(temp_df[COL_SKU], exclude_sku_list)
        ]

    return temp_df

# =========================================================
# DASHBOARD RENDERER FOR ONE PERIOD
# =========================================================

def render_period_dashboard(period_col, period_months, period_id, period_label):
    """
    Renders the full dashboard (Overview + Compare Segments)
    for a single time-period column (L3M / L15M).
    All Streamlit widget keys are prefixed with `period_id`
    so each tab keeps independent state.
    """

    # ----- Build aggregated frames for this period -----

    outlet_sku_df = build_outlet_sku_df(
        filtered_df,
        period_col
    )

    compare_base_df = build_outlet_sku_df(
        df,
        period_col
    )

    # ----- Apply visualization-only filters at the rows level -----
    # These filters now drive both the chart/table AND the headline
    # KPI tiles, so picking a single SKU collapses every metric to
    # outlets that actually sold that SKU.

    visual_outlet_sku_df = outlet_sku_df.copy()

    # Cooler / Ambient SKU Type — visualization-only. Cooler =
    # Silk + Bournville + Temptations. We filter on COL_BRAND
    # rather than a pre-computed SKU_Type column so the slice
    # stays in sync with COOLER_BRANDS even if the source data
    # adds new brands later.
    if sku_type_filter != "All":
        is_cooler_vis = (
            visual_outlet_sku_df[COL_BRAND]
            .astype(str).str.strip().str.lower()
            .isin(COOLER_BRANDS)
        )
        if sku_type_filter == "Cooler":
            visual_outlet_sku_df = visual_outlet_sku_df[is_cooler_vis]
        else:  # Ambient
            visual_outlet_sku_df = visual_outlet_sku_df[~is_cooler_vis]

    if brand_visual_filter:

        visual_outlet_sku_df = visual_outlet_sku_df[
            visual_outlet_sku_df[COL_BRAND]
            .astype(str)
            .isin(brand_visual_filter)
        ]

    if sku_tier_visual_filter:

        visual_outlet_sku_df = visual_outlet_sku_df[
            visual_outlet_sku_df[COL_SKU_TIER]
            .astype(str)
            .isin(sku_tier_visual_filter)
        ]

    if sku_visual_filter:

        visual_outlet_sku_df = visual_outlet_sku_df[
            visual_outlet_sku_df[COL_SKU]
            .astype(str)
            .isin(sku_visual_filter)
        ]

    overview_df = compute_matrix(
        visual_outlet_sku_df,
        period_months=period_months
    )

    # ----- Per-SKU change vs prior 12 months -----
    # Computed from the row-level filtered_df (with visualization
    # filters applied) so the Δ columns reflect the same slice as
    # the Penetration % and Throughput columns.
    visual_filtered_df = filtered_df.copy()
    if sku_type_filter != "All":
        is_cooler_vis2 = (
            visual_filtered_df[COL_BRAND]
            .astype(str).str.strip().str.lower()
            .isin(COOLER_BRANDS)
        )
        if sku_type_filter == "Cooler":
            visual_filtered_df = visual_filtered_df[is_cooler_vis2]
        else:  # Ambient
            visual_filtered_df = visual_filtered_df[~is_cooler_vis2]
    if brand_visual_filter:
        visual_filtered_df = visual_filtered_df[
            visual_filtered_df[COL_BRAND].astype(str)
            .isin(brand_visual_filter)
        ]
    if sku_tier_visual_filter:
        visual_filtered_df = visual_filtered_df[
            visual_filtered_df[COL_SKU_TIER].astype(str)
            .isin(sku_tier_visual_filter)
        ]
    if sku_visual_filter:
        visual_filtered_df = visual_filtered_df[
            visual_filtered_df[COL_SKU].astype(str)
            .isin(sku_visual_filter)
        ]

    sku_change_df = compute_sku_change(visual_filtered_df)

    if not overview_df.empty and not sku_change_df.empty:
        overview_df = overview_df.merge(
            sku_change_df,
            on=[COL_BRAND, COL_SKU],
            how="left"
        )
        overview_df["Δ Pen (pp)"] = (
            overview_df["Δ Pen (pp)"].fillna(0).round(1)
        )
        overview_df["Δ Throughput (mo)"] = (
            overview_df["Δ Throughput (mo)"].fillna(0).round(1)
        )

    # ----- Per-SKU "Penetration Impact" -----
    # Slope b from throughput ≈ a + b·penetration, fitted across
    # 13 rolling 3-month windows. Penetration in each window =
    # share of outlets that sold the SKU in any of those 3 months
    # (per user spec). Computed once and merged onto overview_df
    # so the SKU Performance Table can show it as a column.
    sku_pen_impact_df = compute_sku_penetration_impact(visual_filtered_df)

    if not overview_df.empty and not sku_pen_impact_df.empty:
        overview_df = overview_df.merge(
            sku_pen_impact_df,
            on=[COL_BRAND, COL_SKU],
            how="left"
        )

    # ----- Opportunity Volume (historical, units/month) -----
    # Reframed as a HISTORICAL representation, not a prediction.
    # Reads as: "In the past, +1pp of penetration co-moved with
    # this much additional monthly volume across the universe."
    # That number is mathematically clean (it is what the linear
    # fit says) but it implicitly assumes the next outlet you
    # reach performs like the average existing outlet — which is
    # not true. The "Incremental TP / New Outlet" metric below
    # corrects for that practical reality.
    #
    # Derivation (unchanged):
    #   Let N = total outlets in scope, P = current penetration %,
    #       T = current monthly throughput, b = Penetration Impact.
    #   Current monthly volume = N × (P/100) × T
    #   After +1pp:              N × ((P+1)/100) × (T + b)
    #   Δ volume = N × [(P+1)(T+b) − P·T] / 100
    #            = N × [T + (P+1)·b] / 100
    #
    # `total_outlets_universe` is the universe N — the count of
    # distinct outlets in the visualization-filtered slice, which
    # matches the denominator used to compute Penetration % in
    # `compute_matrix`.
    if (not overview_df.empty
            and "Penetration Impact" in overview_df.columns
            and "Penetration %" in overview_df.columns
            and "Throughput" in overview_df.columns):
        total_outlets_universe = (
            visual_filtered_df[COL_OUTLET].nunique()
            if not visual_filtered_df.empty else 0
        )
        overview_df["Opportunity Pieces"] = (
            total_outlets_universe
            * (
                overview_df["Throughput"]
                + (overview_df["Penetration %"] + 1)
                * overview_df["Penetration Impact"]
            )
            / 100.0
        )
        # Round to whole units — fractional chocolates aren't useful
        # for a field conversation.
        overview_df["Opportunity Pieces"] = (
            overview_df["Opportunity Pieces"].round(0)
        )

    # ----- Realistic Incremental Throughput per New Outlet -----
    # The practical counterpart to Opportunity Volume. Answers
    # "if I take this SKU to ONE new store, how much extra
    # monthly volume should I realistically expect from THAT
    # store?". Multiplier is strictly < 1 — higher for under-
    # penetrated SKUs (more headroom, easier next outlet) and
    # lower for saturated SKUs (the easy outlets are gone).
    # Historical penetration-slope `b` nudges the multiplier up
    # or down within a bounded ±20% band. See the function's
    # docstring for the full derivation.
    if (not overview_df.empty
            and "Penetration %" in overview_df.columns
            and "Throughput" in overview_df.columns):
        overview_df = compute_realistic_incremental_per_outlet(
            overview_df
        )

    # ----- Sub-tabs: Overview + Compare Segments + Segment SKU
    # Lists + Auto Insights -----
    # `Segment SKU Lists` is a focused, single-segment view that
    # flags the SKUs covering the top 20% of segment rupee
    # throughput (Throughput × Latest MRP) as **Critical**, the
    # next 30% band (cumulative 20–50%) as **Important**, and
    # the following 30% band (50–80%) as **Moderate**. Sits
    # between the comparison view and Auto Insights because
    # it's a "drill-into-one-segment" operation, not a comparison
    # or a cross-segment heatmap.

    sub_tab1, sub_tab2, sub_tab4, sub_tab3 = st.tabs(
        [
            "Overview",
            "Compare Segments",
            "Segment SKU Lists",
            "Auto Insights"
        ]
    )

    # =====================================================
    # OVERVIEW SUB-TAB
    # =====================================================

    with sub_tab1:

        st.title(f"🍫 Mondelez VC Chocolates Dashboard — {period_label}")

        total_base = filtered_df[COL_OUTLET].nunique()

        # Selling outlets = outlets that sold at least one of the
        # SKUs currently in scope (after visualization filters)
        period_outlets = visual_outlet_sku_df[COL_OUTLET].nunique()

        total_volume = visual_outlet_sku_df["pieces_sold"].sum()

        throughput = (
            total_volume / period_outlets / period_months
        ) if period_outlets > 0 else 0

        penetration_pct = (
            (period_outlets / total_base) * 100
        ) if total_base > 0 else 0

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Base", f"{total_base:,}")
        c2.metric("Selling Outlets", f"{period_outlets:,}")
        c3.metric("Penetration %", f"{penetration_pct:.1f}")
        c4.metric(
            "Throughput (Monthly Avg)",
            f"{throughput:.1f}"
        )

        st.divider()

        st.subheader("Penetration % vs Throughput Matrix")

        render_matrix(
            overview_df,
            f"Penetration vs Throughput — {period_label}",
            key=f"{period_id}_overview_matrix"
        )

        st.divider()

        st.subheader("SKU Quadrant Analysis")

        render_quadrants(overview_df)

        st.divider()

        st.subheader("SKU Performance Table")

        if not overview_df.empty:

            # Per user spec: drop total_volume / Δ / Velocity
            # columns. Keep Penetration %, Throughput, the
            # historical Penetration Impact + Opportunity Volume,
            # and the new practical "per new outlet" columns.
            perf_show_cols = [
                COL_BRAND,
                COL_SKU,
                "Penetration %",
                "Throughput",
            ]
            if "Penetration Impact" in overview_df.columns:
                perf_show_cols.append("Penetration Impact")
            if "Opportunity Pieces" in overview_df.columns:
                perf_show_cols.append("Opportunity Pieces")
            if "Realistic Multiplier" in overview_df.columns:
                perf_show_cols.append("Realistic Multiplier")
            if "Incremental TP / New Outlet" in overview_df.columns:
                perf_show_cols.append("Incremental TP / New Outlet")

            st.dataframe(
                overview_df[perf_show_cols].sort_values(
                    "Throughput",
                    ascending=False
                ),
                use_container_width=True,
                height=700,
                hide_index=True
            )
            st.caption(
                "**Penetration Impact (historical)** — across the "
                "last 15 months, this is how monthly throughput "
                "(units / outlet / month) *has historically moved* "
                "in step with a +1pp change in penetration. Slope "
                "`b` of throughput on penetration across rolling "
                "3-month windows. It is a backward-looking "
                "description of what already happened, not a "
                "forecast of what a new store will do.  \n"
                "**Opportunity Pieces (historical)** — if the "
                "historical relationship between penetration and "
                "throughput were applied to the current outlet "
                "universe, +1pp would have meant N × (T + (P+1)·b) "
                "÷ 100 extra monthly units. Same backward-looking "
                "framing as Penetration Impact, scaled to the "
                "universe. Note: this implicitly assumes new "
                "outlets perform like the average existing outlet, "
                "which over-states the practical prize. Use the "
                "two columns below for the realistic on-the-ground "
                "number.  \n"
                "**Realistic Multiplier** — practical adjustment "
                "factor (always < 1) that translates an existing "
                "outlet's average throughput into the expected "
                "throughput of a *new* outlet for this SKU. Built "
                "from a marginal-efficiency baseline of "
                f"{MARGINAL_EFFICIENCY} (new outlets are by "
                "construction less productive than established "
                "ones — no built-up shopper familiarity, no "
                "steady-state rotation), a *saturation drag* that "
                "tapers linearly from 1.0 at 0% penetration down "
                f"to {SAT_FLOOR:.2f} at 100% (the un-reached "
                "stores are weaker than the median, but they are "
                "still real stores in the same segment — so the "
                "drag has a floor, not a crash to zero), and a "
                "smooth bounded (±20%) tanh tilt from the "
                "historical penetration slope. Higher when the "
                "SKU is under-penetrated (lots of headroom and a "
                "good distribution story); lower when it is "
                "already saturated, but never collapses.  \n"
                "**Incremental TP / New Outlet** — the practical "
                "answer to *'if I take this SKU to one new store, "
                "what extra monthly pieces do I get from THAT "
                "store?'*. Equals Throughput × Realistic "
                "Multiplier. Multiply by the number of new stores "
                "you actually open to size the total prize. Adding "
                "stores does not change the throughput of existing "
                "stores, so this is the column to plan against."
            )

        else:

            st.info("No data available.")

    # =====================================================
    # COMPARE SEGMENTS SUB-TAB
    # =====================================================

    with sub_tab2:

        st.title(f"Compare Segments — {period_label}")

        left_col, right_col = st.columns(2)

        def compare_filter(container, label, column, key):

            if column not in df.columns:
                return []

            return container.multiselect(
                label,
                sorted(
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                ),
                key=f"{period_id}_{key}"
            )

        def _safe_sku_multiselect(container, label, key):
            """Column-safe SKU multiselect used inside the compare
            tab — returns [] if COL_SKU isn't in the file."""
            if COL_SKU not in df.columns:
                return []
            return container.multiselect(
                label,
                sorted(df[COL_SKU].dropna().astype(str).unique()),
                key=key,
            )

        # ----- Left filters -----

        with left_col:

            st.subheader("Left segment")

            left_asm = compare_filter(left_col, "ASM", COL_ASM, "l_asm")
            left_channel = compare_filter(left_col, "Channel", COL_CHANNEL, "l_channel")
            left_brand = compare_filter(left_col, "Brand", COL_BRAND, "l_brand")
            left_sku_tier = compare_filter(left_col, "SKU_tier", COL_SKU_TIER, "l_tier")

            left_sku = _safe_sku_multiselect(
                left_col, "SKU name", f"{period_id}_l_sku"
            )

            with st.expander("More filters", expanded=False):

                left_pctype = compare_filter(st, "PC Type", COL_PCTYPE, "l_pctype")
                left_rd = compare_filter(st, "RD Name", COL_RD, "l_rd")
                left_re = compare_filter(st, "RE", COL_RE, "l_re")
                left_region = compare_filter(st, "Region", COL_REGION, "l_region")
                left_region_cat = compare_filter(st, "Region_Cat", COL_REGION_CAT, "l_region_cat")
                left_status = compare_filter(st, "Status", COL_STATUS, "l_status")
                left_vc = compare_filter(st, "VC", COL_VC, "l_vc")
                left_vc_category = compare_filter(st, "VC_Category", COL_VC_CATEGORY, "l_vc_cat")
                left_setty = compare_filter(st, "se_tty", COL_SETTY, "l_setty")

            left_exclude_skus = _safe_sku_multiselect(
                left_col,
                "Exclude SKU from Analysis (Left)",
                f"{period_id}_l_exclude_sku",
            )

        # ----- Right filters -----

        with right_col:

            st.subheader("Right segment")

            right_asm = compare_filter(right_col, "ASM ", COL_ASM, "r_asm")
            right_channel = compare_filter(right_col, "Channel ", COL_CHANNEL, "r_channel")
            right_brand = compare_filter(right_col, "Brand ", COL_BRAND, "r_brand")
            right_sku_tier = compare_filter(right_col, "SKU_tier ", COL_SKU_TIER, "r_tier")

            right_sku = _safe_sku_multiselect(
                right_col, "SKU name ", f"{period_id}_r_sku"
            )

            with st.expander("More filters", expanded=False):

                right_pctype = compare_filter(st, "PC Type ", COL_PCTYPE, "r_pctype")
                right_rd = compare_filter(st, "RD Name ", COL_RD, "r_rd")
                right_re = compare_filter(st, "RE ", COL_RE, "r_re")
                right_region = compare_filter(st, "Region ", COL_REGION, "r_region")
                right_region_cat = compare_filter(st, "Region_Cat ", COL_REGION_CAT, "r_region_cat")
                right_status = compare_filter(st, "Status ", COL_STATUS, "r_status")
                right_vc = compare_filter(st, "VC ", COL_VC, "r_vc")
                right_vc_category = compare_filter(st, "VC_Category ", COL_VC_CATEGORY, "r_vc_cat")
                right_setty = compare_filter(st, "se_tty ", COL_SETTY, "r_setty")

            right_exclude_skus = _safe_sku_multiselect(
                right_col,
                "Exclude SKU from Analysis (Right)",
                f"{period_id}_r_exclude_sku",
            )

        # ----- Build segment maps -----

        left_filters = {
            COL_ASM: left_asm,
            COL_CHANNEL: left_channel,
            COL_PCTYPE: left_pctype,
            COL_RD: left_rd,
            COL_RE: left_re,
            COL_REGION: left_region,
            COL_REGION_CAT: left_region_cat,
            COL_STATUS: left_status,
            COL_VC: left_vc,
            COL_VC_CATEGORY: left_vc_category,
            COL_SETTY: left_setty,
            COL_BRAND: left_brand,
            COL_SKU_TIER: left_sku_tier,
            COL_SKU: left_sku
        }

        right_filters = {
            COL_ASM: right_asm,
            COL_CHANNEL: right_channel,
            COL_PCTYPE: right_pctype,
            COL_RD: right_rd,
            COL_RE: right_re,
            COL_REGION: right_region,
            COL_REGION_CAT: right_region_cat,
            COL_STATUS: right_status,
            COL_VC: right_vc,
            COL_VC_CATEGORY: right_vc_category,
            COL_SETTY: right_setty,
            COL_BRAND: right_brand,
            COL_SKU_TIER: right_sku_tier,
            COL_SKU: right_sku
        }

        left_df = apply_filters(compare_base_df, left_filters, left_exclude_skus)
        right_df = apply_filters(compare_base_df, right_filters, right_exclude_skus)

        left_matrix = compute_matrix(left_df, period_months=period_months)
        right_matrix = compute_matrix(right_df, period_months=period_months)

        # ----- Headline metrics -----
        # Volume / Outlet (mo)  =  total monthly volume across all
        # SKUs in the segment ÷ outlets in the segment. Tells you
        # whether the outlets in this cut are "high-value" (lots of
        # chocolate moving per outlet) or "low-value", independent
        # of which SKUs are stocked.

        def _vol_per_outlet(seg_df, months):
            n = seg_df[COL_OUTLET].nunique()
            if n <= 0:
                return 0.0
            return seg_df["pieces_sold"].sum() / n / months

        left_vpo  = _vol_per_outlet(left_df, period_months)
        right_vpo = _vol_per_outlet(right_df, period_months)

        l1, l2, l3, l4 = left_col.columns(4)

        l1.metric(
            "Outlets",
            f"{left_df[COL_OUTLET].nunique():,}"
        )
        l2.metric(
            "Penetration %",
            f"{left_matrix['Penetration %'].mean():.1f}"
            if not left_matrix.empty else "0"
        )
        l3.metric(
            "Mean Throughput",
            f"{left_matrix['Throughput'].mean():.1f}"
            if not left_matrix.empty else "0"
        )
        l4.metric(
            "Pieces / Outlet  (mo)",
            f"{left_vpo:.1f}",
            help=(
                "Average monthly chocolate pieces per outlet "
                "across the whole portfolio. Higher = "
                "richer / higher-value outlets in this segment."
            )
        )

        r1, r2, r3, r4 = right_col.columns(4)

        r1.metric(
            "Outlets",
            f"{right_df[COL_OUTLET].nunique():,}"
        )
        r2.metric(
            "Penetration %",
            f"{right_matrix['Penetration %'].mean():.1f}"
            if not right_matrix.empty else "0"
        )
        r3.metric(
            "Mean Throughput",
            f"{right_matrix['Throughput'].mean():.1f}"
            if not right_matrix.empty else "0"
        )
        r4.metric(
            "Pieces / Outlet  (mo)",
            f"{right_vpo:.1f}",
            delta=(
                f"{right_vpo - left_vpo:+.1f} vs Left"
                if (left_vpo + right_vpo) > 0 else None
            ),
            help=(
                "Average monthly chocolate pieces per outlet "
                "across the whole portfolio."
            )
        )

        # ----- Top 5 by Throughput -----

        left_col.markdown("### Top 5 SKUs by Throughput")
        left_col.dataframe(
            left_matrix.sort_values("Throughput", ascending=False).head(5)[
                [COL_SKU, COL_BRAND, "Throughput", "Penetration %"]
            ] if not left_matrix.empty else pd.DataFrame(),
            use_container_width=True,
            hide_index=True
        )

        right_col.markdown("### Top 5 SKUs by Throughput")
        right_col.dataframe(
            right_matrix.sort_values("Throughput", ascending=False).head(5)[
                [COL_SKU, COL_BRAND, "Throughput", "Penetration %"]
            ] if not right_matrix.empty else pd.DataFrame(),
            use_container_width=True,
            hide_index=True
        )

        # ----- Top 5 by Penetration % -----

        left_col.markdown("### Top 5 SKUs by Penetration %")
        left_col.dataframe(
            left_matrix.sort_values("Penetration %", ascending=False).head(5)[
                [COL_SKU, COL_BRAND, "Penetration %", "Throughput"]
            ] if not left_matrix.empty else pd.DataFrame(),
            use_container_width=True,
            hide_index=True
        )

        right_col.markdown("### Top 5 SKUs by Penetration %")
        right_col.dataframe(
            right_matrix.sort_values("Penetration %", ascending=False).head(5)[
                [COL_SKU, COL_BRAND, "Penetration %", "Throughput"]
            ] if not right_matrix.empty else pd.DataFrame(),
            use_container_width=True,
            hide_index=True
        )

        # ----- Segment matrices -----

        with left_col:
            render_matrix(
                left_matrix,
                f"Left Segment — {period_label}",
                key=f"{period_id}_left_matrix"
            )

        with right_col:
            render_matrix(
                right_matrix,
                f"Right Segment — {period_label}",
                key=f"{period_id}_right_matrix"
            )

        # ----- Differences -----

        st.markdown("## Differences")

        if left_matrix.empty and right_matrix.empty:

            st.info("No data available.")

        else:

            left_compare = (
                left_matrix[[COL_SKU, "Penetration %", "Throughput"]]
                .rename(
                    columns={
                        "Penetration %": "Left Penetration %",
                        "Throughput": "Left Throughput"
                    }
                )
                if not left_matrix.empty
                else pd.DataFrame(
                    columns=[COL_SKU, "Left Penetration %", "Left Throughput"]
                )
            )

            right_compare = (
                right_matrix[[COL_SKU, "Penetration %", "Throughput"]]
                .rename(
                    columns={
                        "Penetration %": "Right Penetration %",
                        "Throughput": "Right Throughput"
                    }
                )
                if not right_matrix.empty
                else pd.DataFrame(
                    columns=[COL_SKU, "Right Penetration %", "Right Throughput"]
                )
            )

            diff_df = pd.merge(
                left_compare,
                right_compare,
                on=COL_SKU,
                how="outer"
            ).fillna(0)

            diff_df["Penetration Δ (L−R)"] = (
                diff_df["Left Penetration %"] -
                diff_df["Right Penetration %"]
            ).round(1)

            diff_df["Throughput Δ (L−R)"] = (
                diff_df["Left Throughput"] -
                diff_df["Right Throughput"]
            ).round(1)

            diff_df = diff_df[
                [
                    COL_SKU,
                    "Left Penetration %",
                    "Right Penetration %",
                    "Penetration Δ (L−R)",
                    "Left Throughput",
                    "Right Throughput",
                    "Throughput Δ (L−R)"
                ]
            ]

            st.dataframe(
                diff_df.round(1),
                use_container_width=True,
                height=800,
                hide_index=True
            )

            # =====================================================
            # ACTION-LIST EXCEL BUILDER
            # =====================================================

            st.markdown("### 🔧 Action List for Field Team")
            st.caption(
                "Identifies SKUs with a meaningful gap between the "
                "two segments — choose Penetration (outlets that "
                "DON'T stock the SKU), Throughput (outlets stocking "
                "it but selling poorly), or both. Each outlet row "
                "is paired with the RD code, outlet name, channel, "
                "the specific Problem, and the Work Needed by the "
                "field rep. Loads only when you click below."
            )

            act_c1, act_c2, act_c3 = st.columns([1.2, 1, 1])

            with act_c1:
                trigger_mode = st.selectbox(
                    "Trigger gap on",
                    options=[
                        "Penetration",
                        "Throughput",
                        "Penetration or Throughput"
                    ],
                    index=0,
                    key=f"{period_id}_act_mode",
                    help=(
                        "• Penetration — flag SKUs whose distribution "
                        "differs by ≥ Min Pen Δ; surfaces outlets NOT "
                        "stocking them.\n"
                        "• Throughput — flag SKUs whose monthly "
                        "throughput differs by ≥ Min Thr Δ; surfaces "
                        "outlets stocking them but underselling.\n"
                        "• Either — both kinds, in one sheet."
                    )
                )

            with act_c2:
                pen_gap = st.slider(
                    "Min Pen Δ (pp)",
                    min_value=2,
                    max_value=40,
                    value=10,
                    step=1,
                    key=f"{period_id}_act_gap",
                    disabled=(trigger_mode == "Throughput"),
                    help=(
                        "Pp gap between Left and Right penetration. "
                        "Smaller = more SKUs flagged."
                    )
                )

            with act_c3:
                thr_gap = st.slider(
                    "Min Thr Δ (units / mo)",
                    min_value=0.1,
                    max_value=10.0,
                    value=0.5,
                    step=0.1,
                    key=f"{period_id}_act_thr",
                    disabled=(trigger_mode == "Penetration"),
                    help=(
                        "Monthly-throughput gap between segments "
                        "(units / outlet / month)."
                    )
                )

            build_col1, build_col2 = st.columns([1, 2])
            with build_col1:
                build_clicked = st.button(
                    "🛠  Build Action List",
                    key=f"{period_id}_act_build",
                    type="primary",
                    use_container_width=True
                )

            xls_key = f"{period_id}_action_xls"

            if build_clicked:
                with st.spinner("Building outlet action list…"):
                    xls_bytes = build_segment_action_excel(
                        left_df=left_df,
                        right_df=right_df,
                        left_matrix=left_matrix,
                        right_matrix=right_matrix,
                        diff_df=diff_df,
                        left_label="Left Segment",
                        right_label="Right Segment",
                        pen_gap_threshold=float(pen_gap),
                        thr_gap_threshold=float(thr_gap),
                        trigger_mode=trigger_mode,
                        outlet_name_col=COL_OUTLET_NAME
                    )
                    st.session_state[xls_key] = xls_bytes
                st.success("Action list ready — download below.")

            if xls_key in st.session_state:
                ts_act = dt.datetime.now().strftime("%Y%m%d_%H%M")
                with build_col2:
                    st.download_button(
                        "⬇️  Download Excel  (RD + Outlet + Problem + Work)",
                        data=st.session_state[xls_key],
                        file_name=(
                            f"VC_Action_List_{period_label}_{ts_act}.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        ),
                        key=f"{period_id}_act_dl",
                        use_container_width=True
                    )

    # =====================================================
    # SEGMENT SKU LISTS SUB-TAB
    # =====================================================
    # Drill into ONE segment value and surface the SKUs that
    # drive the bulk of the segment's business:
    #
    #   Critical    — the smallest set of SKUs whose
    #                 cumulative Throughput × Latest MRP
    #                 covers the top 20% of the segment's
    #                 total rupee throughput. Non-negotiables
    #                 for a retailer in this band.
    #   Important   — the next 30% band: SKUs whose
    #                 cumulative share lies between 20% and
    #                 50%. Strongly recommended but not
    #                 non-negotiable.
    #   Moderate    — the next 30% band: SKUs whose
    #                 cumulative share lies between 50% and
    #                 80%. Useful range-fillers.
    #
    # Everything past the 80% cumulative line is left unmarked.
    # The ranking-mode picker (Priority / Balanced / Hybrid /
    # Throughput) below still drives the Rank Score column and
    # the All-SKUs sort, but does not affect Critical /
    # Important / Moderate membership — those are always
    # defined on TP × Latest MRP, with a fallback to pure
    # Throughput when no MRP is loaded.

    with sub_tab4:

        st.title(f"Segment SKU Lists — {period_label}")

        st.caption(
            "Pick a segment dimension and one value within it. "
            "SKUs in that segment are flagged as **Critical** "
            "if their cumulative **Throughput × Latest MRP** "
            "covers the top 20% of the segment's total rupee "
            "throughput — the smallest set of SKUs that drives "
            "the core of the business. The next 30% band "
            "(cumulative share between 20% and 50%) is flagged "
            "as **Important** — strongly recommended but not "
            "non-negotiable. The following 30% band (50% to "
            "80%) is flagged as **Moderate** — useful range-"
            "fillers. Everything past 80% is left unmarked. "
            "SKUs below 15% universal penetration "
            "(share of the whole filtered base) are excluded "
            "from the calculation. The **Rank Score** column "
            "(and the All-SKUs sort) is still controlled by "
            "the ranking mode you pick below — Priority / "
            "Balanced / Hybrid / Throughput — but it only "
            "affects display ordering, not Critical / "
            "Important / Moderate membership. Use **Optional "
            "restrictions** below to narrow the universe "
            "further (e.g. restrict to specific SKU tiers, "
            "REs, ASM areas, etc.)."
        )

        # ----- Segment picker -----

        seg_pick_a, seg_pick_b = st.columns([1, 2])

        with seg_pick_a:
            # Candidate segment dimensions, filtered to those the
            # uploaded file actually carries — so a missing VC_CAT /
            # RE / Region / etc. doesn't crash the tab. VC_CAT stays
            # first (and default) when present.
            _ssl_seg_candidates = [
                COL_VC_CATEGORY,
                COL_RE,
                COL_REGION_CAT,
                COL_REGION,
                COL_CHANNEL,
                COL_PCTYPE,
                COL_VC,
            ]
            _ssl_seg_options = [
                c for c in _ssl_seg_candidates
                if c in filtered_df.columns
            ]
            if not _ssl_seg_options:
                st.info(
                    "No segment dimension (VC_CAT, RE, Region, "
                    "Channel, PC Type, VC_MODEL) is present in the "
                    "uploaded file, so the Segment SKU Lists view "
                    "can't group SKUs. Upload a file that carries "
                    "at least one of these columns (or a VC "
                    "Category mapping file) to use this tab."
                )
                return
            ssl_seg_choice = st.selectbox(
                "Segment dimension",
                options=_ssl_seg_options,
                format_func=display_name,
                index=0,
                key=f"{period_id}_ssl_seg_dim",
                help=(
                    "Defaults to VC_CAT because the Must/Should/"
                    "May-Not split was originally sized against "
                    "VC cooler bands. Any other dimension works "
                    "too and falls back to a 4 / 5 / 4 split. "
                    "Only dimensions present in the uploaded file "
                    "are listed."
                )
            )

        # Available values for the chosen dimension, ordered via
        # the same canonical ordering used elsewhere so VC_CAT
        # values appear smallest→largest, regions appear ultra-
        # premium→aspirational, etc.
        ssl_value_options = order_segment_values(
            ssl_seg_choice,
            filtered_df[ssl_seg_choice]
            .dropna().astype(str).unique().tolist()
        )

        with seg_pick_b:
            if ssl_value_options:
                ssl_seg_values = st.multiselect(
                    f"{display_name(ssl_seg_choice)} value(s)",
                    options=ssl_value_options,
                    default=[ssl_value_options[0]],
                    key=f"{period_id}_ssl_seg_value",
                    help=(
                        "Pick one or more values. When multiple "
                        "are selected, the SKU ranking is "
                        "computed across the **combined** universe "
                        "of those segment values."
                    )
                )
            else:
                ssl_seg_values = []
                st.info(
                    f"No values available for "
                    f"`{display_name(ssl_seg_choice)}` in the "
                    "current sidebar filter."
                )

        if not ssl_seg_values:
            st.warning(
                f"Pick at least one value for "
                f"`{display_name(ssl_seg_choice)}`."
            )
            st.stop()

        # Back-compat scalar used for tier counts and report
        # captions. When the user picks several segment values, we
        # use the first one as the "lead" value for the Must/Should/
        # Can tier-count lookup; this matches the prior single-value
        # behaviour and keeps the cooler-band sizing logic honest.
        ssl_seg_value = ssl_seg_values[0]

        # =========================================================
        # OPTIONAL RESTRICTIONS — SKU tier(s) + up to 4 extra dims
        # =========================================================
        # Lets the ZSM narrow the universe before the Must/Should/
        # Can split is computed. All filters are AND-combined with
        # the primary segment selection above and with the sidebar
        # filters that already shaped `filtered_df`.

        with st.expander(
            "🎯 Optional restrictions — SKU tiers, extra "
            "dimensions, ranking mode",
            expanded=False
        ):
            # ----- SKU tier multi-select -----

            ssl_tier_options = sorted(
                filtered_df[COL_SKU_TIER]
                .dropna().astype(str).unique().tolist()
            )
            ssl_tier_include = st.multiselect(
                "Restrict to SKU tier(s)",
                options=ssl_tier_options,
                default=[],
                key=f"{period_id}_ssl_tier_include",
                help=(
                    "Empty = include all tiers. Pick one or more "
                    "tiers (e.g. Tier 1) to compute the Must/Should/"
                    "Can split **within those tiers only** — useful "
                    "when you want a focused list of priority SKUs."
                )
            )

            # ----- Up to 4 extra filter dimensions -----

            st.markdown(
                "**Additional filters** — pick up to 4 extra "
                "dimensions and the values to keep within each. "
                "All are AND-combined."
            )

            # Universe of dimensions that the user can layer on top.
            # Excludes the primary `ssl_seg_choice` (no point
            # filtering on the same column twice) and the brand /
            # SKU / tier columns which are surfaced separately.
            ssl_extra_dim_universe = [
                COL_VC_CATEGORY, COL_RE, COL_REGION_CAT, COL_REGION,
                COL_CHANNEL, COL_PCTYPE, COL_VC, COL_ASM, COL_RD,
                COL_STATUS, COL_SETTY,
            ]
            ssl_extra_dim_universe = [
                c for c in ssl_extra_dim_universe
                if c != ssl_seg_choice and c in filtered_df.columns
            ]

            ssl_extra_filters = {}
            for slot_i in range(4):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    dim_pick = st.selectbox(
                        f"Filter {slot_i + 1} — dimension",
                        options=["(none)"] + ssl_extra_dim_universe,
                        format_func=lambda c: (
                            "(none)" if c == "(none)"
                            else display_name(c)
                        ),
                        index=0,
                        key=(
                            f"{period_id}_ssl_extra_dim_{slot_i}"
                        )
                    )
                with col_b:
                    if dim_pick != "(none)":
                        dim_val_options = order_segment_values(
                            dim_pick,
                            filtered_df[dim_pick]
                            .dropna().astype(str).unique().tolist()
                        )
                        chosen_vals = st.multiselect(
                            f"Filter {slot_i + 1} — value(s)",
                            options=dim_val_options,
                            default=[],
                            key=(
                                f"{period_id}_ssl_extra_val_"
                                f"{slot_i}"
                            ),
                            help=(
                                "Empty = no filtering on this "
                                "dimension."
                            )
                        )
                        if chosen_vals:
                            # Last write wins if the user picks the
                            # same dimension twice across slots.
                            ssl_extra_filters[dim_pick] = chosen_vals
                    else:
                        # Spacer so the row visually aligns even
                        # when no dimension is picked.
                        st.caption(" ")

            # ----- Ranking mode -----

            st.markdown("**Ranking mode**")
            # Default to the balanced value-weighted mode when an
            # MRP file is available — this is the new spec, so
            # neither volume nor MRP dominates the ranking. Falls
            # back to Throughput when no MRP data is loaded.
            # Priority Score (the new default) uses log-scaled,
            # P95-normalised TP and MRP raised to user-controlled
            # importance powers — see the formula footnote below
            # the tier panels for the exact form.
            try:
                _ssl_default_mode_idx = (
                    0 if (_mrp_lookup is not None
                          and not _mrp_lookup.empty)
                    else 3
                )
            except NameError:
                _ssl_default_mode_idx = 3
            ssl_rank_mode = st.radio(
                "How to rank SKUs within the segment",
                options=[
                    "Priority Score (power-weighted TP × MRP)",
                    "Balanced (½·norm Throughput + ½·norm MRP)",
                    "Hybrid (Throughput × Latest MRP)",
                    "Throughput",
                ],
                index=_ssl_default_mode_idx,
                horizontal=True,
                key=f"{period_id}_ssl_rank_mode",
                help=(
                    "**Priority Score** — default when an MRP "
                    "file is loaded. Log-scales Throughput and "
                    "Latest MRP, normalises each to its segment "
                    "P95, then raises them to user-set "
                    "importance powers. Defaults to TP^0.25 × "
                    "(0.3 + 0.7·MRP)^0.75 — MRP-tilted so "
                    "premium SKUs aren't crushed by cheap high-"
                    "pieces ones.  \n"
                    "**Balanced** — equal-weight average of min-"
                    "max-normalised Throughput and Latest MRP.  \n"
                    "**Hybrid (Throughput × Latest MRP)** — rupee-"
                    "throughput weighting.  \n"
                    "**Throughput** — pure units / month / "
                    "outlet.  \n"
                    "All MRP-aware modes use the *latest* "
                    "available MRP per SKU (most recent non-null "
                    "month in the MRP master), not a period "
                    "average.  \n"
                    "Priority / Balanced / Hybrid require an MRP "
                    "file in the sidebar; without one they "
                    "silently fall back to Throughput."
                )
            )

            # ----- Priority Score importance powers -----
            # Only relevant when the user picks the Priority
            # Score ranking mode. We surface the two exponents
            # as number inputs so the ZSM can tilt the ranking
            # toward throughput or MRP without leaving the page.
            # Defaults match the spec: TP=0.25, MRP=0.75.
            if ssl_rank_mode.startswith("Priority Score"):
                pri_c1, pri_c2 = st.columns(2)
                with pri_c1:
                    ssl_tp_importance = st.number_input(
                        "TP importance (exponent on TPₙ)",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.25,
                        step=0.05,
                        key=f"{period_id}_ssl_tp_imp",
                        help=(
                            "Exponent applied to the normalised "
                            "throughput term TPₙ in the Priority "
                            "Score formula. Higher = throughput "
                            "matters more. 0 disables the term."
                        )
                    )
                with pri_c2:
                    ssl_mrp_importance = st.number_input(
                        "MRP importance (exponent on MRPₙ)",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.75,
                        step=0.05,
                        key=f"{period_id}_ssl_mrp_imp",
                        help=(
                            "Exponent applied to the normalised "
                            "MRP term (0.3 + 0.7·MRPₙ) in the "
                            "Priority Score formula. Higher = "
                            "MRP / premium-mix matters more. "
                            "0 disables the term."
                        )
                    )
            else:
                ssl_tp_importance = 0.25
                ssl_mrp_importance = 0.75

        # ----- Restrict to the chosen segment value(s) and tiers -----

        ssl_src = filtered_df[
            filtered_df[ssl_seg_choice].astype(str)
            .isin([str(v) for v in ssl_seg_values])
        ].copy()

        # Apply SKU-tier restriction (universe-level — affects
        # which SKUs are eligible for the Must/Should/Can split).
        if ssl_tier_include:
            ssl_src = ssl_src[
                ssl_src[COL_SKU_TIER].astype(str).isin(ssl_tier_include)
            ]

        # Apply up to 4 extra dimension filters.
        for fcol, fvals in ssl_extra_filters.items():
            ssl_src = ssl_src[
                ssl_src[fcol].astype(str).isin(
                    [str(v) for v in fvals]
                )
            ]

        if ssl_src.empty:
            chosen_values_str = ", ".join(
                str(v) for v in ssl_seg_values
            )
            st.warning(
                f"No outlets in `{display_name(ssl_seg_choice)} "
                f"∈ {{{chosen_values_str}}}` after the sidebar / "
                f"optional filters."
            )
            st.stop()

        # ----- Compute SKU-level metrics within the segment -----
        # Mirrors the per-segment math used elsewhere:
        #   Penetration % = (# outlets selling SKU in period)
        #                   / (# outlets in segment) × 100
        #   Throughput    = sum(period_col volume)
        #                   / (# outlets selling SKU) / months
        # Filtered to SKUs with at least one selling outlet so
        # zero-volume SKUs don't crowd out the tiers.

        ssl_seg_outlets = ssl_src[COL_OUTLET].nunique()

        ssl_active = ssl_src[ssl_src[period_col] > 0]

        if ssl_active.empty:
            st.warning(
                "No SKU sold in this segment for the selected "
                "period."
            )
            st.stop()

        ssl_grp = (
            ssl_active.groupby([COL_BRAND, COL_SKU], as_index=False)
            .agg(
                seg_vol=(period_col, "sum"),
                selling_outlets=(COL_OUTLET, "nunique"),
            )
        )
        ssl_grp["Throughput"] = (
            ssl_grp["seg_vol"]
            / ssl_grp["selling_outlets"]
            / period_months
        ).round(2)
        ssl_grp["Penetration %"] = (
            ssl_grp["selling_outlets"]
            / ssl_seg_outlets
            * 100
        ).round(1)

        # ----- Universal Penetration % floor -----
        # A SKU must reach at least UNIVERSAL_PEN_MIN_PCT of the
        # entire filtered universe (i.e. all outlets after the
        # sidebar filters, **before** narrowing to this segment)
        # to qualify for the tier ranking. This drops thin/
        # niche SKUs that have a strong rank inside one segment
        # but are essentially absent from the overall base —
        # they aren't realistic Must-Have / Should-Have / May-
        # Not-Have candidates.
        UNIVERSAL_PEN_MIN_PCT = 15.0

        _universe_outlets = filtered_df[COL_OUTLET].nunique()
        if _universe_outlets > 0:
            _univ_active = filtered_df[filtered_df[period_col] > 0]
            _univ_pen = (
                _univ_active.groupby(
                    [COL_BRAND, COL_SKU], as_index=False
                )[COL_OUTLET].nunique()
                .rename(columns={COL_OUTLET: "_univ_outlets"})
            )
            _univ_pen["Universal Penetration %"] = (
                _univ_pen["_univ_outlets"]
                / _universe_outlets * 100
            ).round(1)
            ssl_grp = ssl_grp.merge(
                _univ_pen[[COL_BRAND, COL_SKU,
                           "Universal Penetration %"]],
                on=[COL_BRAND, COL_SKU],
                how="left",
            )
            ssl_grp["Universal Penetration %"] = (
                ssl_grp["Universal Penetration %"].fillna(0.0)
            )
        else:
            ssl_grp["Universal Penetration %"] = 0.0

        _n_before_floor = len(ssl_grp)
        ssl_grp = ssl_grp[
            ssl_grp["Universal Penetration %"]
            >= UNIVERSAL_PEN_MIN_PCT
        ].reset_index(drop=True)
        _n_dropped_by_floor = _n_before_floor - len(ssl_grp)

        if ssl_grp.empty:
            st.warning(
                f"No SKU in this segment reaches the "
                f"{UNIVERSAL_PEN_MIN_PCT:.0f}% universal-"
                f"penetration floor. Loosen the sidebar filters "
                f"or pick a broader segment value."
            )
            st.stop()

        # ----- Latest MRP per SKU (from uploaded MRP file) -----
        # Looks up the SKU in the MRP table and takes the *latest*
        # available MRP — i.e. the most recent non-null value
        # across the MRP_MONTH_COLS columns. This deliberately
        # ignores the selected period: the SKU lists are a
        # "what should I stock now" view, so they price each SKU
        # at its current shelf MRP rather than a historical
        # average. If no MRP file is loaded, the column is
        # omitted from the tables.
        try:
            ssl_mrp_table = _mrp_lookup
        except NameError:
            ssl_mrp_table = None

        if ssl_mrp_table is not None and not ssl_mrp_table.empty:
            mrp_cols_ssl = [
                c for c in MRP_MONTH_COLS
                if c in ssl_mrp_table.columns
            ]
            if mrp_cols_ssl:
                # For each SKU, walk the MRP month columns from
                # newest → oldest and pick the first non-null
                # value. bfill along axis=1 after reversing the
                # column order does this in one shot.
                _mrp_recent_first = (
                    ssl_mrp_table[mrp_cols_ssl[::-1]]
                )
                latest_mrp_series = (
                    _mrp_recent_first
                    .bfill(axis=1)
                    .iloc[:, 0]
                    .rename("Latest MRP (₹)")
                )
                sku_keys_ssl = (
                    ssl_grp[COL_SKU]
                    .astype(str).str.strip().str.lower()
                )
                ssl_grp["Latest MRP (₹)"] = (
                    latest_mrp_series.reindex(sku_keys_ssl.values)
                    .fillna(0)
                    .round(2)
                    .values
                )
            else:
                ssl_grp["Latest MRP (₹)"] = np.nan
        else:
            ssl_grp["Latest MRP (₹)"] = np.nan

        # ----- GM Index per SKU (from uploaded GM file) -----
        # Looks up each SKU in `_gm_lookup` (a Series keyed by
        # lowercased SKU/Line Name → GM index) so we can compute
        # the Total Volume × MRP × GM rupee-margin column below.
        # When no GM file is loaded the column stays NaN and the
        # Total Value column is omitted from the tables.
        try:
            _ssl_gm_lookup = _gm_lookup
        except NameError:
            _ssl_gm_lookup = None

        if _ssl_gm_lookup is not None and not _ssl_gm_lookup.empty:
            _gm_keys_ssl = (
                ssl_grp[COL_SKU]
                .astype(str).str.strip().str.lower()
            )
            ssl_grp["GM Index"] = (
                _ssl_gm_lookup
                .reindex(_gm_keys_ssl.values)
                .values
            )
        else:
            ssl_grp["GM Index"] = np.nan

        # ----- Total Value (Vol × MRP × GM) per SKU -----
        # Raw rupee gross-margin for the SKU over the selected
        # period: total units sold (seg_vol) × Latest MRP × GM
        # index. Unlike Throughput × MRP (which is per-outlet
        # per-month), this is the absolute period-total margin
        # contribution of the SKU within the chosen segment.
        # Computed whenever both MRP and GM are available — if
        # either is missing, the column is left as NaN and
        # omitted from the display.
        _has_gm_ssl = not ssl_grp["GM Index"].isna().all()
        _has_mrp_ssl = not ssl_grp["Latest MRP (₹)"].isna().all()
        if _has_mrp_ssl and _has_gm_ssl:
            ssl_grp["Total Value (Vol × MRP × GM)"] = (
                ssl_grp["seg_vol"].astype(float)
                * ssl_grp["Latest MRP (₹)"]
                    .fillna(0.0).astype(float)
                * ssl_grp["GM Index"]
                    .fillna(0.0).astype(float)
            ).round(2)
        else:
            ssl_grp["Total Value (Vol × MRP × GM)"] = np.nan

        # ----- Build Rank Score per ssl_rank_mode -----
        # Priority : (TP_n)^TP_imp × (0.3 + 0.7·MRP_n)^MRP_imp
        #            where TP_n  = log(1+TP)  / log(1+P95(TP))
        #                  MRP_n = log(1+MRP) / log(1+P95(MRP))
        #            Log-scaling compresses outliers; P95 (rather
        #            than max) prevents one massive SKU from
        #            squashing the rest. The (0.3 + 0.7·MRP_n)
        #            floor on the MRP term ensures low-MRP SKUs
        #            aren't multiplied to zero — they're penalised
        #            but not eliminated. Defaults TP_imp=0.25,
        #            MRP_imp=0.75 (MRP-tilted).
        # Balanced : (norm(Throughput) + norm(Latest MRP)) / 2 —
        #            min-max rescaled, equal weight.
        # Hybrid   : Throughput × Latest MRP  (rupee-throughput,
        #            MRP-heavier than Balanced)
        # Throughput : raw monthly units / outlet (old behaviour)
        # All MRP-aware modes use the *latest* available MRP per
        # SKU (most recent non-null month in the MRP master), not
        # a period average — see the Latest MRP lookup above.
        # When MRP is missing for every SKU we silently fall back
        # to Throughput and post a note alongside the headline so
        # the user knows the radio choice didn't take effect.
        _avg_mrp_for_rank = (
            ssl_grp["Latest MRP (₹)"].fillna(0.0).astype(float)
        )
        _has_any_mrp = (_avg_mrp_for_rank > 0).any()

        effective_rank_mode = ssl_rank_mode
        if (
            ssl_rank_mode != "Throughput"
            and not _has_any_mrp
        ):
            effective_rank_mode = "Throughput"

        def _minmax_01(series):
            """
            Min-max rescale a numeric series to [0, 1]. If the
            series is constant (max == min) or empty, returns
            zeros so the term contributes nothing — avoids
            divide-by-zero and prevents a single-SKU segment
            from getting a meaningless score.
            """
            s = series.astype(float)
            lo, hi = s.min(), s.max()
            if not np.isfinite(lo) or not np.isfinite(hi):
                return pd.Series(0.0, index=s.index)
            span = hi - lo
            if span <= 0:
                return pd.Series(0.0, index=s.index)
            return (s - lo) / span

        def _log_p95_norm(series):
            """
            log(1+x) / log(1+P95(x)) — log-scale and normalise
            by the 95th percentile of the segment's values.

            * Log compresses very tall outliers without pushing
              tiny values to zero (unlike a raw P95 divide that
              loses gradient at the high end).
            * Dividing by log(1+P95) means typical SKUs land near
              1.0 and the few above-P95 SKUs go slightly above 1
              — they're rewarded but not exponentially so.
            * Clipped to [0, ∞) so any negative MRP/throughput
              data noise (shouldn't happen but guard rails are
              cheap) can't break the power calculation.
            * If P95 is 0 (all-zero series, or single-value tiny
              segment) returns zeros so the term collapses
              gracefully.
            """
            s = series.astype(float).clip(lower=0.0)
            p95 = np.nanpercentile(s, 95) if len(s) else 0.0
            denom = np.log1p(p95)
            if not np.isfinite(denom) or denom <= 0:
                return pd.Series(0.0, index=s.index)
            return np.log1p(s) / denom

        if effective_rank_mode.startswith("Priority Score"):
            _tp_norm = _log_p95_norm(ssl_grp["Throughput"])
            _mrp_norm = _log_p95_norm(_avg_mrp_for_rank)

            # Floors so the bases of the powers are never < 0
            # (which would explode under fractional exponents)
            # and never 0 when their importance is also 0
            # (0**0 is conventionally 1 in numpy, which is the
            # behaviour we want, but we clip to a tiny epsilon
            # to keep the score strictly monotonic in the input).
            _tp_base = _tp_norm.clip(lower=0.0).astype(float)
            _mrp_base = (
                0.3 + 0.7 * _mrp_norm.clip(lower=0.0)
            ).astype(float)

            _tp_imp = float(ssl_tp_importance)
            _mrp_imp = float(ssl_mrp_importance)

            # np.power handles the 0**positive = 0 case correctly
            # and 0**0 = 1 (so a zeroed-out importance simply
            # neutralises that term, leaving the other to drive
            # the ranking).
            ssl_grp["Rank Score"] = (
                np.power(_tp_base, _tp_imp)
                * np.power(_mrp_base, _mrp_imp)
            ).round(4)
            ssl_score_label = (
                f"Priority Score "
                f"(TPₙ^{_tp_imp:g} × (0.3+0.7·MRPₙ)^{_mrp_imp:g})"
            )
        elif effective_rank_mode.startswith("Balanced"):
            _t_norm = _minmax_01(ssl_grp["Throughput"])
            _m_norm = _minmax_01(_avg_mrp_for_rank)
            ssl_grp["Rank Score"] = (
                0.5 * _t_norm + 0.5 * _m_norm
            ).round(4)
            ssl_score_label = (
                "Balanced Score (0–1, equal-weighted)"
            )
        elif effective_rank_mode == "Hybrid (Throughput × Latest MRP)":
            ssl_grp["Rank Score"] = (
                ssl_grp["Throughput"].astype(float)
                * _avg_mrp_for_rank
            ).round(2)
            ssl_score_label = "Hybrid Score (₹ / outlet / month)"
        else:
            ssl_grp["Rank Score"] = (
                ssl_grp["Throughput"].astype(float).round(2)
            )
            ssl_score_label = "Throughput (units / outlet / month)"

        # ----- Rank, then build Critical + Important + Moderate
        # via cumulative-value bands -----
        # The ranking-mode (Priority / Balanced / Hybrid /
        # Throughput) still drives the **sort order** of the All
        # SKUs view via Rank Score, but tier membership is now
        # defined purely on the **TP × Latest MRP** rupee-
        # throughput value. SKUs are sorted by TP × MRP desc, then
        # we walk the cumulative sum and split into three bands:
        #
        #   Critical   — SKUs covering the top 20% of the
        #                segment's total TP × MRP. The smallest
        #                set that drives the bulk of segment
        #                value; non-negotiables on the shelf.
        #   Important  — the next 30% of cumulative value (i.e.
        #                SKUs whose cumulative share lands
        #                between 20% and 50%). Strongly
        #                recommended but not non-negotiable.
        #   Moderate   — the next 30% of cumulative value (i.e.
        #                SKUs whose cumulative share lands
        #                between 50% and 80%). Useful range-
        #                fillers that round out the assortment.
        #
        # Everything beyond 80% cumulative share is left unmarked.
        #
        # Fallback: if no MRP is available for any SKU in the
        # segment, TP × MRP is uniformly zero and "% of total"
        # is meaningless. In that case we fall back to using
        # plain Throughput as the value for the cumulative band
        # calculation, so the lists still work.
        ssl_ranked = ssl_grp.sort_values(
            "Rank Score", ascending=False
        ).reset_index(drop=True)
        n_total = len(ssl_ranked)

        # Cumulative-share cut-offs for the three tiers.
        # NOTE on naming: the in-code variables MUST_HAVE_CUM_PCT
        # and SHOULD_HAVE_CUM_PCT are kept for back-compatibility
        # with downstream references; they now represent the
        # Critical (0–20%) and Important (20–50%) cut-offs
        # respectively. MODERATE_CUM_PCT is the new third
        # cut-off at 80%.
        MUST_HAVE_CUM_PCT = 20.0    # top 20% of value → Critical
        SHOULD_HAVE_CUM_PCT = 50.0  # next 30% (20–50%) → Important
        MODERATE_CUM_PCT = 80.0     # next 30% (50–80%) → Moderate

        _tp_for_tier = ssl_grp["Throughput"].astype(float)
        _mrp_for_tier = (
            ssl_grp["Latest MRP (₹)"].fillna(0.0).astype(float)
        )
        _tpmrp_for_tier = (_tp_for_tier * _mrp_for_tier).astype(float)
        if _tpmrp_for_tier.sum() <= 0:
            # No MRP loaded (or all zero) — fall back to Throughput
            # so the cumulative-band logic still produces sensible
            # Critical / Important / Moderate lists.
            _tier_value = _tp_for_tier
            _tier_value_basis = "Throughput"
        else:
            _tier_value = _tpmrp_for_tier
            _tier_value_basis = "TP × Latest MRP"

        # Sort the segment SKUs by tier value desc, walk cumulative
        # share, and pick three cut-offs: 20% (end of Critical),
        # 50% (end of Important) and 80% (end of Moderate).
        # Always include at least one SKU in Critical even if the
        # top SKU alone is >20%.
        _tier_sort = (
            ssl_grp.assign(_tier_value=_tier_value.values)
            .sort_values("_tier_value", ascending=False)
            .reset_index(drop=True)
        )

        # SKU Type (Cooler / Ambient) visualization filter — applied
        # BEFORE the cumulative-share walk so the Critical / Important
        # / Moderate bands are computed among ONLY the selected SKU
        # type. Per-SKU metrics (Throughput, Penetration %, Latest
        # MRP, Rank Score) on each row stay identical to the "All"
        # view — only the universe over which the cumulative share
        # is normalised changes, which is what re-ranks the SKUs
        # against each other.
        if sku_type_filter != "All":
            _is_cooler_tier = (
                _tier_sort[COL_BRAND]
                .astype(str).str.strip().str.lower()
                .isin(COOLER_BRANDS)
            )
            if sku_type_filter == "Cooler":
                _tier_sort = _tier_sort[_is_cooler_tier]
            else:  # Ambient
                _tier_sort = _tier_sort[~_is_cooler_tier]
            _tier_sort = _tier_sort.reset_index(drop=True)

        _total_tier_value = float(_tier_sort["_tier_value"].sum())
        if _total_tier_value > 0:
            _cum_pct = (
                _tier_sort["_tier_value"].cumsum()
                / _total_tier_value * 100.0
            )
            # Critical: first index where cumulative share ≥ 20%.
            _must_mask = _cum_pct >= MUST_HAVE_CUM_PCT
            if _must_mask.any():
                _must_cutoff = int(_must_mask.idxmax()) + 1
            else:
                _must_cutoff = len(_tier_sort)
            # Important ends at the first index where cumulative
            # share ≥ 50%. If even the full list doesn't reach 50%,
            # everything past Critical becomes Important.
            _should_mask = _cum_pct >= SHOULD_HAVE_CUM_PCT
            if _should_mask.any():
                _should_cutoff = int(_should_mask.idxmax()) + 1
            else:
                _should_cutoff = len(_tier_sort)
            # Moderate ends at the first index where cumulative
            # share ≥ 80%. If even the full list doesn't reach 80%,
            # everything past Important becomes Moderate.
            _moderate_mask = _cum_pct >= MODERATE_CUM_PCT
            if _moderate_mask.any():
                _moderate_cutoff = int(_moderate_mask.idxmax()) + 1
            else:
                _moderate_cutoff = len(_tier_sort)
            # Guarantee tier boundaries are monotonically non-
            # decreasing even in the degenerate case where a
            # single SKU is already past a later cut-off.
            if _should_cutoff < _must_cutoff:
                _should_cutoff = _must_cutoff
            if _moderate_cutoff < _should_cutoff:
                _moderate_cutoff = _should_cutoff
        else:
            _must_cutoff = 0
            _should_cutoff = 0
            _moderate_cutoff = 0

        must_df = _tier_sort.iloc[:_must_cutoff].drop(
            columns=["_tier_value"], errors="ignore"
        ).copy()
        should_df = _tier_sort.iloc[_must_cutoff:_should_cutoff].drop(
            columns=["_tier_value"], errors="ignore"
        ).copy()
        moderate_df = _tier_sort.iloc[_should_cutoff:_moderate_cutoff].drop(
            columns=["_tier_value"], errors="ignore"
        ).copy()
        must_n = len(must_df)
        should_n = len(should_df)
        moderate_n = len(moderate_df)
        _must_share_pct = (
            float(_tier_sort["_tier_value"].iloc[:_must_cutoff].sum())
            / _total_tier_value * 100.0
            if _total_tier_value > 0 else 0.0
        )
        _should_share_pct = (
            float(
                _tier_sort["_tier_value"]
                .iloc[_must_cutoff:_should_cutoff].sum()
            )
            / _total_tier_value * 100.0
            if _total_tier_value > 0 else 0.0
        )
        _moderate_share_pct = (
            float(
                _tier_sort["_tier_value"]
                .iloc[_should_cutoff:_moderate_cutoff].sum()
            )
            / _total_tier_value * 100.0
            if _total_tier_value > 0 else 0.0
        )

        # ----- Headline metrics for the chosen segment -----

        st.divider()

        # Compact summary string for the segment value(s) chosen.
        # When the user picks one value we show it directly; when
        # they pick several we show a count + the first few values
        # so the metric tile stays readable.
        if len(ssl_seg_values) == 1:
            seg_value_display = str(ssl_seg_values[0])
        else:
            preview = ", ".join(
                str(v) for v in ssl_seg_values[:3]
            )
            if len(ssl_seg_values) > 3:
                preview += f", +{len(ssl_seg_values) - 3} more"
            seg_value_display = (
                f"{len(ssl_seg_values)} values "
                f"({preview})"
            )

        # When the SKU Type filter is active, the headline counts
        # below should reflect the selected subset — same universe
        # the tier walk and All SKUs table operate on. `_tier_sort`
        # has already been filtered upstream, so its length is the
        # correct "SKUs with sales in period" count for the subset.
        _ssl_subset_n = len(_tier_sort)
        _subset_label = (
            f"{sku_type_filter} " if sku_type_filter != "All" else ""
        )
        _value_basis_help = (
            f"{sku_type_filter} subset" if sku_type_filter != "All"
            else "segment"
        )

        h1, h2, h3, h4, h5, h6 = st.columns(6)
        h1.metric(
            f"{display_name(ssl_seg_choice)}",
            seg_value_display
        )
        h2.metric("Outlets in segment", f"{ssl_seg_outlets:,}")
        h3.metric(
            f"{_subset_label}SKUs with sales in period",
            f"{_ssl_subset_n:,}"
        )
        h4.metric(
            f"Critical {_subset_label}SKUs (top {MUST_HAVE_CUM_PCT:.0f}% of value)",
            f"{must_n}",
            help=(
                f"Number of SKUs whose cumulative {_tier_value_basis} "
                f"covers the top {MUST_HAVE_CUM_PCT:.0f}% of the "
                f"{_value_basis_help}'s total {_tier_value_basis}. "
                f"These SKUs actually account for "
                f"{_must_share_pct:.1f}% of {_value_basis_help} value."
            )
        )
        h5.metric(
            f"Important {_subset_label}SKUs (next {SHOULD_HAVE_CUM_PCT - MUST_HAVE_CUM_PCT:.0f}% of value)",
            f"{should_n}",
            help=(
                f"Number of SKUs whose cumulative {_tier_value_basis} "
                f"falls between {MUST_HAVE_CUM_PCT:.0f}% and "
                f"{SHOULD_HAVE_CUM_PCT:.0f}% of the {_value_basis_help} "
                f"total. These SKUs account for an additional "
                f"{_should_share_pct:.1f}% of {_value_basis_help} value."
            )
        )
        h6.metric(
            f"Moderate {_subset_label}SKUs (next {MODERATE_CUM_PCT - SHOULD_HAVE_CUM_PCT:.0f}% of value)",
            f"{moderate_n}",
            help=(
                f"Number of SKUs whose cumulative {_tier_value_basis} "
                f"falls between {SHOULD_HAVE_CUM_PCT:.0f}% and "
                f"{MODERATE_CUM_PCT:.0f}% of the {_value_basis_help} "
                f"total. These SKUs account for an additional "
                f"{_moderate_share_pct:.1f}% of {_value_basis_help} value."
            )
        )

        # Active-restrictions summary — only printed when any
        # optional restriction is actually in use, so the default
        # view stays uncluttered.
        active_bits = []
        if sku_type_filter != "All":
            active_bits.append(
                f"SKU Type = {sku_type_filter} "
                f"(ranking & tiers computed within this subset)"
            )
        if ssl_tier_include:
            active_bits.append(
                f"SKU tier ∈ {{{', '.join(ssl_tier_include)}}}"
            )
        for fcol, fvals in ssl_extra_filters.items():
            preview = ", ".join(str(v) for v in fvals[:3])
            if len(fvals) > 3:
                preview += f", +{len(fvals) - 3} more"
            active_bits.append(
                f"{display_name(fcol)} ∈ {{{preview}}}"
            )
        if effective_rank_mode != "Throughput":
            active_bits.append(f"ranked by: {effective_rank_mode}")
        if active_bits:
            st.caption(
                "**Active restrictions:** " + " · ".join(active_bits)
            )

        # Note when the user picked a value-weighted mode but no
        # MRP data is available — the ranking silently fell back
        # to Throughput and they should know.
        if (
            ssl_rank_mode != "Throughput"
            and effective_rank_mode == "Throughput"
        ):
            st.caption(
                f"ℹ️ Ranking mode `{ssl_rank_mode}` requires Latest "
                f"MRP per SKU. No MRP data is available for these "
                f"SKUs, so ranking has fallen back to **Throughput**. "
                f"Upload an MRP file in the sidebar to enable "
                f"value-weighted ranking."
            )

        # MRP availability hint — useful context if MRP file is
        # missing, otherwise the column is silently absent.
        if ssl_grp["Latest MRP (₹)"].isna().all():
            st.caption(
                "ℹ️ No MRP file uploaded (or no matching SKUs in "
                "the MRP master) — the Latest MRP column is hidden. "
                "Upload an MRP file in the sidebar to see ₹ "
                "values alongside throughput and penetration."
            )

        st.divider()

        # ----- Display columns -----
        # Drop Latest MRP from view when entirely missing so the
        # table is not cluttered with a column of dashes.
        # Show the Rank Score column only when the chosen ranking
        # mode is something other than plain Throughput (otherwise
        # Rank Score == Throughput and duplicates the column).
        ssl_display_cols = [
            COL_BRAND, COL_SKU,
            "Throughput", "Penetration %"
        ]
        if not ssl_grp["Latest MRP (₹)"].isna().all():
            ssl_display_cols.append("Latest MRP (₹)")
        if not ssl_grp["Total Value (Vol × MRP × GM)"].isna().all():
            ssl_display_cols.append("Total Value (Vol × MRP × GM)")
        if effective_rank_mode != "Throughput":
            ssl_display_cols.append("Rank Score")

        # ----- SKU Type (Cooler / Ambient) visualization note -----
        # The Cooler / Ambient view filter is applied UPSTREAM —
        # to `_tier_sort` (before the cumulative-share walk) and to
        # `ssl_all_ranked` (before the Rank column is inserted) —
        # so the four tables below already contain only the
        # selected SKU subset, and the Critical / Important /
        # Moderate band boundaries are computed within that subset.
        # Per-SKU metric values (Throughput, Penetration %, Latest
        # MRP, Rank Score) remain identical to the "All" view — only
        # the Rank column and the tier-band boundaries are subset-
        # relative.

        # ----- Critical list -----
        # Top panel showing the SKUs whose cumulative TP × Latest
        # MRP covers the top 20% of the segment's rupee throughput
        # — the non-negotiable core of the assortment.
        st.subheader("🟢 Critical")
        _basis_scope = (
            f"this segment's **{sku_type_filter}**"
            if sku_type_filter != "All"
            else "this segment's"
        )
        _share_scope = (
            f"{sku_type_filter} subset"
            if sku_type_filter != "All" else "segment"
        )
        st.caption(
            f"SKUs covering the top {MUST_HAVE_CUM_PCT:.0f}% of "
            f"{_basis_scope} total **{_tier_value_basis}** (rupee "
            f"throughput per outlet per month). Sorted by "
            f"{_tier_value_basis} descending. Together these "
            f"{must_n} SKU{'s' if must_n != 1 else ''} account "
            f"for {_must_share_pct:.1f}% of {_share_scope} value."
        )
        if must_df.empty:
            if sku_type_filter != "All":
                st.info(
                    f"No **{sku_type_filter}** SKUs in this segment "
                    f"have non-zero {_tier_value_basis} — the "
                    f"Critical list is empty. Switch the **SKU "
                    f"Type** sidebar radio to see the other SKUs."
                )
            else:
                st.info(
                    "No SKUs in this segment have non-zero "
                    f"{_tier_value_basis} — Critical list is empty."
                )
        else:
            st.dataframe(
                must_df[ssl_display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=min(60 + 38 * len(must_df), 480)
            )

        # ----- Important list -----
        # Second panel showing the next band — SKUs whose
        # cumulative TP × Latest MRP share lies between 20% and
        # 50% of the segment total. Strongly recommended but not
        # non-negotiable.
        st.markdown("")
        st.subheader("🟡 Important")
        st.caption(
            f"SKUs in the next "
            f"{SHOULD_HAVE_CUM_PCT - MUST_HAVE_CUM_PCT:.0f}% band of "
            f"{_basis_scope} total **{_tier_value_basis}** "
            f"(cumulative share between "
            f"{MUST_HAVE_CUM_PCT:.0f}% and "
            f"{SHOULD_HAVE_CUM_PCT:.0f}%). Sorted by "
            f"{_tier_value_basis} descending. Together these "
            f"{should_n} SKU{'s' if should_n != 1 else ''} account "
            f"for an additional {_should_share_pct:.1f}% of "
            f"{_share_scope} value."
        )
        if should_df.empty:
            st.info(
                "No SKUs fall in the Important band for this "
                "segment — either the segment's value is highly "
                "concentrated in the Critical set, or there are "
                "no further SKUs available."
            )
        else:
            st.dataframe(
                should_df[ssl_display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=min(60 + 38 * len(should_df), 480)
            )

        # ----- Moderate list -----
        # Third panel showing the next band — SKUs whose
        # cumulative TP × Latest MRP share lies between 50% and
        # 80% of the segment total. Useful range-fillers that
        # round out the assortment.
        st.markdown("")
        st.subheader("🔵 Moderate")
        st.caption(
            f"SKUs in the next "
            f"{MODERATE_CUM_PCT - SHOULD_HAVE_CUM_PCT:.0f}% band of "
            f"{_basis_scope} total **{_tier_value_basis}** "
            f"(cumulative share between "
            f"{SHOULD_HAVE_CUM_PCT:.0f}% and "
            f"{MODERATE_CUM_PCT:.0f}%). Sorted by "
            f"{_tier_value_basis} descending. Together these "
            f"{moderate_n} SKU{'s' if moderate_n != 1 else ''} account "
            f"for an additional {_moderate_share_pct:.1f}% of "
            f"{_share_scope} value."
        )
        if moderate_df.empty:
            st.info(
                "No SKUs fall in the Moderate band for this "
                "segment — either the segment's value is highly "
                "concentrated in the Critical / Important sets, "
                "or there are no further SKUs available."
            )
        else:
            st.dataframe(
                moderate_df[ssl_display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=min(60 + 38 * len(moderate_df), 480)
            )

        # ----- All SKUs (ranked) -----
        # A second list that contains *every* eligible SKU in this
        # segment, sorted by descending Rank Score (or TP × Latest
        # MRP — toggle below). Useful when the ZSM wants to see
        # where a specific SKU lands relative to the Critical
        # cut-off, or to scan the long tail of small contributors.
        st.markdown("")
        st.subheader("📊 All SKUs (ranked)")

        # Tier-tag column so the user can see, at a glance, which
        # SKUs are in the Critical (top 20%), Important (20–50%)
        # or Moderate (50–80%) cumulative bands. Everything past
        # the 80% cumulative line is left blank.
        must_keys = set(
            zip(must_df[COL_BRAND].astype(str),
                must_df[COL_SKU].astype(str))
        )
        should_keys = set(
            zip(should_df[COL_BRAND].astype(str),
                should_df[COL_SKU].astype(str))
        )
        moderate_keys = set(
            zip(moderate_df[COL_BRAND].astype(str),
                moderate_df[COL_SKU].astype(str))
        )

        def _tier_tag(row):
            key = (str(row[COL_BRAND]), str(row[COL_SKU]))
            if key in must_keys:
                return "🟢 Critical"
            if key in should_keys:
                return "🟡 Important"
            if key in moderate_keys:
                return "🔵 Moderate"
            return ""

        ssl_all_ranked = ssl_ranked.copy()

        # SKU Type (Cooler / Ambient) visualization filter — applied
        # BEFORE the Rank column is inserted so the Rank reflects
        # position within the selected subset (1..N among Cooler-
        # only or Ambient-only). Per-SKU metric values on each row
        # (Throughput, Penetration %, Latest MRP, Rank Score, TP ×
        # MRP) are unchanged from the "All" view — only the rank
        # ordering is relative to the subset.
        if sku_type_filter != "All":
            _is_cooler_all = (
                ssl_all_ranked[COL_BRAND]
                .astype(str).str.strip().str.lower()
                .isin(COOLER_BRANDS)
            )
            if sku_type_filter == "Cooler":
                ssl_all_ranked = ssl_all_ranked[_is_cooler_all]
            else:  # Ambient
                ssl_all_ranked = ssl_all_ranked[~_is_cooler_all]
            ssl_all_ranked = ssl_all_ranked.reset_index(drop=True)

        # ----- TP × Latest MRP (simple rupee throughput) -----
        # Compute the straightforward Throughput × Latest MRP
        # product as a separate column. This is the "simple"
        # alternative to whatever Rank Score the chosen ranking
        # mode produces (which may be a log-scaled, power-
        # weighted, or normalised composite). Always available
        # whenever Latest MRP is available — independent of the
        # rank mode selected above.
        _has_latest_mrp = (
            not ssl_grp["Latest MRP (₹)"].isna().all()
        )
        if _has_latest_mrp and not ssl_all_ranked.empty:
            ssl_all_ranked["TP × MRP (₹)"] = (
                ssl_all_ranked["Throughput"].astype(float)
                * ssl_all_ranked["Latest MRP (₹)"]
                    .fillna(0.0).astype(float)
            ).round(2)

        # ----- Sort toggle -----
        # Let the user flip the All-SKUs table between sorting by
        # the active Rank Score (the "complex formula" — Priority /
        # Balanced / Hybrid / Throughput, whichever is selected
        # above) or by the plain Throughput × Latest MRP product.
        # The Tier column stays the same either way — tiers are
        # always defined by Rank Score, even if the table is
        # sorted by TP × MRP — so a user sorting by TP × MRP can
        # still see which Tier each SKU lands in under the active
        # Rank Score.
        if _has_latest_mrp:
            ssl_all_sort_choice = st.radio(
                "Sort All SKUs by",
                options=[
                    f"Rank Score ({ssl_score_label})",
                    "TP × Latest MRP (simple ₹ throughput)",
                ],
                index=0,
                horizontal=True,
                key=f"{period_id}_ssl_all_sort_choice",
                help=(
                    "**Rank Score** — sort by the score from the "
                    "ranking mode picked above (Priority / "
                    "Balanced / Hybrid / Throughput). This is the "
                    "same ordering used to build the Must / "
                    "Should / May-Not tiers.  \n"
                    "**TP × Latest MRP** — sort by the plain "
                    "Throughput × Latest MRP product (rupees / "
                    "outlet / month), independent of the rank "
                    "mode. Useful for a quick rupee-throughput "
                    "view when the active ranking mode is doing "
                    "something fancier (log scaling, power "
                    "weighting, min-max normalisation, etc.).  \n"
                    "Tiers (Must / Should / May-Not) are always "
                    "defined by Rank Score regardless of which "
                    "sort you pick here."
                )
            )
            _sort_by_tp_mrp = ssl_all_sort_choice.startswith(
                "TP × Latest MRP"
            )
        else:
            _sort_by_tp_mrp = False

        if _sort_by_tp_mrp and "TP × MRP (₹)" in ssl_all_ranked.columns:
            ssl_all_ranked = ssl_all_ranked.sort_values(
                "TP × MRP (₹)", ascending=False
            ).reset_index(drop=True)

        ssl_all_ranked.insert(
            0, "Rank", range(1, len(ssl_all_ranked) + 1)
        )
        ssl_all_ranked["Tier"] = ssl_all_ranked.apply(
            _tier_tag, axis=1
        )

        # Reuse the same display columns the tier panels use, but
        # add Rank (first) and Tier (after SKU name) so the combined
        # view stays scannable. The TP × MRP column sits next to
        # Latest MRP and is shown whenever MRP is available.
        ssl_all_display_cols = ["Rank", COL_BRAND, COL_SKU, "Tier",
                                "Throughput", "Penetration %"]
        if _has_latest_mrp:
            ssl_all_display_cols.append("Latest MRP (₹)")
            ssl_all_display_cols.append("TP × MRP (₹)")
        if (
            "Total Value (Vol × MRP × GM)" in ssl_all_ranked.columns
            and not ssl_all_ranked["Total Value (Vol × MRP × GM)"]
                .isna().all()
        ):
            ssl_all_display_cols.append("Total Value (Vol × MRP × GM)")
        if effective_rank_mode != "Throughput":
            ssl_all_display_cols.append("Rank Score")

        # Caption adapts to the sort choice so the user sees
        # which column the table is ordered by.
        _sort_label = (
            "TP × Latest MRP" if _sort_by_tp_mrp
            else ssl_score_label
        )

        if sku_type_filter == "All":
            st.caption(
                f"All {len(ssl_all_ranked)} eligible SKUs in this "
                f"segment, sorted by descending **{_sort_label}**. "
                f"The **Tier** column flags **🟢 Critical** SKUs "
                f"(cumulative {_tier_value_basis} covers the top "
                f"{MUST_HAVE_CUM_PCT:.0f}% of segment value), "
                f"**🟡 Important** SKUs ({MUST_HAVE_CUM_PCT:.0f}–"
                f"{SHOULD_HAVE_CUM_PCT:.0f}% band) and "
                f"**🔵 Moderate** SKUs ({SHOULD_HAVE_CUM_PCT:.0f}–"
                f"{MODERATE_CUM_PCT:.0f}% band). All other SKUs are "
                f"left unmarked."
            )
        else:
            st.caption(
                f"All {len(ssl_all_ranked)} eligible **"
                f"{sku_type_filter}** SKU"
                f"{'s' if len(ssl_all_ranked) != 1 else ''} in this "
                f"segment, sorted by descending **{_sort_label}** "
                f"and re-ranked among the {sku_type_filter} subset. "
                f"Per-SKU values (Throughput, Penetration %, Latest "
                f"MRP, Rank Score) are unchanged from the All view — "
                f"only the Rank column and the Critical / Important / "
                f"Moderate tier boundaries are computed within the "
                f"{sku_type_filter} subset. Switch the **SKU Type** "
                f"sidebar radio to **All** to see every eligible SKU."
            )

        if ssl_all_ranked.empty:
            st.info(
                f"No **{sku_type_filter}** SKUs in this segment "
                f"slice. Switch the **SKU Type** sidebar radio to "
                f"**All** to see every eligible SKU."
            )
        else:
            st.dataframe(
                ssl_all_ranked[ssl_all_display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=min(60 + 38 * len(ssl_all_ranked), 600)
            )

        # ----- Formula footnote -----
        # State the ranking formula plainly so the ZSM can
        # explain the numbers in a retailer conversation
        # without having to dig into help-text. The active-
        # filter line also captures the universal-penetration
        # floor so users know why a SKU they expected didn't
        # appear.
        if effective_rank_mode.startswith("Priority Score"):
            _formula_str = (
                f"Priority Score = (TPₙ)^{float(ssl_tp_importance):g} "
                f"× (0.3 + 0.7 × MRPₙ)^{float(ssl_mrp_importance):g}, "
                f"where TPₙ = log(1+TP) / log(1+P95(TP)) and "
                f"MRPₙ = log(1+MRP) / log(1+P95(MRP)). "
                f"P95 is taken across the segment's eligible SKUs. "
                f"Edit the TP / MRP importance inputs above to "
                f"re-tilt the ranking."
            )
        elif effective_rank_mode.startswith("Balanced"):
            _formula_str = (
                "Rank Score = ½ × norm(Throughput) + "
                "½ × norm(Latest MRP), where norm(x) min-max "
                "rescales x to 0–1 across the segment's "
                "eligible SKUs."
            )
        elif effective_rank_mode == "Hybrid (Throughput × Latest MRP)":
            _formula_str = (
                "Rank Score = Throughput × Latest MRP "
                "(rupee throughput per outlet per month)."
            )
        else:
            _formula_str = (
                "Rank Score = Throughput "
                "(units / outlet / month)."
            )
        st.caption(
            f"📐 **Formula** — {_formula_str}  \n"
            f"📊 **Eligibility** — SKUs must reach ≥ "
            f"{UNIVERSAL_PEN_MIN_PCT:.0f}% universal "
            f"penetration (share of the whole filtered outlet "
            f"base) to enter the ranking. "
            + (
                f"_{_n_dropped_by_floor} SKU"
                f"{'s' if _n_dropped_by_floor != 1 else ''} "
                f"dropped by this floor in the current view._"
                if _n_dropped_by_floor > 0 else ""
            )
        )

        st.divider()

        # ----- Critical + Important + Moderate long table + download -----
        # Useful for export / pasting into ZSM templates. The
        # `Tier` column carries the Critical / Important / Moderate
        # flag so downstream pivots / sorts keep the grouping.

        with st.expander(
            "📋 Critical + Important + Moderate table + CSV download",
            expanded=False
        ):
            _export_parts = []
            if not must_df.empty:
                _m = must_df[ssl_display_cols].copy()
                _m.insert(0, "Tier", "Critical")
                _export_parts.append(_m)
            if not should_df.empty:
                _s = should_df[ssl_display_cols].copy()
                _s.insert(0, "Tier", "Important")
                _export_parts.append(_s)
            if not moderate_df.empty:
                _md = moderate_df[ssl_display_cols].copy()
                _md.insert(0, "Tier", "Moderate")
                _export_parts.append(_md)
            if _export_parts:
                combined_df = pd.concat(
                    _export_parts, ignore_index=True
                )
                st.dataframe(
                    combined_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(60 + 38 * len(combined_df), 600)
                )
                csv_bytes = combined_df.to_csv(
                    index=False
                ).encode("utf-8")
                ts_ssl = dt.datetime.now().strftime("%Y%m%d_%H%M")
                # Sanitize segment value(s) for the filename — strip
                # any path-unsafe characters and collapse multi-
                # value selections to a short tag so the download
                # lands cleanly even when labels are e.g. "OWN A/C"
                # or "700-1000L" or when several values are selected.
                if len(ssl_seg_values) == 1:
                    safe_seg = "".join(
                        c if c.isalnum() else "_"
                        for c in str(ssl_seg_values[0])
                    )
                else:
                    safe_seg = f"{len(ssl_seg_values)}values"
                st.download_button(
                    "⬇️  Download CSV",
                    data=csv_bytes,
                    file_name=(
                        f"SKU_CriticalImportantModerate_{display_name(ssl_seg_choice)}_"
                        f"{safe_seg}_{period_label}_{ts_ssl}.csv"
                    ),
                    mime="text/csv",
                    key=f"{period_id}_ssl_csv",
                    use_container_width=True
                )
            else:
                st.info(
                    "No SKUs in the Critical, Important or "
                    "Moderate lists for this segment value."
                )

    # =====================================================
    # AUTO INSIGHTS SUB-TAB
    # =====================================================

    with sub_tab3:

        st.title(f"Auto Insights — {period_label}")

        with st.expander(
            "📖 How to read these — L3 vs L15 framework",
            expanded=False
        ):
            st.markdown(
                """
- **L15 = strategic baseline** — what's the long-run shape.
- **L3 = current signal** — what's happening right now.
- The honest momentum number is **L3 monthly avg ÷ prior-12 monthly avg**, where prior-12 = (L15 − L3) / 12. This isolates the recent quarter from the baseline so the ratio actually means something.
- A SKU lifting in **L3** is a fresh signal worth investigating. Cross-check against the L15 baseline before acting.
                """
            )

        # ----- Curation panel -----

        with st.container():

            st.markdown("##### 🔧 Curate what counts as 'relevant'")

            curate_c1, curate_c2 = st.columns(2)

            status_options = (
                sorted(df[COL_STATUS].dropna().astype(str).unique())
                if COL_STATUS in df.columns else []
            )
            default_status = [
                s for s in status_options
                if s.upper().startswith("OPEN")
            ] or status_options

            with curate_c1:
                status_include = st.multiselect(
                    "Status to include",
                    options=status_options,
                    default=default_status,
                    key=f"{period_id}_status_inc",
                    help=(
                        "Pick which outlet Status values to "
                        "include in the analysis."
                    )
                )

                min_outlets_floor = st.slider(
                    "Min outlets in L15M to qualify",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    key=f"{period_id}_min_outlets",
                    help=(
                        "SKUs reaching fewer outlets than this "
                        "are too thin to call insights on."
                    )
                )

                vc_model_exclude = st.multiselect(
                    "Exclude VC_MODEL values",
                    options=(
                        sorted(df[COL_VC].dropna().astype(str).unique())
                        if COL_VC in df.columns else []
                    ),
                    default=[],
                    key=f"{period_id}_vc_model_excl",
                    help=(
                        "E.g. exclude `NONE` to drop outlets "
                        "without a cooler from the analysis."
                    )
                )

            with curate_c2:
                tier_choices = sorted(
                    df[COL_SKU_TIER].dropna().astype(str).unique()
                )

                tier_include = st.multiselect(
                    "Restrict to SKU tiers",
                    options=tier_choices,
                    default=[],
                    key=f"{period_id}_tier_include",
                    help="Empty = include all tiers."
                )

                manual_exclude = st.multiselect(
                    "Manually exclude these SKUs",
                    options=sorted(
                        df[COL_SKU].dropna().astype(str).unique()
                    ),
                    default=[],
                    key=f"{period_id}_manual_excl"
                )

        st.divider()

        # ----- Build the curated source frame for this tab -----

        curated_src = filtered_df.copy()

        if status_include and COL_STATUS in curated_src.columns:
            curated_src = curated_src[
                curated_src[COL_STATUS]
                .astype(str)
                .isin(status_include)
            ]

        if vc_model_exclude and COL_VC in curated_src.columns:
            curated_src = curated_src[
                ~curated_src[COL_VC]
                .astype(str)
                .isin(vc_model_exclude)
            ]

        if tier_include and COL_SKU_TIER in curated_src.columns:
            curated_src = curated_src[
                curated_src[COL_SKU_TIER]
                .astype(str)
                .isin(tier_include)
            ]

        if manual_exclude and COL_SKU in curated_src.columns:
            curated_src = curated_src[
                ~curated_src[COL_SKU]
                .astype(str)
                .isin(manual_exclude)
            ]

        # Build the period-specific outlet x SKU frame from the
        # curated source
        curated_outlet_sku = build_outlet_sku_df(
            curated_src,
            period_col
        )

        full_matrix = compute_matrix(
            curated_outlet_sku,
            period_months=period_months
        )

        # Apply min-outlets floor at the matrix level
        full_matrix = full_matrix[
            full_matrix["outlet_count"] >= min_outlets_floor
        ]

        # ----- Live subset metrics -----
        # ─ Outlets selling the curated SKUs: how many outlets are
        #   actually selling at least one of the SKUs that survived
        #   curation (Status / Tier / Excludes etc.) in the chosen
        #   period. This is the "in how many outlets is this Tier 1
        #   SKU sold" answer.
        # ─ SKUs after curation: unchanged.
        # ─ Qualifying outlets (universal): how many outlets pass
        #   the universal sidebar filters before the Auto-Insights
        #   local curation is layered on.

        sub_c1, sub_c2, sub_c3 = st.columns(3)

        selling_outlets_count = (
            curated_outlet_sku[COL_OUTLET].nunique()
            if not curated_outlet_sku.empty else 0
        )

        # If the user restricted to specific SKU tier(s), make the
        # label explicit about that — otherwise keep it generic.
        if tier_include:
            sku_metric_label = (
                f"Outlets selling {', '.join(tier_include)} SKUs"
            )
        else:
            sku_metric_label = "Outlets selling curated SKUs"

        sub_c1.metric(
            sku_metric_label,
            f"{selling_outlets_count:,}"
        )

        sub_c2.metric(
            "SKUs after curation",
            f"{full_matrix.shape[0]:,}"
        )

        sub_c3.metric(
            "Qualifying outlets (universal filters)",
            f"{filtered_df[COL_OUTLET].nunique():,}"
        )

        # Place a placeholder for the download button — the actual
        # bytes are built at the end of this tab once every insight
        # frame has been computed.
        report_placeholder = st.container()

        st.divider()

        # =================================================
        # PEN ↔ THROUGHPUT REGRESSION  (R² across curated SKUs)
        # =================================================
        # OLS fit of monthly Throughput on Penetration % across
        # the curated SKU set. Updates live with every filter
        # (sidebar universal filters, Auto-Insights curation,
        # SKU-tier / VC-model excludes). Re-pick a sidebar slice
        # (e.g. CHANNEL = CHEMIST, ASM = some Bandra area) and
        # the R², slope and chart all re-fit on just that slice.

        st.subheader("📈 Penetration ↔ Throughput Regression")
        st.caption(
            "Linear fit of monthly Throughput on Penetration % "
            "across the curated SKUs. **Slope** tells you how "
            "many extra units / outlet / month each +1pp of "
            "penetration is worth on average; **R²** tells you "
            "how tightly the SKUs hug that line in the current "
            "slice. Re-slice with the sidebar (e.g. Channel = "
            "CHEMIST, ASM = Bandra) and the fit re-computes."
        )

        # Need at least 2 points for an OLS fit.
        reg_matrix = full_matrix.dropna(
            subset=["Penetration %", "Throughput"]
        ) if not full_matrix.empty else pd.DataFrame()

        if reg_matrix.shape[0] < 3:
            st.info(
                "Need at least 3 SKUs after curation to run the "
                "regression. Loosen the curation filters or lower "
                "the Min-outlets floor."
            )
        else:
            x = reg_matrix["Penetration %"].to_numpy(dtype=float)
            y = reg_matrix["Throughput"].to_numpy(dtype=float)

            # OLS via polyfit — slope, intercept, R²
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            # Pearson r (signed) — useful sign cue alongside R²
            pearson_r = (
                float(np.corrcoef(x, y)[0, 1])
                if np.std(x) > 0 and np.std(y) > 0 else 0.0
            )

            reg_m1, reg_m2, reg_m3, reg_m4 = st.columns(4)
            reg_m1.metric("R²", f"{r2:.3f}")
            reg_m2.metric("Pearson r", f"{pearson_r:+.3f}")
            reg_m3.metric(
                "Slope",
                f"{slope:+.2f}",
                help=(
                    "Δ Throughput (units / outlet / month) per "
                    "+1pp of Penetration."
                )
            )
            reg_m4.metric(
                "SKUs in fit",
                f"{reg_matrix.shape[0]:,}"
            )

            # Plain-English interpretation
            if slope >= 0:
                slope_msg = (
                    f"Each +1pp of penetration is associated with "
                    f"about **{slope:+.2f}** extra units / outlet "
                    f"/ month on the fitted line."
                )
            else:
                slope_msg = (
                    f"Slope is negative ({slope:+.2f}) — wider "
                    f"distribution in this slice is associated "
                    f"with **lower** throughput per outlet (mass-"
                    f"distributed value SKUs vs niche high-pieces "
                    f"SKUs)."
                )
            st.markdown(slope_msg)

            # ---- Per-SKU slope contribution table ----
            # The single global slope hides which SKUs sit above /
            # below the line — i.e. who's punching above their
            # weight for the penetration they have. We surface a
            # "Throughput vs Expected" column = actual − fitted.
            reg_view = reg_matrix[
                [COL_BRAND, COL_SKU,
                 "Penetration %", "Throughput", "outlet_count"]
            ].copy()
            reg_view["Expected Throughput"] = (
                slope * reg_view["Penetration %"] + intercept
            ).round(1)
            reg_view["Residual (Actual − Expected)"] = (
                reg_view["Throughput"] - reg_view["Expected Throughput"]
            ).round(1)
            reg_view = reg_view.sort_values(
                "Residual (Actual − Expected)", ascending=False
            )

            # ---- Scatter + fitted line ----
            fit_x = np.array([x.min(), x.max()])
            fit_y = slope * fit_x + intercept
            fit_line = pd.DataFrame({
                "Penetration %": fit_x,
                "Throughput": fit_y
            })

            scatter_df = reg_matrix.copy()
            scatter_df["Hover SKU"] = (
                scatter_df[COL_BRAND].astype(str)
                + " · "
                + scatter_df[COL_SKU].astype(str)
            )

            reg_fig = px.scatter(
                scatter_df,
                x="Penetration %",
                y="Throughput",
                color=COL_BRAND,
                hover_name="Hover SKU",
                hover_data={
                    COL_BRAND: False,
                    "Penetration %": ":.1f",
                    "Throughput": ":.1f",
                    "outlet_count": ":,",
                },
                labels={"Throughput": "Throughput (units/outlet/mo)"}
            )
            reg_fig.update_traces(marker=dict(size=10))

            # Fitted line on top
            reg_fig.add_scatter(
                x=fit_line["Penetration %"],
                y=fit_line["Throughput"],
                mode="lines",
                line=dict(color="#FFD24C", width=3, dash="solid"),
                name=(
                    f"OLS fit  (y = {slope:.2f}·x "
                    f"{'+' if intercept >= 0 else '−'} "
                    f"{abs(intercept):.1f},  R²={r2:.2f})"
                ),
                hoverinfo="skip"
            )

            reg_fig.update_layout(
                height=480,
                plot_bgcolor="#050816",
                paper_bgcolor="#050816",
                font=dict(color="white"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.1)",
                    zerolinecolor="rgba(255,255,255,0.2)"
                ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.1)",
                    zerolinecolor="rgba(255,255,255,0.2)"
                )
            )

            st.plotly_chart(
                reg_fig,
                use_container_width=True,
                key=f"{period_id}_pen_thr_regression"
            )

            with st.expander(
                "🔍 Residuals — who's above / below the fitted "
                "line",
                expanded=False
            ):
                st.caption(
                    "Positive residual = SKU throughput beats the "
                    "fitted expectation for its penetration "
                    "(punching above its weight). Negative = "
                    "under-performing relative to the line."
                )
                st.dataframe(
                    reg_view.style.format({
                        "Penetration %": "{:.1f}",
                        "Throughput": "{:.1f}",
                        "Expected Throughput": "{:.1f}",
                        "Residual (Actual − Expected)": "{:+.1f}",
                        "outlet_count": "{:,}"
                    }).background_gradient(
                        subset=["Residual (Actual − Expected)"],
                        cmap="RdYlGn"
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=min(
                        500, 60 + 36 * len(reg_view)
                    )
                )

        st.divider()

        # ----- Heatmap segment selector -----
        # The heatmaps are the only insight tables left on this tab.
        # Pick the segment dimension first since the heatmap and the
        # categorical-sheet builder both share it.

        st.subheader("🎯 Segment dimension")
        st.caption(
            "Pick the dimension to slice SKUs against. Used by "
            "both the heatmaps below and the Categorical Sheet."
        )

        seg_pick_c1, seg_pick_c2 = st.columns([1, 2])

        with seg_pick_c1:
            # Candidate dimensions filtered to those present in the
            # data, so a missing column (e.g. no RE / no VC_CAT)
            # doesn't break the Auto-Insights heatmap / categorical
            # sheet. Default order keeps RE first when present.
            _ai_seg_candidates = [
                COL_RE,
                COL_VC_CATEGORY,
                COL_REGION_CAT,
                COL_REGION,
                COL_CHANNEL,
                COL_PCTYPE,
                COL_VC,
            ]
            _ai_seg_options = [
                c for c in _ai_seg_candidates
                if c in curated_src.columns
            ]
            # Brand is always materialised by the loader, so it's a
            # safe last-resort segment dimension when the file
            # carries none of the standard ones — the heatmap /
            # categorical sheet then slice SKUs by Brand rather than
            # disappearing.
            if COL_BRAND in curated_src.columns and COL_BRAND not in _ai_seg_options:
                _ai_seg_options.append(COL_BRAND)
            if not _ai_seg_options:
                st.info(
                    "No segment dimension (RE, VC_CAT, Region, "
                    "Channel, PC Type, VC_MODEL, Brand) is present "
                    "in the uploaded file, so the heatmap and "
                    "categorical sheet can't be built. Everything "
                    "else on this tab still works."
                )
                seg_choice = None
            else:
                seg_choice = st.selectbox(
                    "Segment dimension",
                    options=_ai_seg_options,
                    format_func=display_name,
                    key=f"{period_id}_seg_dim"
                )

        # Sub-filter: pick which values of the selected dimension
        # to keep in the segment analysis (e.g. only specific RE
        # channels). Empty = include all.
        if seg_choice is None:
            seg_value_options = []
        else:
            seg_value_options = sorted(
                curated_src[seg_choice]
                .dropna()
                .astype(str)
                .unique()
            )

        with seg_pick_c2:
            if seg_choice is not None:
                seg_value_filter = st.multiselect(
                    f"Restrict {display_name(seg_choice)} values",
                    options=seg_value_options,
                    default=[],
                    key=f"{period_id}_seg_value_filter",
                    help=(
                        "Limit the analysis to specific values of "
                        "the chosen dimension. Empty = all values."
                    )
                )
            else:
                seg_value_filter = []

        seg_src = curated_src.copy()
        if seg_choice is not None and seg_value_filter:
            seg_src = seg_src[
                seg_src[seg_choice]
                .astype(str)
                .isin(seg_value_filter)
            ]

        st.divider()

        # ----- Heatmap visual: SKU x Segment metric -----

        st.subheader("🌡️ SKU × Segment Heatmap")
        st.caption(
            "Seven lenses on the same SKU × Segment matrix. "
            "Pick the metric that answers your question. "
            "**Green = high / good, Red = low / bad.**"
        )

        heat_c1, heat_c2 = st.columns([2, 1])

        with heat_c1:
            metric_keys = list(HEATMAP_METRICS.keys())
            metric_labels = [
                HEATMAP_METRICS[k]["label"] for k in metric_keys
            ]
            metric_choice_label = st.selectbox(
                "Metric to plot",
                options=metric_labels,
                index=0,
                key=f"{period_id}_heat_metric",
                help=(
                    "Lift = relative; Throughput = absolute "
                    "pieces; Penetration = distribution density; "
                    "Pen Lift = where it's stocked unusually; "
                    "Throughput Δ = segment momentum; "
                    "Pen Impact = historical slope of throughput "
                    "on penetration within each segment; "
                    "Incremental Sales Lift = historical ₹ "
                    "associated with +1pp penetration "
                    "(N_seg × b / 100 × MRP); "
                    "Realistic Incremental TP per New Outlet = "
                    "practical extra units / month from adding "
                    "ONE new outlet, after a saturation-aware "
                    "multiplier (< 1)."
                )
            )
            metric_key = metric_keys[
                metric_labels.index(metric_choice_label)
            ]

        with heat_c2:
            heat_top_n = st.slider(
                "Top SKUs (by national pieces)",
                min_value=10,
                max_value=60,
                value=25,
                step=5,
                key=f"{period_id}_heat_top_n"
            )

        # ----- Optional secondary breakdown -----
        # When set, each primary-segment column on the heatmap is
        # sub-divided into one cell per value of this dimension.
        # Default = "None" → heatmap renders exactly as before.
        # Excludes the primary dimension itself.
        sub_seg_options = [
            d for d in [
                COL_RE, COL_VC_CATEGORY, COL_REGION_CAT,
                COL_REGION, COL_CHANNEL, COL_PCTYPE, COL_VC
            ]
            if d != seg_choice
        ]
        sub_c1, sub_c2 = st.columns([1, 2])
        with sub_c1:
            sub_seg_label = st.selectbox(
                "Secondary breakdown (optional)",
                options=["— None —"] + [
                    display_name(d) for d in sub_seg_options
                ],
                index=0,
                key=f"{period_id}_heat_sub_seg",
                help=(
                    "Optional: split each primary-segment column "
                    "into sub-cells by a second dimension. "
                    "Leave on 'None' for the standard view."
                )
            )

        sub_seg_choice = None
        if sub_seg_label != "— None —":
            # Map back from display label to underlying column name
            for d in sub_seg_options:
                if display_name(d) == sub_seg_label:
                    sub_seg_choice = d
                    break

        # Optional restrict-values for the secondary dimension —
        # only shown once a sub-segment is picked.
        sub_seg_value_filter = []
        if sub_seg_choice is not None:
            sub_seg_value_options = sorted(
                seg_src[sub_seg_choice]
                .dropna()
                .astype(str)
                .unique()
            )
            with sub_c2:
                sub_seg_value_filter = st.multiselect(
                    f"Restrict {display_name(sub_seg_choice)} "
                    f"values",
                    options=sub_seg_value_options,
                    default=[],
                    key=f"{period_id}_heat_sub_seg_values",
                    help=(
                        "Limit the secondary breakdown to "
                        "specific values. Empty = all values."
                    )
                )

        # Apply the secondary value filter to the heatmap source
        # (does not affect anything else on the tab — only the
        # heatmap below).
        heat_src = seg_src
        if sub_seg_choice is not None and sub_seg_value_filter:
            heat_src = seg_src[
                seg_src[sub_seg_choice]
                .astype(str)
                .isin(sub_seg_value_filter)
            ]

        metric_meta = HEATMAP_METRICS[metric_key]

        st.caption(metric_meta["caption"])

        # For the ₹-opportunity metric, surface whether MRP data
        # is actually feeding the calculation. Without an MRP file
        # the heatmap silently falls back to opportunity *volume*,
        # which has different units and a very different scale —
        # the caller should know.
        if metric_key == "opportunity_value":
            if _mrp_available:
                st.caption(
                    f"💰 Using uploaded MRP file — cells are ₹ "
                    f"per outlet-month at +1pp penetration. "
                    f"({_mrp_status})"
                )
            else:
                st.caption(
                    "ℹ️ No MRP file loaded — showing opportunity "
                    "**pieces** (units/month) instead of ₹. "
                    "Upload an MRP master in the sidebar to "
                    "convert to rupees."
                )

        heat_df = insight_heatmap(
            heat_src,
            seg_choice,
            period_col,
            period_months,
            top_skus=heat_top_n,
            min_seg_outlets=max(min_outlets_floor // 2, 25),
            metric=metric_key,
            sub_segment_col=sub_seg_choice
        )

        # SKU Type (Cooler / Ambient) visualization filter — hide
        # rows that don't match the sidebar radio. Cell values
        # themselves are computed over the full universe by
        # insight_heatmap above; we only drop rows here, so
        # segment totals and metric normalisation stay unbiased.
        if not heat_df.empty and sku_type_filter != "All":
            _heat_brand_lookup = (
                heat_src.dropna(subset=[COL_SKU])
                .drop_duplicates(subset=[COL_SKU])
                .set_index(COL_SKU)[COL_BRAND]
                .astype(str).str.strip().str.lower()
                .to_dict()
            )
            if sku_type_filter == "Cooler":
                _keep_rows = [
                    s for s in heat_df.index
                    if _heat_brand_lookup.get(s, "") in COOLER_BRANDS
                ]
            else:  # Ambient
                _keep_rows = [
                    s for s in heat_df.index
                    if _heat_brand_lookup.get(s, "") not in COOLER_BRANDS
                ]
            heat_df = heat_df.loc[_keep_rows]

        if heat_df.empty:
            if sku_type_filter != "All":
                st.info(
                    f"No **{sku_type_filter}** SKUs in the current "
                    f"heatmap slice. Switch the **SKU Type** "
                    f"sidebar radio to **All** to see every SKU."
                )
            else:
                st.info(
                    "Not enough data to build heatmap with the "
                    "current curation and metric."
                )
        else:
            # Sort rows: group by SKU tier first (Tier 1 on top,
            # Tier 2 below, etc.), then by greenness within each
            # tier (greenest SKU at the top of its tier block,
            # reddest at the bottom). Must happen BEFORE we rename
            # the columns (the helper reads numeric cell values,
            # column names don't matter, but it's clearer to sort
            # first).
            ui_tier_map = _build_sku_tier_map(heat_src)
            heat_df = _sort_heatmap_rows_by_greenness(
                heat_df, metric_key,
                sku_tier_map=ui_tier_map
            )

            # Append the bottom TOTAL row (sum for additive metrics,
            # mean for ratio/per-outlet metrics). Done after the
            # greenness sort so the total stays anchored at the
            # bottom, and before the (n=…) column rename so column
            # alignment with the source data is still intact.
            heat_df = _append_heatmap_total_row(heat_df, metric_key)

            # Whether the heatmap has a 2-level column index
            # (primary × secondary). Drives column-label formatting
            # and primary-group separator overlays below.
            has_sub = isinstance(
                heat_df.columns, pd.MultiIndex
            )

            if has_sub:
                # Column labels: "<primary> | <secondary>\n(n=...)"
                # where n = outlets in that (primary, secondary)
                # cell of the heatmap source.
                cell_counts = (
                    heat_src.groupby(
                        [seg_choice, sub_seg_choice]
                    )[COL_OUTLET]
                    .nunique()
                    .to_dict()
                )
                new_cols = []
                primary_seq = []  # ordered primary value per col
                for prim, sub in heat_df.columns:
                    n = cell_counts.get((prim, sub), 0)
                    new_cols.append(
                        f"{prim} | {sub}\n(n={n:,})"
                    )
                    primary_seq.append(prim)
                heat_df = pd.DataFrame(
                    heat_df.values,
                    index=heat_df.index,
                    columns=new_cols
                )
            else:
                # Annotate each column header with the total outlets
                # in that segment value, using the heatmap source.
                seg_counts = (
                    heat_src.groupby(seg_choice)[COL_OUTLET]
                    .nunique()
                    .to_dict()
                )
                heat_df = heat_df.rename(
                    columns={
                        c: f"{c}\n(n={seg_counts.get(c, 0):,})"
                        for c in heat_df.columns
                    }
                )
                primary_seq = None

            imshow_kwargs = dict(
                color_continuous_scale=metric_meta["colorscale"],
                aspect="auto",
                labels=dict(
                    x=(
                        f"{display_name(seg_choice)} × "
                        f"{display_name(sub_seg_choice)}"
                        if has_sub else seg_choice
                    ),
                    y="SKU",
                    color=metric_meta["colorbar"]
                ),
                text_auto=metric_meta["fmt"]
            )

            # ----- Dynamic colour range so red is actually visible -----
            # Goal (per user spec): the lowest data points should
            # actually paint near the BOTTOM of the colour scale
            # (red), not get squashed into the green/yellow band.
            #
            # Strategy:
            #   • Anchor the bottom of the scale at the data MIN
            #     (so the weakest cells reach the deepest red).
            #   • Anchor the top of the scale at P90 of the data
            #     (clamps a few outliers from washing the rest into
            #     the green band — they still paint the top colour).
            #   • For diverging metrics (midpoint = 1.0 or 0.0): if
            #     all data sits above the midpoint, force the bottom
            #     of the range to be BELOW the midpoint by enough
            #     that the red half of the diverging palette is
            #     still used. Otherwise just use (min, P90) so the
            #     midpoint sits naturally between them.
            heat_vals = (
                heat_df.to_numpy(dtype=float).ravel()
            )
            heat_vals = heat_vals[~np.isnan(heat_vals)]

            if heat_vals.size > 0:
                v_min = float(heat_vals.min())
                v_max = float(heat_vals.max())
                p90 = float(np.quantile(heat_vals, 0.90))

                if metric_meta["midpoint"] is not None:
                    mid = float(metric_meta["midpoint"])
                    # Top: at least P90, but never below the midpoint
                    # (otherwise nothing turns green).
                    hi = max(p90, mid + 1e-6)
                    # Bottom: data min, but if everything is above
                    # the midpoint pull the floor below the midpoint
                    # so the red half of the diverging scale is used.
                    if v_min >= mid:
                        lo = mid - max((hi - mid) * 0.6, 0.2)
                    else:
                        lo = v_min
                    imshow_kwargs["range_color"] = (lo, hi)
                    imshow_kwargs["color_continuous_midpoint"] = mid
                else:
                    # Sequential — bottom = data min, top = P90.
                    lo = v_min
                    hi = p90
                    if hi - lo < 1e-6:
                        # Degenerate (all same value) — fall back
                        hi = v_max + 1e-6
                    imshow_kwargs["range_color"] = (lo, hi)
            elif metric_meta["midpoint"] is not None:
                imshow_kwargs["color_continuous_midpoint"] = (
                    metric_meta["midpoint"]
                )

            heat_fig = px.imshow(heat_df, **imshow_kwargs)

            heat_fig.update_layout(
                height=max(450, 22 * len(heat_df)),
                plot_bgcolor="#050816",
                paper_bgcolor="#050816",
                font=dict(color="white"),
                coloraxis_colorbar=dict(
                    title=metric_meta["colorbar"]
                ),
                # Extra top margin so the column tick labels —
                # which now live at the top of the matrix — have
                # room and aren't clipped under the colour bar.
                margin=dict(t=80)
            )

            # Column labels at the TOP of the heatmap (not bottom).
            # Easier to scan when reading the matrix top-down.
            heat_fig.update_xaxes(side="top")

            # When a secondary breakdown is active, overlay
            # vertical separators between primary-segment groups
            # and add a top annotation showing each primary value
            # spanning its sub-cells. This is what gives the
            # subdivided cells their grouped look.
            if has_sub and primary_seq:
                n_cols = len(primary_seq)
                # Compute the column-index span of each primary
                # value, in left-to-right order. Plotly imshow
                # places each cell at integer x-coordinates
                # (0, 1, …); group boundaries fall at x = k - 0.5.
                spans = []
                cur_val = primary_seq[0]
                cur_start = 0
                for i in range(1, n_cols):
                    if primary_seq[i] != cur_val:
                        spans.append((cur_val, cur_start, i - 1))
                        cur_val = primary_seq[i]
                        cur_start = i
                spans.append((cur_val, cur_start, n_cols - 1))

                # Vertical lines BETWEEN primary groups (not at
                # the chart edges). Drawn on top of the heatmap.
                shapes = list(heat_fig.layout.shapes or [])
                for _, _, end_idx in spans[:-1]:
                    shapes.append(dict(
                        type="line",
                        xref="x", yref="paper",
                        x0=end_idx + 0.5, x1=end_idx + 0.5,
                        y0=0, y1=1,
                        line=dict(
                            color="#ffffff", width=2
                        )
                    ))
                heat_fig.update_layout(shapes=shapes)

                # Primary-group labels along the top edge,
                # centred over each group's sub-cells. Sit *above*
                # the column-tick labels (which are now at the top
                # of the heatmap), so they aren't overlapped by
                # the per-column "<value>\n(n=…)" text.
                annotations = list(
                    heat_fig.layout.annotations or []
                )
                for val, s_idx, e_idx in spans:
                    annotations.append(dict(
                        x=(s_idx + e_idx) / 2.0,
                        y=1.0,
                        xref="x", yref="paper",
                        text=f"<b>{val}</b>",
                        showarrow=False,
                        yanchor="bottom",
                        yshift=50,
                        font=dict(color="#FFD166", size=12)
                    ))
                heat_fig.update_layout(annotations=annotations)
                # Give the title area a bit more headroom so the
                # group labels don't clip — and so the moved-to-top
                # tick labels have somewhere to live.
                heat_fig.update_layout(
                    margin=dict(t=110)
                )

            st.plotly_chart(
                heat_fig,
                use_container_width=True,
                key=f"{period_id}_heatmap_{metric_key}"
            )

        st.divider()

        # ----- 4. Seasonal trend line chart -----

        st.subheader("🌊 Seasonal Trend  —  Monthly Pieces by SKU")
        st.caption(
            "Pick one or more SKUs to see how their pieces "
            "moves month-by-month across the 15-month window. "
            "Spikes around festival months (Oct–Nov for Diwali, "
            "Feb for Valentine, Apr for Easter, Aug for Rakhi) "
            "are easiest to read in `Per Active Outlet` mode "
            "since it strips out distribution growth."
        )

        # SKU dropdown — sensible default is top 3 by L15M
        # volume from the curated set
        sku_volume_rank = (
            curated_src
            .groupby(COL_SKU)["L15M"]
            .sum()
            .sort_values(ascending=False)
        )

        sku_options_for_trend = sku_volume_rank.index.tolist()

        # SKU Type (Cooler / Ambient) visualization filter — restrict
        # the options list in the dropdown. The trend math itself
        # uses curated_src (unfiltered by SKU type) for whichever
        # SKUs the user actually selects, so per-SKU numbers stay
        # identical to the "All" view.
        if sku_type_filter != "All":
            _trend_brand_lookup = (
                curated_src.dropna(subset=[COL_SKU])
                .drop_duplicates(subset=[COL_SKU])
                .set_index(COL_SKU)[COL_BRAND]
                .astype(str).str.strip().str.lower()
                .to_dict()
            )
            if sku_type_filter == "Cooler":
                sku_options_for_trend = [
                    s for s in sku_options_for_trend
                    if _trend_brand_lookup.get(s, "") in COOLER_BRANDS
                ]
            else:  # Ambient
                sku_options_for_trend = [
                    s for s in sku_options_for_trend
                    if _trend_brand_lookup.get(s, "") not in COOLER_BRANDS
                ]

        default_skus = sku_options_for_trend[:3]

        trend_c1, trend_c2 = st.columns([3, 1])

        with trend_c1:
            # The widget key embeds the active SKU Type so flipping
            # the sidebar radio resets the selection — otherwise a
            # previously-selected Cooler SKU would remain stored
            # after switching to Ambient and Streamlit would raise
            # "default not in options".
            chosen_skus = st.multiselect(
                "SKUs to plot",
                options=sku_options_for_trend,
                default=default_skus,
                key=f"{period_id}_trend_skus__{sku_type_filter}"
            )

        with trend_c2:
            trend_mode_label = st.radio(
                "Y-axis",
                options=[
                    "Per Active Outlet",
                    "Total Pieces"
                ],
                index=0,
                key=f"{period_id}_trend_mode"
            )

        trend_mode = (
            "per_outlet"
            if trend_mode_label == "Per Active Outlet"
            else "total"
        )

        if not chosen_skus:
            st.info("Select at least one SKU to draw the chart.")
        else:

            trend_df = build_seasonal_trend(
                curated_src,
                chosen_skus,
                mode=trend_mode
            )

            if trend_df.empty:
                st.info(
                    "No monthly data available for the selected "
                    "SKUs in the curated slice."
                )
            else:

                trend_fig = px.line(
                    trend_df,
                    x="Month",
                    y="Volume",
                    color=COL_SKU,
                    markers=True,
                    labels={
                        "Volume": (
                            "Pieces / Active Outlet"
                            if trend_mode == "per_outlet"
                            else "Total Pieces"
                        )
                    }
                )

                trend_fig.update_layout(
                    height=500,
                    plot_bgcolor="#050816",
                    paper_bgcolor="#050816",
                    font=dict(color="white"),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified"
                )

                # Light vertical guides for the festival months.
                # Plotly's add_vline doesn't accept categorical x
                # values, so we draw a paper-referenced shape and
                # an annotation manually instead.
                festival_marks = {
                    "Oct25": "Diwali (approx)",
                    "Feb26": "Valentine"
                }

                months_present = (
                    trend_df["Month"].astype(str).unique().tolist()
                )

                for m, label in festival_marks.items():
                    if m not in months_present:
                        continue

                    trend_fig.add_shape(
                        type="line",
                        x0=m,
                        x1=m,
                        xref="x",
                        y0=0,
                        y1=1,
                        yref="paper",
                        line=dict(
                            color="rgba(255,255,255,0.25)",
                            dash="dot",
                            width=1
                        )
                    )

                    trend_fig.add_annotation(
                        x=m,
                        y=1.02,
                        xref="x",
                        yref="paper",
                        text=label,
                        showarrow=False,
                        font=dict(
                            color="rgba(255,255,255,0.6)",
                            size=11
                        ),
                        xanchor="center"
                    )

                st.plotly_chart(
                    trend_fig,
                    use_container_width=True,
                    key=f"{period_id}_trend_chart"
                )

        st.divider()

        # =================================================
        # CATEGORICAL SHEET  —  per-SKU heatmap by selected segment
        # =================================================
        # User picks: (a) the metric (Throughput or Penetration%),
        # (b) the SKUs to exclude (defaults to none, i.e. every
        # qualifying SKU is included on first render). The
        # currently-selected segment dimension (`seg_choice`) is
        # reused, with the optional value-restrict already applied
        # (`seg_src`).
        # The result is rendered inline AND attached to the
        # downloadable report below.
        # =================================================

        st.subheader("📊 Categorical Sheet — SKU × Segment")
        st.caption(
            f"All qualifying SKUs are included by default — exclude "
            f"any you want to drop, then pick a metric to get a SKU × "
            f"`{display_name(seg_choice)}` grid showing throughput "
            f"or penetration in each segment value, plus a Pen% "
            f"Range column. Will be attached to the report download."
        )

        # ----- Categorical-sheet-only filters -----
        # These narrow the SKU-pool used by the categorical sheet
        # WITHOUT touching the rest of the tab. Useful when the
        # field team wants e.g. "only Premium tier coolers".
        cat_f1, cat_f2 = st.columns([1, 1])

        with cat_f1:
            cat_tier_options = sorted(
                curated_src[COL_SKU_TIER]
                .dropna().astype(str).unique()
            )
            cat_tier_filter = st.multiselect(
                "SKU Tier (categorical sheet only)",
                options=cat_tier_options,
                default=[],
                key=f"{period_id}_cat_tier_filter",
                help=(
                    "Restrict the categorical sheet to specific "
                    "SKU tiers. Empty = all tiers. Does not "
                    "affect heatmaps or other tab sections."
                )
            )

        with cat_f2:
            cat_sku_type_filter = st.selectbox(
                "SKU Type (categorical sheet only)",
                options=["All", "Cooler", "Ambient"],
                index=0,
                key=f"{period_id}_cat_sku_type_filter",
                help=(
                    "Visualization-only filter. Cooler = Silk + "
                    "Bournville + Temptations. Ambient = rest of "
                    "portfolio. Selecting Cooler or Ambient only "
                    "hides the non-matching SKU rows from the "
                    "table — it does NOT change the underlying "
                    "outlet universe or segment denominators."
                )
            )

        # Build the SKU-restricted slice for the categorical sheet.
        # NOTE: SKU Type is a VISUALIZATION-ONLY filter; it is
        # applied to the displayed SKU pool (cat_sku_options) below,
        # NOT to cat_seg_src. That keeps segment outlet counts and
        # penetration denominators identical to the "All" case.
        cat_seg_src = seg_src.copy()
        if cat_tier_filter:
            cat_seg_src = cat_seg_src[
                cat_seg_src[COL_SKU_TIER]
                .astype(str).isin(cat_tier_filter)
            ]

        cat_c1, cat_c2 = st.columns([2, 1])

        with cat_c1:
            # SKU options ranked by L15M volume in the
            # categorical-sheet-filtered slice so the most-relevant
            # SKUs appear first.
            cat_sku_options = (
                cat_seg_src.groupby(COL_SKU)["L15M"]
                .sum()
                .sort_values(ascending=False)
                .index.tolist()
            )

            # Apply the Cooler / Ambient visualization filter to the
            # SKU pool itself. We resolve each SKU's brand from
            # cat_seg_src (the full tier-filtered slice) so the
            # classification is independent of how the SKU was
            # ranked above.
            if cat_sku_type_filter != "All":
                _sku_brand_lookup = (
                    cat_seg_src
                    .dropna(subset=[COL_SKU])
                    .drop_duplicates(subset=[COL_SKU])
                    .set_index(COL_SKU)[COL_BRAND]
                    .astype(str).str.strip().str.lower()
                    .to_dict()
                )
                if cat_sku_type_filter == "Cooler":
                    cat_sku_options = [
                        s for s in cat_sku_options
                        if _sku_brand_lookup.get(s, "") in COOLER_BRANDS
                    ]
                else:  # Ambient
                    cat_sku_options = [
                        s for s in cat_sku_options
                        if _sku_brand_lookup.get(s, "") not in COOLER_BRANDS
                    ]
            # Default behaviour: nothing is excluded, so the
            # categorical sheet starts out covering EVERY SKU in
            # the current slice. The user explicitly drops SKUs
            # they don't want — this is the inverse of the old
            # "pick SKUs to include" flow and avoids forcing the
            # field team to re-tick every SKU when they only want
            # to hide one or two.
            #
            # The multiselect's `key` embeds both the SKU Type and
            # the Tier filter signature. When either changes,
            # Streamlit treats it as a fresh widget and falls back
            # to `default=[]`, so changing SKU Type/Tier wipes any
            # stale exclusions that may no longer apply to the new
            # SKU pool.
            cat_tier_sig = ",".join(sorted(cat_tier_filter)) or "none"
            cat_excl_key = (
                f"{period_id}_cat_excluded_skus"
                f"__{cat_sku_type_filter}"
                f"__{cat_tier_sig}"
            )

            cat_excluded_skus = st.multiselect(
                "SKUs to exclude",
                options=cat_sku_options,
                default=[],
                key=cat_excl_key,
                help=(
                    "Leave empty to include every qualifying SKU "
                    "(the default). Add SKUs here to drop them "
                    "from the categorical sheet. Changing SKU "
                    "Type or Tier clears the exclusion list."
                )
            )

            # Preserve the original L15M-volume ordering of the
            # included set so downstream rendering stays stable.
            excluded_set = set(cat_excluded_skus)
            cat_skus = [
                s for s in cat_sku_options if s not in excluded_set
            ]

        with cat_c2:
            cat_metric_label = st.selectbox(
                "Metric",
                options=["Throughput", "Penetration %"],
                index=0,
                key=f"{period_id}_cat_metric",
                help=(
                    "Throughput = monthly units per active outlet "
                    "in the segment. Penetration % = share of "
                    "segment outlets stocking the SKU."
                )
            )
        cat_metric = (
            "throughput"
            if cat_metric_label == "Throughput"
            else "penetration"
        )

        if cat_skus:
            categorical_block = build_categorical_sku_block(
                cat_seg_src,
                seg_choice,
                cat_skus,
                period_col,
                period_months,
                metric=cat_metric
            )
            cat_table = categorical_block.get(
                "table", pd.DataFrame()
            )

            if cat_table.empty:
                st.info(
                    "No data for the selected SKUs in this "
                    "segment slice."
                )
            else:
                # Render the table with a clean, readable
                # green / neutral / red colour scheme per row.
                # Each row is colour-coded independently against
                # its own median so high-volume SKUs don't all
                # turn green and low-volume SKUs all turn red.
                # Values >= row median go green, < row median go
                # red. Intensity scales with distance from median
                # (clamped to P10/P90 so outliers don't wash the
                # rest out). Text colour stays dark so numbers
                # remain readable on every shade.
                num_cols = [
                    c for c in cat_table.columns
                    if c not in [COL_SKU, "Pen% Range"]
                ]

                def _row_gradient(row):
                    vals = pd.to_numeric(
                        row[num_cols], errors="coerce"
                    ).dropna()
                    if vals.empty:
                        return [""] * len(row)

                    median_v = float(vals.median())
                    p10 = float(vals.quantile(0.10))
                    p90 = float(vals.quantile(0.90))

                    # Spread on each side of the median, used to
                    # normalise distance into a 0..1 intensity.
                    low_span = max(median_v - p10, 1e-9)
                    high_span = max(p90 - median_v, 1e-9)

                    # Solid, readable greens / reds. Light shade
                    # near the median, deeper shade at the tails.
                    # Text stays dark slate for full contrast.
                    GREEN_LIGHT = (220, 245, 220)   # near median
                    GREEN_DEEP  = (102, 187, 106)   # >> median
                    RED_LIGHT   = (252, 224, 224)   # near median
                    RED_DEEP    = (229, 115, 115)   # << median
                    TEXT_COLOR  = "#1a1a1a"

                    def _blend(c_light, c_deep, t):
                        # t in [0,1]; 0 = light, 1 = deep
                        t = max(0.0, min(1.0, t))
                        return tuple(
                            int(round(c_light[i] +
                                      (c_deep[i] - c_light[i]) * t))
                            for i in range(3)
                        )

                    styles = []
                    for col in row.index:
                        if col not in num_cols:
                            styles.append("")
                            continue
                        v = row[col]
                        if pd.isna(v):
                            styles.append("")
                            continue
                        v_f = float(v)
                        if v_f >= median_v:
                            # 0 at median → 1 at p90 or beyond
                            t = (v_f - median_v) / high_span
                            r, g, b = _blend(
                                GREEN_LIGHT, GREEN_DEEP, t
                            )
                        else:
                            # 0 at median → 1 at p10 or below
                            t = (median_v - v_f) / low_span
                            r, g, b = _blend(
                                RED_LIGHT, RED_DEEP, t
                            )
                        styles.append(
                            f"background-color: rgb({r},{g},{b}); "
                            f"color: {TEXT_COLOR}; "
                            f"font-weight: 500;"
                        )
                    return styles

                styled = (
                    cat_table.style
                    .apply(_row_gradient, axis=1)
                    .format(
                        {c: "{:.2f}" for c in num_cols}
                    )
                )
                st.dataframe(
                    styled,
                    use_container_width=True,
                    hide_index=True,
                    height=min(700, 60 + 38 * len(cat_table))
                )
                st.caption(categorical_block["caption"])
        else:
            categorical_block = None
            st.info(
                "Pick at least one SKU to build the categorical "
                "sheet."
            )

        # =================================================
        # POPULATE THE ZSM ONE-PAGER DOWNLOAD BUTTON
        # =================================================

        with report_placeholder:

            # ----- Universal filters (sidebar selections) -----
            universal_bits = []
            if sku_type_filter and sku_type_filter != "All":
                universal_bits.append(
                    f"SKU Type={sku_type_filter}"
                )
            sidebar_filter_summary = {
                "ASM": asm_filter,
                "Channel": channel_filter,
                "PC Type": pctype_filter,
                "RD": rd_filter,
                "RE": re_filter,
                "Region": region_filter,
                "Region_Cat": region_cat_filter,
                "Status": status_filter,
                "VC": vc_filter,
                "VC_Cat": vc_category_filter,
                "SE_TTY": setty_filter,
                "Brand": brand_visual_filter,
                "SKU_Tier": sku_tier_visual_filter,
                "SKU": sku_visual_filter,
            }
            for label, vals in sidebar_filter_summary.items():
                if vals:
                    if len(vals) <= 3:
                        universal_bits.append(
                            f"{label}={','.join(vals)}"
                        )
                    else:
                        universal_bits.append(
                            f"{label}=[{len(vals)} values]"
                        )
            if exclude_skus:
                universal_bits.append(
                    f"Excluded SKUs=[{len(exclude_skus)}]"
                )
            if smoothing_range != (0, 100):
                _basis_short = (
                    "value"
                    if (
                        smoothing_basis.startswith("Sale Value")
                        and _mrp_available
                    )
                    else "pieces"
                )
                universal_bits.append(
                    f"Smoothing P{smoothing_range[0]}–"
                    f"P{smoothing_range[1]} ({_basis_short})"
                )

            universal_filters_str = "  ·  ".join(universal_bits)

            # ----- Local curation (Auto-Insights tab) -----
            local_bits = []
            local_bits.append(
                f"Status={','.join(status_include) or 'All'}"
            )
            local_bits.append(
                f"Min outlets≥{min_outlets_floor}"
            )
            if vc_model_exclude:
                local_bits.append(
                    f"Excl VC={','.join(vc_model_exclude)}"
                )
            if tier_include:
                local_bits.append(
                    f"Tiers={','.join(tier_include)}"
                )
            if manual_exclude:
                local_bits.append(
                    f"Manually excl=[{len(manual_exclude)} SKUs]"
                )
            if seg_value_filter:
                local_bits.append(
                    f"{seg_choice}∈{','.join(seg_value_filter)}"
                )
            local_filters_str = "  ·  ".join(local_bits)

            subset_metrics_str = (
                f"{selling_outlets_count:,} selling outlets · "
                f"{full_matrix.shape[0]:,} SKUs · "
                f"{filtered_df[COL_OUTLET].nunique():,} qualifying "
                f"outlets (universal)"
            )

            # Pre-compute Overview-style metrics for the report,
            # using the curated source so the report matches the
            # subset on this tab.
            total_base_pdf = filtered_df[COL_OUTLET].nunique()
            sell_outlets_pdf = curated_outlet_sku[COL_OUTLET].nunique()
            total_vol_pdf = curated_outlet_sku["pieces_sold"].sum()
            ov_metrics = {
                "total_base":      total_base_pdf,
                "selling_outlets": sell_outlets_pdf,
                "penetration_pct": (
                    sell_outlets_pdf / total_base_pdf * 100
                    if total_base_pdf > 0 else 0
                ),
                "throughput": (
                    total_vol_pdf / sell_outlets_pdf / period_months
                    if sell_outlets_pdf > 0 else 0
                ),
            }

            # Merge Penetration Impact into the matrix the report
            # uses, then compute Opportunity Volume from it. (Δ
            # and Velocity columns are no longer shown per user
            # spec.)
            full_matrix_for_report = full_matrix.copy()
            if not full_matrix_for_report.empty:
                cur_pen_impact_df = compute_sku_penetration_impact(
                    curated_src
                )
                if not cur_pen_impact_df.empty:
                    full_matrix_for_report = (
                        full_matrix_for_report.merge(
                            cur_pen_impact_df,
                            on=[COL_BRAND, COL_SKU],
                            how="left"
                        )
                    )

                # Opportunity Volume = N × (T + (P+1)·b) / 100
                # N here is the outlet universe used to build
                # `full_matrix` (curated_outlet_sku), so that the
                # Penetration % column and the Opportunity Volume
                # column share a consistent denominator and the
                # numbers reconcile if the user back-solves them.
                if ("Penetration Impact" in
                        full_matrix_for_report.columns
                        and "Penetration %" in
                        full_matrix_for_report.columns
                        and "Throughput" in
                        full_matrix_for_report.columns):
                    total_outlets_report = (
                        curated_outlet_sku[COL_OUTLET].nunique()
                        if not curated_outlet_sku.empty else 0
                    )
                    full_matrix_for_report["Opportunity Pieces"] = (
                        total_outlets_report
                        * (
                            full_matrix_for_report["Throughput"]
                            + (full_matrix_for_report["Penetration %"]
                               + 1)
                            * full_matrix_for_report[
                                "Penetration Impact"
                            ]
                        )
                        / 100.0
                    )
                    full_matrix_for_report["Opportunity Pieces"] = (
                        full_matrix_for_report["Opportunity Pieces"]
                        .round(0)
                    )

                # Append the practical "per new outlet" columns
                # so the printed SKU Performance Table tells both
                # halves of the story (historical + realistic).
                full_matrix_for_report = (
                    compute_realistic_incremental_per_outlet(
                        full_matrix_for_report
                    )
                )

            pdf_key  = f"{period_id}_pdf_bytes"
            html_key = f"{period_id}_html_bytes"
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M")

            st.markdown(
                "📄 **ZSM Comprehensive Report** — packages the "
                "Overview, Quadrant Analysis, SKU Performance "
                "Table (with Penetration Impact), all "
                "Heatmaps, and the Categorical Sheet (your SKU "
                "selection) into a print-ready PDF."
            )

            gen_c1, gen_c2 = st.columns([2, 3])

            with gen_c1:
                generate_clicked = st.button(
                    "🛠  Generate Report",
                    key=f"{period_id}_gen_report",
                    type="primary",
                    use_container_width=True,
                    help=(
                        f"Building the report renders "
                        f"{len(HEATMAP_METRICS)} heatmaps as "
                        f"images — takes ~10 s."
                    )
                )

            if generate_clicked:

                with st.spinner(
                    f"Rendering {len(HEATMAP_METRICS)} heatmaps + "
                    f"report tables…"
                ):

                    # Build the canonical SKU order ONCE for this
                    # report so every heatmap panel keeps each SKU
                    # in the same row. Order: Tier 1 first, then
                    # Tier 2, …, unknown tier last; within each
                    # tier, descending by L15M national volume.
                    canonical_sku_order = _build_canonical_sku_order(
                        seg_src, top_skus=heat_top_n
                    )

                    # Build the heatmap PNGs (one per metric).
                    heatmap_pngs = []
                    for mkey in HEATMAP_METRICS.keys():
                        png_bytes = _render_heatmap_png(
                            seg_src,
                            seg_choice,
                            period_col,
                            period_months,
                            mkey,
                            top_skus=heat_top_n,
                            min_seg_outlets=max(
                                min_outlets_floor // 2, 25
                            ),
                            sku_order=canonical_sku_order
                        )
                        heatmap_pngs.append(
                            (HEATMAP_METRICS[mkey]["label"], png_bytes)
                        )

                    pdf_bytes = _build_one_pager_pdf(
                        period_label,
                        universal_filters_str,
                        local_filters_str,
                        subset_metrics_str,
                        ov_metrics,
                        full_matrix_for_report,
                        heatmap_pngs,
                        categorical_block=categorical_block
                    )

                    if pdf_bytes is not None:
                        st.session_state[pdf_key] = pdf_bytes
                        st.session_state.pop(html_key, None)
                    else:
                        # reportlab missing — fall back to HTML
                        html_bytes = _build_one_pager_html(
                            period_label,
                            universal_filters_str,
                            local_filters_str,
                            subset_metrics_str,
                            ov_metrics,
                            full_matrix_for_report,
                            categorical_block=categorical_block
                        )
                        st.session_state[html_key] = html_bytes
                        st.session_state.pop(pdf_key, None)

                st.success("Report ready — download below.")

            # Download button(s) appear once a report has been
            # generated in this session.
            with gen_c2:
                if pdf_key in st.session_state:
                    st.download_button(
                        "⬇️  Download PDF",
                        data=st.session_state[pdf_key],
                        file_name=(
                            f"VC_SKU_Brief_{period_label}_{ts}.pdf"
                        ),
                        mime="application/pdf",
                        key=f"{period_id}_dl_pdf",
                        use_container_width=True
                    )
                elif html_key in st.session_state:
                    st.download_button(
                        "⬇️  Download HTML  (install reportlab for PDF)",
                        data=st.session_state[html_key],
                        file_name=(
                            f"VC_SKU_Brief_{period_label}_{ts}.html"
                        ),
                        mime="text/html",
                        key=f"{period_id}_dl_html",
                        use_container_width=True
                    )

def render_sku_priority_lister():
    """
    Self-contained tab that ranks SKUs by Throughput × Latest MRP ×
    Gross Margin, splits them into Top (30%) / Mid (30%) / Bottom
    (40%) buckets by cumulative-share, shows a summary header,
    a selectable seasonality chart, and five quarter-by-quarter
    comparison lists with Gainer / Loser / Stable / Massive Gainer
    / Massive Loser tags.

    Only rendered if the optional Gross Margin file has been
    uploaded — gated upstream by the `_gm_available` flag.
    """

    st.title("⭐ SKU Priority Lister")
    st.caption(
        "Rank SKUs by Throughput × Latest MRP × Gross Margin "
        "and split them into Top (30%) / Mid (30%) / Bottom (40%) "
        "tiers by cumulative share. The sidebar filters apply "
        "first; the tab-level filters below stack on top of them."
    )

    if df is None or df.empty:
        st.info("No data available. Please upload a source file.")
        return

    # The Priority Lister now starts from `filtered_df` (the
    # sidebar-filtered slice — universal multiselects + Exclude
    # SKU + Smoothing trim already applied). The Cooler / Ambient
    # sidebar radio is intentionally NOT inherited here because
    # this tab has its own SKU Type radio further down.
    if filtered_df is None or filtered_df.empty:
        st.warning(
            "Current sidebar filters return no rows. Relax the "
            "sidebar filters to use the SKU Priority Lister."
        )
        return

    # -------- Local filter widgets --------

    st.subheader("🎛️ Filters")

    # IMPORTANT: Tab-level filter widgets derive their option lists
    # from `filtered_df` (the sidebar-filtered slice), NOT from
    # the unfiltered `df`. This means the sidebar filters truly
    # cascade into the Priority Lister tab: if the sidebar
    # excludes a region/channel/VC_CAT, that value won't appear in
    # the corresponding tab-level multiselect either, so the user
    # can't accidentally pick a value the sidebar has already
    # filtered out (which would otherwise yield an empty result).
    _opt_src = filtered_df  # sidebar-filtered universe

    def _opts(col):
        """Sorted unique values of `col` in the sidebar-filtered
        slice. Returns empty list if column missing or all-null."""
        if col not in _opt_src.columns:
            return []
        return sorted(
            _opt_src[col].dropna().astype(str).unique()
        )

    # VC_CAT first — the headline filter, on its own row.
    vc_cat_opts = _opts(COL_VC_CATEGORY)
    vc_cat_opts_ordered = order_segment_values(
        COL_VC_CATEGORY, vc_cat_opts
    )

    vc_cat_sel = st.multiselect(
        "VC_CAT (volume capacity band)",
        options=vc_cat_opts_ordered,
        default=[],
        key="spl_vc_cat",
        help=(
            "Pick one or more VC_CAT bands. Empty = include all "
            "bands present in the sidebar-filtered slice. Options "
            "reflect the current sidebar filters."
        )
    )

    # Secondary filters — two rows of multiselects.
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        vc_model_sel = st.multiselect(
            "VC_MODEL",
            options=_opts(COL_VC),
            default=[],
            key="spl_vc_model"
        )
    with fc2:
        pc_type_sel = st.multiselect(
            "PC Type",
            options=_opts(COL_PCTYPE),
            default=[],
            key="spl_pctype"
        )
    with fc3:
        re_sel = st.multiselect(
            "RE",
            options=_opts(COL_RE),
            default=[],
            key="spl_re"
        )
    with fc4:
        channel_sel = st.multiselect(
            "CHANNEL",
            options=_opts(COL_CHANNEL),
            default=[],
            key="spl_channel"
        )

    fc5, fc6, fc7, _fc8 = st.columns(4)
    with fc5:
        region_sel = st.multiselect(
            "Region",
            options=_opts(COL_REGION),
            default=[],
            key="spl_region"
        )
    with fc6:
        asm_sel = st.multiselect(
            "ASM",
            options=_opts(COL_ASM),
            default=[],
            key="spl_asm"
        )
    with fc7:
        setty_sel = st.multiselect(
            "SE_TTY (SE Territory)",
            options=_opts(COL_SETTY),
            default=[],
            key="spl_setty"
        )

    # ---- SKU-level subsets (don't reshape totals, only ranking) ----
    sc1, sc2 = st.columns([2, 1])

    sku_tier_opts = _opts(COL_SKU_TIER)
    with sc1:
        sku_tier_sel = st.multiselect(
            "SKU Tiers (ranking scope)",
            options=sku_tier_opts,
            default=[],
            key="spl_sku_tier",
            help=(
                "Restrict the ranking to SKUs in these tiers. "
                "Empty = all tiers."
            )
        )

    with sc2:
        sku_type_local = st.radio(
            "SKU Type (ranking scope)",
            options=["All", "Cooler", "Ambient"],
            index=0,
            horizontal=True,
            key="spl_sku_type",
            help=(
                "Cooler = Silk + Bournville + Temptations. "
                "Ambient = everything else."
            )
        )

    st.divider()

    # -------- Apply filters to build the working slice --------

    def _apply_isin(_df, col, sel_vals):
        # Defensive: a missing column means "no filter on it"
        # rather than a KeyError, so the Priority Lister still runs
        # even when the source file lacks a dimension. Uses fuzzy
        # matching for casing / whitespace / punctuation tolerance,
        # consistent with the sidebar filters.
        if not sel_vals:
            return _df
        if col not in _df.columns:
            return _df
        return _df[fuzzy_isin(_df[col], sel_vals)]

    # Start from the sidebar-filtered slice so the tab-level
    # filters stack on top of the universal sidebar filters
    # (Exclude SKU + Smoothing trim are already applied in
    # filtered_df). Local widget options are now derived from
    # `filtered_df` too (see _opts() above), so the user can
    # only pick values that still exist after sidebar filtering;
    # the warning below fires only if the combined sidebar + local
    # picks return no rows.
    work = filtered_df.copy()
    work = _apply_isin(work, COL_VC_CATEGORY, vc_cat_sel)
    work = _apply_isin(work, COL_VC,          vc_model_sel)
    work = _apply_isin(work, COL_PCTYPE,      pc_type_sel)
    work = _apply_isin(work, COL_RE,          re_sel)
    work = _apply_isin(work, COL_CHANNEL,     channel_sel)
    work = _apply_isin(work, COL_REGION,      region_sel)
    work = _apply_isin(work, COL_ASM,         asm_sel)
    work = _apply_isin(work, COL_SETTY,       setty_sel)

    if work.empty:
        st.warning(
            "No rows match the combined sidebar + tab filters. "
            "Loosen the sidebar filters or one of the filters "
            "above."
        )
        return

    # Apply SKU-tier and SKU-type restrictions to the ranking
    # universe. The user spec: "selection of skus will run
    # analysis on the selected group of skus only for their
    # ranking purpose." → these subsets reshape the ranking
    # universe (and therefore the Top/Mid/Bottom percent bands
    # of cumulative contribution, which must sum to 100% within
    # the subset).
    if sku_tier_sel and COL_SKU_TIER in work.columns:
        work = work[
            work[COL_SKU_TIER].astype(str).isin(sku_tier_sel)
        ]

    if sku_type_local != "All" and COL_BRAND in work.columns:
        is_cooler = (
            work[COL_BRAND].astype(str).str.strip().str.lower()
            .isin(COOLER_BRANDS)
        )
        work = work[is_cooler] if sku_type_local == "Cooler" else work[~is_cooler]

    if work.empty:
        st.warning(
            "No SKUs left after applying SKU Tier / SKU Type "
            "scope. Loosen the SKU scope above."
        )
        return

    # =========================================================
    # NEW LAYOUT (v17.10) — replaces the old ranking-basis chooser,
    # quadrant matrix, and three-basis comparison tables with:
    #   (i)   summary header (transactions, SKUs sold, total value)
    #   (ii)  single ranked SKU list by TP × MRP × GM
    #   (iii) seasonality chart with selectable SKUs
    #   (iv)  five side-by-side Q1/Q2/Q3/Q4/Q5 lists with month
    #         selection + Gainer / Loser / Stable / Massive Gainer /
    #         Massive Loser tags computed quarter-over-quarter.
    # =========================================================

    months_in_data = [m for m in MONTH_COLS if m in work.columns]
    if not months_in_data:
        st.error(
            "Source file has no monthly pieces columns "
            "(Jan25_V … Mar26_V). Cannot build the priority "
            "list."
        )
        return

    n_months = len(months_in_data)

    # ---- Build a per-SKU monthly value table (vol × MRP) ----
    # Uses the per-month MRP (not the latest MRP) when the MRP
    # file has that month's column; otherwise falls back to the
    # latest available MRP for that SKU. Months for which a SKU
    # has neither a monthly MRP nor any latest MRP contribute 0.
    #
    # Returns:
    #   sku_monthly_vol     – DataFrame indexed by SKU, cols=months
    #   sku_monthly_value   – DataFrame indexed by SKU, cols=months
    #                         (value in ₹ = vol × MRP for that month)
    #   sku_latest_mrp      – Series indexed by SKU (₹)
    sku_monthly_vol = (
        work.groupby(COL_SKU)[months_in_data].sum()
    )

    # Map MONTH_COLS → MRP_MONTH_COLS for monthly MRP join.
    _vol_to_mrp = dict(zip(MONTH_COLS, MRP_MONTH_COLS))

    sku_monthly_value = pd.DataFrame(
        0.0,
        index=sku_monthly_vol.index,
        columns=months_in_data,
    )
    sku_latest_mrp = pd.Series(
        np.nan, index=sku_monthly_vol.index, name="latest_mrp"
    )

    if _mrp_available and _mrp_lookup is not None:
        # Key lookups by lower-cased SKU.
        _sku_keys = (
            sku_monthly_vol.index.astype(str).str.strip().str.lower()
        )

        mrp_cols_present = [
            c for c in MRP_MONTH_COLS if c in _mrp_lookup.columns
        ]
        mrp_aligned = (
            _mrp_lookup[mrp_cols_present]
            .reindex(_sku_keys.values)
        )
        mrp_aligned.index = sku_monthly_vol.index

        # Latest MRP per SKU = rightmost non-null MRP value.
        def _row_latest_mrp(row):
            non_null = row.dropna()
            if non_null.empty:
                return np.nan
            return float(non_null.iloc[-1])

        sku_latest_mrp = mrp_aligned.apply(
            _row_latest_mrp, axis=1
        )
        sku_latest_mrp.name = "latest_mrp"

        # Per-month value = vol × (monthly MRP, fallback latest).
        for vcol in months_in_data:
            mcol = _vol_to_mrp.get(vcol)
            if mcol and mcol in mrp_aligned.columns:
                monthly_mrp = mrp_aligned[mcol].astype(float)
                # Fallback to latest_mrp where monthly MRP is NaN.
                monthly_mrp = monthly_mrp.fillna(
                    sku_latest_mrp.astype(float)
                )
            else:
                monthly_mrp = sku_latest_mrp.astype(float)
            sku_monthly_value[vcol] = (
                sku_monthly_vol[vcol].astype(float)
                * monthly_mrp.fillna(0.0).values
            )

    # ---- Per-SKU GM index (constant across months) ----
    sku_gm_index = pd.Series(
        np.nan, index=sku_monthly_vol.index, name="gm_index"
    )
    if _gm_available and _gm_lookup is not None:
        _gm_keys = (
            sku_monthly_vol.index.astype(str).str.strip().str.lower()
        )
        sku_gm_index = pd.Series(
            _gm_lookup.reindex(_gm_keys.values).values,
            index=sku_monthly_vol.index,
            name="gm_index",
        )

    # =====================================================
    # (i) SUMMARY HEADER
    # =====================================================
    # • Number of outlets    = unique outlets in the filtered
    #                          slice (the outlet universe the
    #                          report is computed over).
    # • Total SKUs sold      = unique SKUs with any volume > 0
    # • Total TP×MRP value   = sum of vol × MRP across all SKUs
    #                          and months in the slice
    # • Avg TP×MRP value/mo  = total value / n_months
    num_outlets = (
        int(work[COL_OUTLET].nunique())
        if COL_OUTLET in work.columns
        else 0
    )

    skus_sold_mask = sku_monthly_vol.sum(axis=1) > 0
    total_skus_sold = int(skus_sold_mask.sum())

    total_value = float(sku_monthly_value.values.sum())

    if n_months > 0:
        avg_value_per_month = total_value / n_months
    else:
        avg_value_per_month = 0.0

    def _fmt_inr_compact(v):
        """Render large rupee values compactly: ₹1.2 Cr / ₹4.5 L / ₹12,345."""
        v = float(v)
        sign = "-" if v < 0 else ""
        v = abs(v)
        if v >= 1e7:
            return f"{sign}₹{v / 1e7:,.2f} Cr"
        if v >= 1e5:
            return f"{sign}₹{v / 1e5:,.2f} L"
        return f"{sign}₹{v:,.0f}"

    st.subheader("📊 Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Number of outlets",
        f"{num_outlets:,}",
        help=(
            "Unique outlets in the current filtered slice "
            "(sidebar + tab-level filters applied)."
        ),
    )
    m2.metric(
        "Total SKUs sold",
        f"{total_skus_sold:,}",
        help="Unique SKUs with any sales in the slice.",
    )
    m3.metric(
        "Total value (TP × MRP)",
        _fmt_inr_compact(total_value),
        help=(
            "Sum of pieces × MRP across all SKUs and months "
            "in the slice. Uses per-month MRP where available, "
            "falling back to the latest MRP for the SKU."
        ),
    )
    m4, _m5, _m6 = st.columns(3)
    m4.metric(
        "Avg value / month",
        _fmt_inr_compact(avg_value_per_month),
        help=f"Total value ÷ {n_months} months in data.",
    )

    st.divider()

    # =====================================================
    # (ii) RANKED SKU LIST  —  by TP × MRP × GM
    # =====================================================
    # Throughput per SKU is the period-average:
    #   tp = total_vol / active_outlets / n_months
    # Score = tp × latest_mrp × gm_index
    # SKUs without a GM entry get gm_index = 0 (effectively
    # excluded — matches the spec that the chart-of-record is
    # TP × MRP × GM).
    st.subheader("🏆 Ranked SKU list — TP × MRP × Gross Margin")
    st.caption(
        "Ranked by Throughput × Latest MRP × Gross Margin. "
        "Cumulative-share buckets: 🟢 Top (30%) · "
        "🟡 Mid (30%) · 🔴 Bottom (40%)."
    )

    period_vol = work[months_in_data].astype(float).sum(axis=1)
    active_mask_outlet = period_vol > 0
    sku_active_outlets = (
        work[active_mask_outlet]
        .groupby(COL_SKU)[COL_OUTLET]
        .nunique()
        .rename("active_outlets")
    )
    sku_total_vol_series = (
        sku_monthly_vol.sum(axis=1).rename("total_vol")
    )

    rank_df = pd.concat(
        [sku_total_vol_series, sku_active_outlets], axis=1
    )
    rank_df["active_outlets"] = (
        rank_df["active_outlets"].fillna(0).astype(int)
    )
    rank_df = rank_df[rank_df["active_outlets"] > 0]

    if rank_df.empty:
        st.warning(
            "No SKUs with active outlets in the working slice."
        )
        return

    rank_df["throughput"] = (
        rank_df["total_vol"]
        / rank_df["active_outlets"]
        / n_months
    )
    rank_df["latest_mrp"] = sku_latest_mrp.reindex(rank_df.index)
    rank_df["gm_index"] = sku_gm_index.reindex(rank_df.index)
    # Per-SKU total value = sum of monthly (vol × MRP).
    rank_df["total_value"] = (
        sku_monthly_value.sum(axis=1).reindex(rank_df.index)
    )

    # Penetration = active outlets / total outlets in slice.
    total_outlets_in_slice = int(work[COL_OUTLET].nunique())
    if total_outlets_in_slice > 0:
        rank_df["penetration_pct"] = (
            rank_df["active_outlets"]
            / total_outlets_in_slice
            * 100.0
        )
    else:
        rank_df["penetration_pct"] = 0.0

    # Composite ranking score: TP × MRP × GM. NaN GMs contribute 0
    # (those SKUs sink to the bottom). NaN MRPs likewise → 0.
    rank_df["score"] = (
        rank_df["throughput"].fillna(0.0)
        * rank_df["latest_mrp"].fillna(0.0)
        * rank_df["gm_index"].fillna(0.0)
    )
    rank_df = rank_df[rank_df["score"] > 0]
    if rank_df.empty:
        # Diagnose WHY the score is all-zero rather than emitting
        # the generic "check the files" hint. The three causes
        # are: (a) no MRP file uploaded, (b) MRP loaded but joins
        # zero source SKUs (name mismatch), (c) likewise for GM.
        # We surface counts so the user can see which side failed.
        n_skus_in_slice = len(sku_monthly_vol)
        n_mrp_hits = int(sku_latest_mrp.notna().sum())
        n_gm_hits  = int(sku_gm_index.notna().sum())

        reasons = []
        if not _mrp_available:
            reasons.append(
                "**MRP file not uploaded** — upload it via the "
                "'Upload MRP file (optional)' slot in the sidebar."
            )
        elif n_mrp_hits == 0:
            reasons.append(
                f"**MRP file loaded but matches 0 / {n_skus_in_slice} "
                "SKUs in this slice** — the SKU names in the source "
                "data don't line up with the Line Names in the MRP "
                "file (after case / whitespace normalisation)."
            )

        if not _gm_available:
            reasons.append(
                "**GM file not uploaded** — upload it via the "
                "'Upload Gross Margin file (optional)' slot."
            )
        elif n_gm_hits == 0:
            reasons.append(
                f"**GM file loaded but matches 0 / {n_skus_in_slice} "
                "SKUs in this slice** — Line Names don't align."
            )

        if not reasons:
            # Both files loaded AND both hit some SKUs, but their
            # intersection with positive throughput is empty.
            reasons.append(
                f"MRP matches {n_mrp_hits} / {n_skus_in_slice} SKUs, "
                f"GM matches {n_gm_hits} / {n_skus_in_slice} SKUs, "
                "but no single SKU has all three (throughput > 0, "
                "MRP, GM) at the same time. Loosen the filters or "
                "check the SKU-name overlap between the two files."
            )

        st.warning(
            "No SKUs have a positive TP × MRP × GM score.\n\n"
            + "\n\n".join(f"- {r}" for r in reasons)
        )
        return

    rank_df = rank_df.sort_values("score", ascending=False)
    _total_score = rank_df["score"].sum()
    rank_df["share_pct"] = rank_df["score"] / _total_score * 100.0
    rank_df["cum_share_pct"] = rank_df["share_pct"].cumsum()

    def _bucket_by_cum(cum_series):
        """30/30/40 split by cumulative-share boundary."""
        buckets = []
        prev = 0.0
        for c in cum_series:
            if prev < 30.0:
                buckets.append("Top (30%)")
            elif prev < 60.0:
                buckets.append("Mid (30%)")
            else:
                buckets.append("Bottom (40%)")
            prev = c
        return buckets

    rank_df["bucket"] = _bucket_by_cum(rank_df["cum_share_pct"])

    _bucket_bullet = {
        "Top (30%)":    "🟢",
        "Mid (30%)":    "🟡",
        "Bottom (40%)": "🔴",
    }
    _bucket_order = ["Top (30%)", "Mid (30%)", "Bottom (40%)"]

    # Build the display table with subtotal rows interleaved
    # between buckets.
    main_rows = []
    rank_pos = 0
    for bname in _bucket_order:
        bdf = rank_df[rank_df["bucket"] == bname]
        if bdf.empty:
            continue
        for sku, r in bdf.iterrows():
            rank_pos += 1
            main_rows.append({
                "#": rank_pos,
                " ": _bucket_bullet[bname],
                "Bucket": bname,
                "SKU": sku,
                "Penetration %": round(
                    float(r["penetration_pct"]), 2
                ),
                "Throughput": round(float(r["throughput"]), 2),
                "Total Pieces": int(round(float(r["total_vol"]))),
                "Total Value (₹)": float(r["total_value"]),
                "Share %": round(float(r["share_pct"]), 2),
            })
        # Subtotal row for this bucket.
        main_rows.append({
            "#": "—",
            " ": "Σ",
            "Bucket": f"Subtotal {bname}",
            "SKU": f"({len(bdf)} SKUs)",
            "Penetration %": np.nan,
            "Throughput": round(float(bdf["throughput"].sum()), 2),
            "Total Pieces": int(round(float(bdf["total_vol"].sum()))),
            "Total Value (₹)": float(bdf["total_value"].sum()),
            "Share %": round(float(bdf["share_pct"].sum()), 2),
        })

    main_table = pd.DataFrame(main_rows)
    # Pretty-format the rupee column as a string so subtotal rows
    # render cleanly (and we get compact ₹X.XX Cr/L labels).
    main_table["Total Value (₹)"] = main_table["Total Value (₹)"].apply(
        lambda v: _fmt_inr_compact(v) if pd.notna(v) else ""
    )
    main_table["Penetration %"] = main_table["Penetration %"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else ""
    )

    st.dataframe(
        main_table,
        use_container_width=True,
        hide_index=True,
        height=min(640, 40 + 35 * len(main_table)),
    )

    # Quick bucket-share summary cards.
    bs_c1, bs_c2, bs_c3 = st.columns(3)
    _bucket_summary = (
        rank_df.groupby("bucket")
        .agg(
            n_skus=("score", "size"),
            share=("share_pct", "sum"),
        )
    )
    for bcol, bname in zip(
        [bs_c1, bs_c2, bs_c3],
        _bucket_order,
    ):
        if bname in _bucket_summary.index:
            r = _bucket_summary.loc[bname]
            bcol.metric(
                f"{_bucket_bullet[bname]} {bname}",
                f"{int(r['n_skus'])} SKUs",
                f"{r['share']:.1f}% of total",
            )
        else:
            bcol.metric(
                f"{_bucket_bullet[bname]} {bname}",
                "0 SKUs",
                "0% of total",
            )

    # CSV download of the main ranked list.
    _csv = main_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download ranked list (CSV)",
        data=_csv,
        file_name="sku_priority_list_ranked.csv",
        mime="text/csv",
        key="spl_v1710_csv_main",
    )

    st.divider()

    # =====================================================
    # (iii) SEASONALITY CHART — selectable SKUs
    # =====================================================
    st.subheader("📈 Seasonality chart — selectable SKUs")
    st.caption(
        "Plot any subset of the ranked SKUs across the "
        f"{n_months} months. Y-axis is Throughput × MRP × GM "
        "by default (toggle below)."
    )

    # Build monthly throughput table (vol_month / active_outlets).
    monthly_throughput = sku_monthly_vol.div(
        rank_df["active_outlets"], axis=0
    ).reindex(rank_df.index)
    monthly_throughput = monthly_throughput.fillna(0.0)

    # SKU picker — buckets-first multi-select, then SKU multi-select.
    _sel_c1, _sel_c2 = st.columns([1, 3])
    with _sel_c1:
        _season_y_metric = st.radio(
            "Y-axis",
            options=[
                "TP × MRP × GM",
                "Throughput",
                "Total Pieces",
                "TP × MRP",
            ],
            index=0,
            key="spl_v1710_season_y",
        )
    with _sel_c2:
        _season_bucket_pick = st.multiselect(
            "Pre-filter SKUs by bucket",
            options=_bucket_order,
            default=list(_bucket_order),
            key="spl_v1710_season_bkt",
            help=(
                "Restricts the SKU picker below to SKUs in "
                "these buckets. Defaults to all three "
                "(Top + Mid + Bottom)."
            ),
        )

    _avail_skus_for_pick = (
        rank_df[rank_df["bucket"].isin(_season_bucket_pick)].index.tolist()
        if _season_bucket_pick
        else rank_df.index.tolist()
    )
    # Default selection = all available SKUs (after the bucket
    # pre-filter). The user can trim down from there.
    _default_skus = list(_avail_skus_for_pick)
    _season_sku_pick = st.multiselect(
        "Pick SKUs to plot",
        options=_avail_skus_for_pick,
        default=_default_skus,
        key="spl_v1710_season_skus",
    )

    # Safety: keep only SKUs that actually exist in the
    # throughput / rank index (defensive — Streamlit can hold
    # stale multiselect state across reruns when filters change).
    _valid_pool = set(monthly_throughput.index)
    _season_sku_pick = [
        s for s in _season_sku_pick if s in _valid_pool
    ]

    if not _season_sku_pick:
        st.info(
            "Pick at least one SKU above to draw the chart."
        )
    else:
        # Build the per-month series for the chosen Y-axis.
        if _season_y_metric == "Throughput":
            _plot_df = monthly_throughput.loc[_season_sku_pick]
            _y_label = "Monthly throughput (units/outlet)"
        elif _season_y_metric == "Total Pieces":
            _plot_df = sku_monthly_vol.loc[_season_sku_pick]
            _y_label = "Monthly total pieces (units)"
        elif _season_y_metric == "TP × MRP":
            _mrp_vec = rank_df["latest_mrp"].fillna(0.0)
            _plot_df = monthly_throughput.loc[_season_sku_pick].mul(
                _mrp_vec.loc[_season_sku_pick], axis=0
            )
            _y_label = "Monthly TP × Latest MRP (₹)"
        else:  # TP × MRP × GM
            _mrp_vec = rank_df["latest_mrp"].fillna(0.0)
            _gm_vec = rank_df["gm_index"].fillna(0.0)
            _plot_df = (
                monthly_throughput.loc[_season_sku_pick]
                .mul(_mrp_vec.loc[_season_sku_pick], axis=0)
                .mul(_gm_vec.loc[_season_sku_pick], axis=0)
            )
            _y_label = "Monthly TP × MRP × GM"

        # Build a per-SKU long frame for the picked Y-axis metric.
        _long = (
            _plot_df.reset_index()
            .melt(
                id_vars=COL_SKU,
                var_name="Month",
                value_name=_y_label,
            )
        )
        _long["Month"] = _long["Month"].str.replace(
            "_V", "", regex=False
        )
        _month_pretty_order = [
            m.replace("_V", "") for m in months_in_data
        ]
        _long["Month"] = pd.Categorical(
            _long["Month"],
            categories=_month_pretty_order,
            ordered=True,
        )
        _long = _long.sort_values([COL_SKU, "Month"])

        # ----- Average line across the selected SKUs -----
        # Compute the simple mean across the chosen SKUs for the
        # currently-displayed Y-axis metric, month by month, and
        # append it as a synthetic series so it appears alongside
        # the individual SKU lines on the same chart.
        _avg_series = _plot_df.mean(axis=0)
        _avg_long = pd.DataFrame({
            COL_SKU: "Average (selected SKUs)",
            "Month": [
                str(m).replace("_V", "") for m in _avg_series.index
            ],
            _y_label: _avg_series.values,
        })
        _avg_long["Month"] = pd.Categorical(
            _avg_long["Month"],
            categories=_month_pretty_order,
            ordered=True,
        )
        _long_with_avg = pd.concat(
            [_long, _avg_long], ignore_index=True
        ).sort_values([COL_SKU, "Month"])

        _season_fig = px.line(
            _long_with_avg,
            x="Month",
            y=_y_label,
            color=COL_SKU,
            markers=True,
        )
        # Make the Average trace stand out: thicker line, dashed,
        # white colour so it reads as a summary overlay rather
        # than another SKU.
        for _tr in _season_fig.data:
            if _tr.name == "Average (selected SKUs)":
                _tr.line.width = 4
                _tr.line.dash = "dash"
                _tr.line.color = "#FFFFFF"
                _tr.marker.size = 9
                _tr.marker.symbol = "diamond"
        _season_fig.update_layout(
            height=500,
            plot_bgcolor="#050816",
            paper_bgcolor="#050816",
            font=dict(color="white"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
        )
        st.plotly_chart(
            _season_fig,
            use_container_width=True,
            key="spl_v1710_season_chart",
        )

        # =====================================================
        # Month-wise table (selected SKUs) — Throughput OR Penetration %
        # =====================================================
        # This sits directly under the seasonality chart. By
        # default it shows the simple average of monthly Throughput
        # (units / outlet) across the SKUs currently picked above.
        # The user can toggle to Penetration % (active outlets ÷
        # total outlets in slice, per SKU per month). Independent
        # of the Y-axis radio used for the chart.
        _thr_metric_choice = st.radio(
            "Show",
            options=["Throughput (units / outlet)", "Penetration %"],
            index=0,
            horizontal=True,
            key="spl_v1710_season_thr_metric",
            help=(
                "Throughput = pieces / active outlet / month. "
                "Penetration % = outlets stocking this SKU in the "
                "month ÷ total outlets in the slice."
            ),
        )
        _show_penetration = (_thr_metric_choice == "Penetration %")

        if _show_penetration:
            st.markdown(
                "**📊 Month-wise Penetration % — selected SKUs**"
            )
            st.caption(
                "First row is the simple average of per-SKU monthly "
                "Penetration % across the SKUs picked above. "
                "Subsequent rows show monthly Penetration % for each "
                "selected SKU individually. Penetration % = outlets "
                "stocking the SKU that month ÷ total outlets in the "
                "current slice."
            )

            # Build monthly penetration per SKU.
            # Total outlets per month in the slice (denominator).
            _total_outlets_in_slice = int(work[COL_OUTLET].nunique())
            if _total_outlets_in_slice <= 0:
                st.info("No outlets in the slice — penetration is undefined.")
            else:
                # Per (SKU, month): number of distinct outlets with
                # vol > 0. Done in one groupby for speed.
                _sel_skus_set = set(_season_sku_pick)
                _sub = work[work[COL_SKU].isin(_sel_skus_set)]
                # For each month, count distinct outlets per SKU
                # where vol > 0. We loop months (~13 max) which is
                # cheap and avoids a big melt.
                _pen_dict = {}
                for _m in months_in_data:
                    _mask = _sub[_m].astype(float) > 0
                    _counts = (
                        _sub.loc[_mask]
                        .groupby(COL_SKU)[COL_OUTLET]
                        .nunique()
                    )
                    _pen_dict[str(_m).replace("_V", "")] = _counts

                _pen_df = pd.DataFrame(_pen_dict).reindex(
                    _season_sku_pick
                ).fillna(0.0)
                # Convert to percentage of slice outlets.
                _pen_df = (_pen_df / _total_outlets_in_slice) * 100.0

                _avg_pen_row = _pen_df.mean(axis=0).to_frame().T
                _avg_pen_row.index = ["Avg Penetration %"]
                _pen_table = pd.concat(
                    [_avg_pen_row, _pen_df], axis=0
                ).round(2)
                _pen_table.index.name = "SKU"

                st.dataframe(
                    _pen_table,
                    use_container_width=True,
                )
                st.download_button(
                    "⬇ Download monthly penetration table (CSV)",
                    data=_pen_table.to_csv().encode("utf-8"),
                    file_name="sku_priority_penetration_monthly.csv",
                    mime="text/csv",
                    key="spl_v1710_season_avg_pen_dl",
                )
        else:
            st.markdown(
                "**📊 Month-wise Throughput "
                "(units / outlet) — selected SKUs**"
            )
            st.caption(
                "First row is the simple average of per-SKU monthly "
                "throughput across the SKUs picked above. Subsequent "
                "rows show monthly throughput for each selected SKU "
                "individually. Independent of the Y-axis toggle — "
                "always Throughput."
            )
            # Per-SKU monthly throughput for the selected SKUs.
            _per_sku_thr = (
                monthly_throughput.loc[_season_sku_pick].copy()
            )
            # Pretty month column labels (strip the "_V" suffix used
            # internally on the volume columns).
            _per_sku_thr.columns = [
                str(c).replace("_V", "") for c in _per_sku_thr.columns
            ]
            # Average row across selected SKUs (one value per month).
            _avg_thr_row = _per_sku_thr.mean(axis=0).to_frame().T
            _avg_thr_row.index = ["Avg Throughput"]
            # Stack: Avg on top, then each SKU as its own row.
            _thr_table = pd.concat(
                [_avg_thr_row, _per_sku_thr], axis=0
            ).round(2)
            _thr_table.index.name = "SKU"
            st.dataframe(
                _thr_table,
                use_container_width=True,
            )
            st.download_button(
                "⬇ Download monthly throughput table (CSV)",
                data=_thr_table.to_csv().encode("utf-8"),
                file_name="sku_priority_throughput_monthly.csv",
                mime="text/csv",
                key="spl_v1710_season_avg_thr_dl",
            )

    st.divider()

    # =====================================================
    # (iv) FIVE QUARTERLY LISTS — Q1 / Q2 / Q3 / Q4 / Q5
    # =====================================================
    # Each quarter block carries:
    #   • a month picker (any month or group of months from the
    #     available period)
    #   • a header showing total TP×MRP, total volume, and total
    #     outlets selling for the SKUs in that selection
    #   • a ranked list (TP × MRP × GM) with 30/30/40 Top/Mid/
    #     Bottom buckets and same colour coding
    #   • columns: Penetration % (in selected months), avg
    #     monthly TP, total units sold, total value (TP × MRP)
    #   • a Gainer / Loser / Stable / Massive Gainer /
    #     Massive Loser tag based on per-SKU throughput change
    #     from the previous Q.
    st.subheader("🗓️ Quarterly comparison (Q1 / Q2 / Q3 / Q4 / Q5)")
    st.caption(
        "Pick a month or set of months for each quarter. SKUs are "
        "ranked by Throughput × Latest MRP × Gross Margin. Tags "
        "compare each SKU's avg monthly throughput to the previous "
        "quarter's selection (Q1 is the baseline)."
    )

    # ---- Quarterly-section-local extra filters ----
    # These narrow the SKU universe for ALL four quarter blocks
    # below. They sit ON TOP of the page-level filters above
    # (VC_CAT, Region, …, SKU Tiers, SKU Type) — they don't
    # replace them. We re-expose SKU Tiers, Cooler/Ambient, and
    # Brand here so the user can fine-tune the quarterly view
    # without losing the page-wide context.
    _q_brand_opts = (
        sorted(work[COL_BRAND].dropna().astype(str).unique())
        if COL_BRAND in work.columns else []
    )
    _q_sku_tier_opts = (
        sorted(work[COL_SKU_TIER].dropna().astype(str).unique())
        if COL_SKU_TIER in work.columns else []
    )

    _qf1, _qf2, _qf3 = st.columns([2, 2, 1])
    with _qf1:
        q_brand_sel = st.multiselect(
            "Brand",
            options=_q_brand_opts,
            default=[],
            key="spl_v1710_q_brand",
            help=(
                "Restrict the quarterly lists to these brands. "
                "Empty = all brands in the page-wide slice."
            ),
        )
    with _qf2:
        q_sku_tier_sel = st.multiselect(
            "SKU Tier (quarterly)",
            options=_q_sku_tier_opts,
            default=[],
            key="spl_v1710_q_sku_tier",
            help=(
                "Restrict the quarterly lists to these SKU tiers. "
                "Empty = all tiers."
            ),
        )
    with _qf3:
        q_sku_type_sel = st.radio(
            "SKU Type",
            options=["All", "Cooler", "Ambient"],
            index=0,
            horizontal=False,
            key="spl_v1710_q_sku_type",
            help=(
                "Cooler = Silk + Bournville + Temptations. "
                "Ambient = everything else."
            ),
        )

    _qf4, _qf5 = st.columns([2, 2])
    with _qf4:
        q_min_vol = st.slider(
            "Minimum total pieces per quarter",
            min_value=100,
            max_value=1000,
            value=100,
            step=50,
            key="spl_v1710_q_min_vol",
            help=(
                "SKUs whose total pieces in the picked months "
                "falls below this threshold are dropped from "
                "that quarter's list."
            ),
        )
    with _qf5:
        q_min_pen = st.slider(
            "Minimum penetration % per quarter",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            key="spl_v1710_q_min_pen",
            help=(
                "SKUs whose penetration % in the picked months "
                "(active outlets / total outlets in the quarterly "
                "slice) falls below this threshold are dropped "
                "from that quarter's list."
            ),
        )

    # Build a quarterly-local working slice from `work` (the
    # page-level filtered slice).
    q_work = work.copy()
    if q_brand_sel and COL_BRAND in q_work.columns:
        q_work = q_work[
            q_work[COL_BRAND].astype(str).isin(q_brand_sel)
        ]
    if q_sku_tier_sel and COL_SKU_TIER in q_work.columns:
        q_work = q_work[
            q_work[COL_SKU_TIER].astype(str).isin(q_sku_tier_sel)
        ]
    if q_sku_type_sel != "All" and COL_BRAND in q_work.columns:
        _q_is_cooler = (
            q_work[COL_BRAND].astype(str).str.strip().str.lower()
            .isin(COOLER_BRANDS)
        )
        q_work = (
            q_work[_q_is_cooler]
            if q_sku_type_sel == "Cooler"
            else q_work[~_q_is_cooler]
        )

    if q_work.empty:
        st.warning(
            "No SKUs in the quarterly slice after applying the "
            "Brand / Tier / Type filters above. Loosen them to "
            "see the four quarterly lists."
        )
        return

    # SKU-level brand lookup (one brand per SKU) for the
    # quarterly-local slice — used in the Excel export below.
    _q_brand_per_sku = (
        q_work.groupby(COL_SKU)[COL_BRAND]
        .agg(
            lambda s: (
                s.dropna().astype(str).iloc[0]
                if not s.dropna().empty else ""
            )
        )
    )

    _pretty_months = [m.replace("_V", "") for m in months_in_data]
    _pretty_to_raw = dict(zip(_pretty_months, months_in_data))

    # Sensible default month-splits for the 5 quarters: divide
    # the available months into 5 contiguous chunks as evenly as
    # possible. If there are fewer than 5 months, later quarters
    # default to empty (user can still pick manually).
    def _default_quarter_splits(months):
        n = len(months)
        if n == 0:
            return [[], [], [], [], []]
        # ceil-divide for the first chunks, floor for the rest.
        sizes = [n // 5] * 5
        for i in range(n % 5):
            sizes[i] += 1
        out = []
        i = 0
        for s in sizes:
            out.append(months[i:i + s])
            i += s
        return out

    _q_defaults = _default_quarter_splits(_pretty_months)

    def _classify_tag(curr_tp, prev_tp):
        """
        Compare avg-monthly throughput for a SKU between two
        quarters. Returns one of:
          New, Lost, Big Gain, Gainer, Stable, Loser, Big Drop.

        Thresholds (relative change in avg throughput vs. prev Q):
          ≥ +50%   → Big Gain
          +15..+50 → Gainer
          ±15      → Stable
          -50..-15 → Loser
          ≤ -50    → Big Drop
        Edge cases: prev = 0 & curr > 0 → New;
                    curr = 0 & prev > 0 → Lost;
                    both 0 / NaN          → "—".
        """
        if (pd.isna(prev_tp) and pd.isna(curr_tp)):
            return "—"
        prev = 0.0 if pd.isna(prev_tp) else float(prev_tp)
        curr = 0.0 if pd.isna(curr_tp) else float(curr_tp)
        if prev <= 1e-9 and curr <= 1e-9:
            return "—"
        if prev <= 1e-9 and curr > 0:
            return "New"
        if curr <= 1e-9 and prev > 0:
            return "Lost"
        change = (curr - prev) / prev
        if change >= 0.50:
            return "Big Gain"
        if change >= 0.15:
            return "Gainer"
        if change <= -0.50:
            return "Big Drop"
        if change <= -0.15:
            return "Loser"
        return "Stable"

    _tag_emoji = {
        "Big Gain": "🚀",
        "Gainer":   "⬆️",
        "Stable":   "➡️",
        "Loser":    "⬇️",
        "Big Drop": "💥",
        "New":      "✨",
        "Lost":     "❌",
        "—":        "·",
    }

    # Colour hex (used for Excel export and any inline styling).
    _tag_color = {
        "Big Gain": "1B5E20",  # deep green
        "Gainer":   "66BB6A",  # green
        "Stable":   "9E9E9E",  # grey
        "Loser":    "EF5350",  # red-coral
        "Big Drop": "B71C1C",  # deep red
        "New":      "1976D2",  # blue
        "Lost":     "424242",  # dark grey
        "—":        "BDBDBD",  # light grey
    }

    def _quarter_table(qname, picked_pretty_months, prev_avg_tp):
        """
        Returns (df_summary_metrics, dataframe_for_display,
                 per_sku_avg_tp_for_passing_to_next_quarter,
                 raw_qdf).

        df_summary_metrics: dict with total_value, total_vol,
                            total_outlets, n_skus.
        raw_qdf: the per-SKU dataframe (indexed by SKU) carrying
                 every column needed for the Excel export.
        """
        picked_raw = [
            _pretty_to_raw[p]
            for p in picked_pretty_months
            if p in _pretty_to_raw
        ]
        if not picked_raw:
            return (
                {
                    "total_value": 0.0,
                    "total_vol":   0,
                    "total_outlets": 0,
                    "n_skus": 0,
                },
                None,
                pd.Series(dtype=float),
                pd.DataFrame(),
            )

        n_q_months = len(picked_raw)

        # Per-SKU sums over the picked months — operates on the
        # quarterly-local slice (q_work), which already has the
        # Brand / SKU-Tier / Cooler-Ambient filters applied.
        q_vol = (
            q_work.groupby(COL_SKU)[picked_raw].sum().sum(axis=1)
        )
        # Outlets active per SKU in this quarter:
        q_active_mask = q_work[picked_raw].astype(float).sum(axis=1) > 0
        q_active_outlets = (
            q_work[q_active_mask]
            .groupby(COL_SKU)[COL_OUTLET]
            .nunique()
        )
        # Outlets in the slice that sold ANYTHING in this quarter:
        q_total_outlets = int(
            q_work.loc[q_active_mask, COL_OUTLET].nunique()
        )

        # Outlets in the quarterly-local slice — denominator for
        # penetration. Falls back to page-level if zero.
        _q_total_outlets_in_slice = int(q_work[COL_OUTLET].nunique())
        if _q_total_outlets_in_slice <= 0:
            _q_total_outlets_in_slice = total_outlets_in_slice

        # Per-SKU total value (vol × MRP) over the picked months.
        # sku_monthly_value is indexed on the page-level work;
        # reindex to the SKUs present in q_work.
        _q_skus = q_work[COL_SKU].dropna().astype(str).unique()
        if picked_raw:
            q_value = (
                sku_monthly_value
                .reindex(_q_skus)[picked_raw]
                .sum(axis=1)
            )
        else:
            q_value = pd.Series(0.0, index=_q_skus)

        # Build the per-SKU dataframe.
        qdf = pd.DataFrame({
            "total_vol":      q_vol,
            "active_outlets": q_active_outlets,
            "total_value":    q_value,
        })
        qdf["active_outlets"] = (
            qdf["active_outlets"].fillna(0).astype(int)
        )
        qdf["total_vol"] = qdf["total_vol"].fillna(0.0)
        qdf["total_value"] = qdf["total_value"].fillna(0.0)
        qdf = qdf[
            (qdf["active_outlets"] > 0)
            | (qdf["total_vol"] > 0)
        ]

        if qdf.empty:
            return (
                {
                    "total_value":    float(q_value.sum()),
                    "total_vol":      int(round(float(q_vol.sum()))),
                    "total_outlets":  q_total_outlets,
                    "n_skus":         0,
                },
                None,
                pd.Series(dtype=float),
                pd.DataFrame(),
            )

        # Avg monthly TP = total_vol / active_outlets / n_q_months
        # (mirrors the main-list throughput formula).
        qdf["avg_monthly_tp"] = (
            qdf["total_vol"]
            / qdf["active_outlets"].replace(0, np.nan)
            / max(n_q_months, 1)
        )

        # Penetration % = active outlets for this SKU in the
        # picked months / total outlets in the quarterly slice.
        if _q_total_outlets_in_slice > 0:
            qdf["pen_pct"] = (
                qdf["active_outlets"]
                / _q_total_outlets_in_slice
                * 100.0
            )
        else:
            qdf["pen_pct"] = 0.0

        # Join MRP & GM for scoring.
        qdf["latest_mrp"] = sku_latest_mrp.reindex(qdf.index)
        qdf["gm_index"] = sku_gm_index.reindex(qdf.index)
        # GM Value (₹) — period-total rupee gross-margin for the
        # SKU: Total Value (vol × monthly MRP) × GM index. This
        # uses the same monthly-MRP basis as `total_value`, so
        # `gm_value` is exactly `total_value × gm_index` (the GM
        # index is constant across months). When GM is missing
        # the column comes out as 0 (because gm_index is NaN →
        # filled to 0), which is the same convention used by
        # `tp_mrp_gm` below.
        qdf["gm_value"] = (
            qdf["total_value"].fillna(0.0)
            * qdf["gm_index"].fillna(0.0)
        )
        # Composite TP × MRP × GM score — the basis the user
        # asked for ("rank the skus in terms of tp*mrp*margin").
        qdf["tp_mrp_gm"] = (
            qdf["avg_monthly_tp"].fillna(0.0)
            * qdf["latest_mrp"].fillna(0.0)
            * qdf["gm_index"].fillna(0.0)
        )
        qdf["score"] = qdf["tp_mrp_gm"]

        # Apply the minimum-volume filter from the slider above.
        qdf = qdf[qdf["total_vol"] >= q_min_vol]

        # Apply the minimum-penetration filter from the slider above.
        qdf = qdf[qdf["pen_pct"] >= q_min_pen]

        qdf = qdf[qdf["score"] > 0]
        if qdf.empty:
            # All SKUs filtered out — headline must read zero so
            # it stays consistent with the (empty) bucket subtotals.
            return (
                {
                    "total_value":    0.0,
                    "total_vol":      0,
                    "total_outlets":  0,
                    "n_skus":         0,
                },
                None,
                pd.Series(dtype=float),
                pd.DataFrame(),
            )

        # Headline totals computed AFTER the min-volume,
        # min-penetration, and score>0 filters above, so the
        # headline "total value / volume / outlets" equals the sum
        # of the Top / Mid / Bottom bucket subtotals shown below.
        _q_headline_value = float(qdf["total_value"].sum())
        _q_headline_vol = int(round(float(qdf["total_vol"].sum())))
        # Outlets selling = union of outlets that sold any of the
        # SURVIVING SKUs in the picked months. Recomputed here so
        # it reconciles with the filtered SKU set.
        _q_surviving_skus = qdf.index.tolist()
        _q_headline_outlets = int(
            q_work[
                q_active_mask
                & q_work[COL_SKU].astype(str).isin(_q_surviving_skus)
            ][COL_OUTLET].nunique()
        )

        qdf = qdf.sort_values("score", ascending=False)
        _qs_total = qdf["score"].sum()
        qdf["share_pct"] = qdf["score"] / _qs_total * 100.0
        qdf["cum_share_pct"] = qdf["share_pct"].cumsum()
        qdf["bucket"] = _bucket_by_cum(qdf["cum_share_pct"])

        # Brand per SKU for the export.
        qdf["brand"] = _q_brand_per_sku.reindex(qdf.index).fillna("")

        # Per-SKU tag against previous quarter — stored on qdf so
        # the Excel export sees it too.
        if prev_avg_tp is None:
            qdf["tag"] = ""
        else:
            qdf["tag"] = [
                _classify_tag(
                    qdf.loc[s, "avg_monthly_tp"],
                    (
                        prev_avg_tp.get(s, np.nan)
                        if isinstance(prev_avg_tp, pd.Series)
                        else np.nan
                    ),
                )
                for s in qdf.index
            ]

        # Build display rows with subtotals.
        rows = []
        rank_pos = 0
        for bname in _bucket_order:
            bdf = qdf[qdf["bucket"] == bname]
            if bdf.empty:
                continue
            for sku, r in bdf.iterrows():
                rank_pos += 1
                tag = r["tag"] if r["tag"] else "—"
                rows.append({
                    "#": rank_pos,
                    " ": _bucket_bullet[bname],
                    "Bucket": bname,
                    "SKU": sku,
                    "Pen %": round(float(r["pen_pct"]), 2),
                    "Avg TP/mo": round(
                        float(r["avg_monthly_tp"]), 2
                    ),
                    "Total Vol": int(round(float(r["total_vol"]))),
                    "Total Value": float(r["total_value"]),
                    "GM Value": float(r["gm_value"]),
                    "TP×MRP×GM": round(float(r["tp_mrp_gm"]), 2),
                    "Tag": (
                        f"{_tag_emoji.get(tag, '·')} {tag}"
                        if tag and tag != "—" else ""
                    ),
                })
            rows.append({
                "#": "—",
                " ": "Σ",
                "Bucket": f"Subtotal {bname}",
                "SKU": f"({len(bdf)} SKUs)",
                "Pen %": "",
                "Avg TP/mo": round(
                    float(bdf["avg_monthly_tp"].sum()), 2
                ),
                "Total Vol": int(round(
                    float(bdf["total_vol"].sum())
                )),
                "Total Value": float(bdf["total_value"].sum()),
                "GM Value": float(bdf["gm_value"].sum()),
                "TP×MRP×GM": round(
                    float(bdf["tp_mrp_gm"].sum()), 2
                ),
                "Tag": "",
            })

        return (
            {
                "total_value":    _q_headline_value,
                "total_vol":      _q_headline_vol,
                "total_outlets":  _q_headline_outlets,
                "n_skus":         len(qdf),
            },
            pd.DataFrame(rows),
            qdf["avg_monthly_tp"],
            qdf,
        )

    def _render_quarter_block(qname, qkey, default_months, prev_avg_tp):
        st.markdown(f"#### {qname}")
        _picked = st.multiselect(
            f"Months in {qname}",
            options=_pretty_months,
            default=default_months,
            key=f"spl_v1710_qmonths_{qkey}",
        )
        summary, table_df, this_avg_tp, raw_qdf = _quarter_table(
            qname, _picked, prev_avg_tp
        )

        # ---- Compact quarter-level header (smaller than st.metric
        # so the full numbers stay visible in narrow columns). ----
        # We render three small label/value pairs in a single
        # st.markdown call instead of three st.metric widgets;
        # st.metric renders huge values and truncates with "…" in
        # narrow columns, which was the original complaint.
        _hv = _fmt_inr_compact(summary["total_value"])
        _hvol = f"{summary['total_vol']:,}"
        _hout = f"{summary['total_outlets']:,}"
        st.markdown(
            f"""
            <div style="display:flex; gap:0.9rem; flex-wrap:wrap;
                        margin-top:0.1rem; margin-bottom:0.4rem;">
              <div style="flex:1; min-width:6.5rem;
                          padding:0.35rem 0.6rem;
                          background:rgba(255,255,255,0.04);
                          border-radius:0.4rem;">
                <div style="font-size:0.72rem; color:#9CA3AF;
                            letter-spacing:0.02em;">Total value</div>
                <div style="font-size:1.0rem; font-weight:600;
                            color:#FFFFFF;">{_hv}</div>
              </div>
              <div style="flex:1; min-width:6.5rem;
                          padding:0.35rem 0.6rem;
                          background:rgba(255,255,255,0.04);
                          border-radius:0.4rem;">
                <div style="font-size:0.72rem; color:#9CA3AF;
                            letter-spacing:0.02em;">Total pieces</div>
                <div style="font-size:1.0rem; font-weight:600;
                            color:#FFFFFF;">{_hvol}</div>
              </div>
              <div style="flex:1; min-width:6.5rem;
                          padding:0.35rem 0.6rem;
                          background:rgba(255,255,255,0.04);
                          border-radius:0.4rem;">
                <div style="font-size:0.72rem; color:#9CA3AF;
                            letter-spacing:0.02em;">Outlets selling</div>
                <div style="font-size:1.0rem; font-weight:600;
                            color:#FFFFFF;">{_hout}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if table_df is None or table_df.empty:
            st.info(
                "No SKUs with positive TP × MRP × GM (and ≥ "
                f"{q_min_vol} units) in this month selection."
            )
        else:
            # Total Value and GM Value are stored as raw floats
            # so column-header sort works (string-formatted
            # values would sort alphabetically and put
            # "₹1.76 Cr" before "₹47.88 L"). Streamlit's
            # NumberColumn renders them as ₹-prefixed numbers
            # with comma grouping. The printf format string
            # `₹%,d` uses Streamlit/D3's localised number format
            # to produce e.g. ₹4,788,000.
            _rupee_cols = {}
            if "Total Value" in table_df.columns:
                _rupee_cols["Total Value"] = (
                    st.column_config.NumberColumn(
                        "Total Value",
                        format="₹%,d",
                        help="Total Pieces × MRP over the "
                             "selected months (₹).",
                    )
                )
            if "GM Value" in table_df.columns:
                _rupee_cols["GM Value"] = (
                    st.column_config.NumberColumn(
                        "GM Value",
                        format="₹%,d",
                        help="Total Pieces × MRP × GM index "
                             "over the selected months (₹).",
                    )
                )
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                height=min(440, 40 + 32 * len(table_df)),
                column_config=_rupee_cols or None,
            )

        return this_avg_tp, raw_qdf, _picked, summary

    # Two rows of two quarter blocks each plus a third row for
    # Q5 (the spec asked for "four lists with two remaining side
    # by side"; Q5 was added later and gets its own row).
    qrow1_l, qrow1_r = st.columns(2)
    qrow2_l, qrow2_r = st.columns(2)
    qrow3_l, qrow3_r = st.columns(2)

    with qrow1_l:
        q1_avg_tp, q1_qdf, q1_months, q1_summary = (
            _render_quarter_block(
                "Q1", "q1", _q_defaults[0], prev_avg_tp=None
            )
        )
    with qrow1_r:
        q2_avg_tp, q2_qdf, q2_months, q2_summary = (
            _render_quarter_block(
                "Q2", "q2", _q_defaults[1], prev_avg_tp=q1_avg_tp
            )
        )
    with qrow2_l:
        q3_avg_tp, q3_qdf, q3_months, q3_summary = (
            _render_quarter_block(
                "Q3", "q3", _q_defaults[2], prev_avg_tp=q2_avg_tp
            )
        )
    with qrow2_r:
        q4_avg_tp, q4_qdf, q4_months, q4_summary = (
            _render_quarter_block(
                "Q4", "q4", _q_defaults[3], prev_avg_tp=q3_avg_tp
            )
        )
    with qrow3_l:
        q5_avg_tp, q5_qdf, q5_months, q5_summary = (
            _render_quarter_block(
                "Q5", "q5", _q_defaults[4], prev_avg_tp=q4_avg_tp
            )
        )

    # =====================================================
    # Excel export — five quarters side-by-side, colour-coded
    # =====================================================
    st.markdown(" ")
    st.markdown("##### 📥 Export quarterly lists to Excel")
    st.caption(
        "Generates a single .xlsx with all five quarters laid out "
        "side-by-side, colour-coded by bucket (🟢 Top / 🟡 Mid / "
        "🔴 Bottom) and tag (Big Gain / Gainer / Stable / Loser / "
        "Big Drop). Includes the TP × MRP × GM score column."
    )

    _quarter_payloads = [
        ("Q1", q1_qdf, q1_months, q1_summary),
        ("Q2", q2_qdf, q2_months, q2_summary),
        ("Q3", q3_qdf, q3_months, q3_summary),
        ("Q4", q4_qdf, q4_months, q4_summary),
        ("Q5", q5_qdf, q5_months, q5_summary),
    ]

    def _build_quarterly_xlsx(payloads):
        """
        Builds the multi-quarter Excel workbook in memory and
        returns the bytes. Each quarter occupies an 11-column band:
            #  Bucket  SKU  Brand  Pen%  Avg TP/mo  Total Vol
            Total Value (₹)  GM Value (₹)  TP×MRP×GM  Tag
        Five bands sit side-by-side (separated by a blank column)
        on a single sheet. Cell colouring:
          • Bucket cell  → green / yellow / red tint
          • Tag cell     → tag colour from _tag_color
        Each quarter also gets a header strip with the month
        selection and the three slice metrics (value / volume /
        outlets), and a subtotal row at the end of each bucket.
        """
        from openpyxl import Workbook
        from openpyxl.styles import (
            Font, PatternFill, Alignment, Border, Side,
        )
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        ws = wb.active
        ws.title = "Quarterly Comparison"

        # ---- Style palette ----
        font_title = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
        font_header = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        font_body = Font(name="Calibri", size=10, color="000000")
        font_body_bold = Font(name="Calibri", size=10, bold=True, color="000000")
        font_meta = Font(name="Calibri", size=9, italic=True, color="424242")

        fill_title = PatternFill(
            "solid", start_color="0F172A", end_color="0F172A"
        )
        fill_header = PatternFill(
            "solid", start_color="334155", end_color="334155"
        )
        fill_meta = PatternFill(
            "solid", start_color="F1F5F9", end_color="F1F5F9"
        )
        fill_subtotal = PatternFill(
            "solid", start_color="E2E8F0", end_color="E2E8F0"
        )
        fill_top = PatternFill(
            "solid", start_color="C6EFCE", end_color="C6EFCE"
        )
        fill_mid = PatternFill(
            "solid", start_color="FFEB9C", end_color="FFEB9C"
        )
        fill_bot = PatternFill(
            "solid", start_color="FFC7CE", end_color="FFC7CE"
        )
        bucket_fill = {
            "Top (30%)":    fill_top,
            "Mid (30%)":    fill_mid,
            "Bottom (40%)": fill_bot,
        }

        thin = Side(border_style="thin", color="CBD5E1")
        cell_border = Border(left=thin, right=thin, top=thin, bottom=thin)

        align_center = Alignment(horizontal="center", vertical="center")
        align_left = Alignment(horizontal="left", vertical="center")
        align_right = Alignment(horizontal="right", vertical="center")

        # ---- Layout: 5 quarters × 11 columns + 1 spacer column ----
        QCOLS = 11
        SPACER = 1
        BAND = QCOLS + SPACER  # 12 cols per band, last has no spacer

        headers = [
            "#", "Bucket", "SKU", "Brand", "Pen %",
            "Avg TP/mo", "Total Vol", "Total Value (₹)",
            "GM Value (₹)", "TP×MRP×GM", "Tag",
        ]

        # Row plan:
        #  row 1: workbook title
        #  row 2: quarter banner (Q1 / Q2 / Q3 / Q4 / Q5)
        #  row 3: month selection
        #  row 4: total value
        #  row 5: total volume
        #  row 6: outlets selling
        #  row 7: column headers
        #  row 8+ : data

        # ---- Row 1: workbook title spanning all 5 bands ----
        ws.cell(row=1, column=1, value="SKU Priority Lister — Quarterly Comparison")
        ws.cell(row=1, column=1).font = font_title
        ws.cell(row=1, column=1).fill = fill_title
        ws.cell(row=1, column=1).alignment = align_left
        ws.merge_cells(
            start_row=1, start_column=1,
            end_row=1, end_column=BAND * 5 - SPACER,
        )
        ws.row_dimensions[1].height = 22

        # ---- Quarter banners + metadata ----
        for qi, (qname, _qdf, qmonths, qsummary) in enumerate(payloads):
            c0 = 1 + qi * BAND  # 1, 12, 23, 34, 45
            # Quarter banner (row 2)
            ws.cell(row=2, column=c0, value=qname).font = font_title
            ws.cell(row=2, column=c0).fill = fill_header
            ws.cell(row=2, column=c0).alignment = align_center
            ws.merge_cells(
                start_row=2, start_column=c0,
                end_row=2, end_column=c0 + QCOLS - 1,
            )
            ws.row_dimensions[2].height = 20

            # Months row (row 3)
            _month_str = (
                ", ".join(qmonths) if qmonths else "(no months)"
            )
            ws.cell(
                row=3, column=c0,
                value=f"Months: {_month_str}",
            ).font = font_meta
            ws.cell(row=3, column=c0).fill = fill_meta
            ws.cell(row=3, column=c0).alignment = align_left
            ws.merge_cells(
                start_row=3, start_column=c0,
                end_row=3, end_column=c0 + QCOLS - 1,
            )

            # Total value (row 4)
            ws.cell(
                row=4, column=c0,
                value="Total value (₹):",
            ).font = font_meta
            ws.cell(row=4, column=c0).fill = fill_meta
            ws.cell(row=4, column=c0).alignment = align_left
            ws.merge_cells(
                start_row=4, start_column=c0,
                end_row=4, end_column=c0 + 4,
            )
            ws.cell(
                row=4, column=c0 + 5,
                value=float(qsummary["total_value"]),
            ).font = font_body_bold
            ws.cell(row=4, column=c0 + 5).fill = fill_meta
            ws.cell(row=4, column=c0 + 5).alignment = align_right
            ws.cell(row=4, column=c0 + 5).number_format = (
                '"₹"#,##0;[Red]-"₹"#,##0;-'
            )
            ws.merge_cells(
                start_row=4, start_column=c0 + 5,
                end_row=4, end_column=c0 + QCOLS - 1,
            )

            # Total pieces (row 5)
            ws.cell(
                row=5, column=c0,
                value="Total pieces (units):",
            ).font = font_meta
            ws.cell(row=5, column=c0).fill = fill_meta
            ws.cell(row=5, column=c0).alignment = align_left
            ws.merge_cells(
                start_row=5, start_column=c0,
                end_row=5, end_column=c0 + 4,
            )
            ws.cell(
                row=5, column=c0 + 5,
                value=int(qsummary["total_vol"]),
            ).font = font_body_bold
            ws.cell(row=5, column=c0 + 5).fill = fill_meta
            ws.cell(row=5, column=c0 + 5).alignment = align_right
            ws.cell(row=5, column=c0 + 5).number_format = "#,##0"
            ws.merge_cells(
                start_row=5, start_column=c0 + 5,
                end_row=5, end_column=c0 + QCOLS - 1,
            )

            # Outlets selling (row 6)
            ws.cell(
                row=6, column=c0,
                value="Outlets selling:",
            ).font = font_meta
            ws.cell(row=6, column=c0).fill = fill_meta
            ws.cell(row=6, column=c0).alignment = align_left
            ws.merge_cells(
                start_row=6, start_column=c0,
                end_row=6, end_column=c0 + 4,
            )
            ws.cell(
                row=6, column=c0 + 5,
                value=int(qsummary["total_outlets"]),
            ).font = font_body_bold
            ws.cell(row=6, column=c0 + 5).fill = fill_meta
            ws.cell(row=6, column=c0 + 5).alignment = align_right
            ws.cell(row=6, column=c0 + 5).number_format = "#,##0"
            ws.merge_cells(
                start_row=6, start_column=c0 + 5,
                end_row=6, end_column=c0 + QCOLS - 1,
            )

            # Column headers (row 7)
            for hi, htxt in enumerate(headers):
                cell = ws.cell(row=7, column=c0 + hi, value=htxt)
                cell.font = font_header
                cell.fill = fill_header
                cell.alignment = align_center
                cell.border = cell_border
            ws.row_dimensions[7].height = 22

        # ---- Data rows for each quarter ----
        # Pad shorter quarters with blank cells so all four bands
        # stay visually aligned.
        data_start_row = 8

        # Build per-quarter row payload: list of dicts in the
        # order they should appear (data + subtotals).
        def _band_rows(qdf):
            if qdf is None or qdf.empty:
                return []
            out = []
            rank_pos = 0
            for bname in _bucket_order:
                bdf = qdf[qdf["bucket"] == bname]
                if bdf.empty:
                    continue
                for sku, r in bdf.iterrows():
                    rank_pos += 1
                    out.append({
                        "kind": "data",
                        "rank": rank_pos,
                        "bucket": bname,
                        "sku": sku,
                        "brand": r.get("brand", ""),
                        "pen_pct": float(r["pen_pct"]),
                        "avg_tp": float(r["avg_monthly_tp"]),
                        "total_vol": float(r["total_vol"]),
                        "total_value": float(r["total_value"]),
                        "gm_value": float(r["gm_value"]),
                        "tp_mrp_gm": float(r["tp_mrp_gm"]),
                        "tag": r.get("tag", "") or "",
                    })
                out.append({
                    "kind": "subtotal",
                    "bucket": bname,
                    "label": f"Subtotal {bname} ({len(bdf)} SKUs)",
                    "avg_tp": float(bdf["avg_monthly_tp"].sum()),
                    "total_vol": float(bdf["total_vol"].sum()),
                    "total_value": float(bdf["total_value"].sum()),
                    "gm_value": float(bdf["gm_value"].sum()),
                    "tp_mrp_gm": float(bdf["tp_mrp_gm"].sum()),
                })
            return out

        bands = [_band_rows(p[1]) for p in payloads]
        max_rows = max((len(b) for b in bands), default=0)

        for qi, band in enumerate(bands):
            c0 = 1 + qi * BAND
            for ri in range(max_rows):
                row_excel = data_start_row + ri
                if ri >= len(band):
                    # Pad with empty bordered cells.
                    for ci in range(QCOLS):
                        ws.cell(row=row_excel, column=c0 + ci).border = cell_border
                    continue
                rec = band[ri]
                if rec["kind"] == "data":
                    bname = rec["bucket"]
                    cells = [
                        rec["rank"],
                        bname,
                        rec["sku"],
                        rec["brand"],
                        rec["pen_pct"],
                        rec["avg_tp"],
                        rec["total_vol"],
                        rec["total_value"],
                        rec["gm_value"],
                        rec["tp_mrp_gm"],
                        rec["tag"],
                    ]
                    for ci, val in enumerate(cells):
                        cell = ws.cell(row=row_excel, column=c0 + ci, value=val)
                        cell.font = font_body
                        cell.border = cell_border
                        if ci == 0:  # rank
                            cell.alignment = align_center
                        elif ci in (1, 2, 3):  # bucket, sku, brand
                            cell.alignment = align_left
                        else:
                            cell.alignment = align_right
                    # Number formats
                    ws.cell(row=row_excel, column=c0 + 4).number_format = "0.00"
                    ws.cell(row=row_excel, column=c0 + 5).number_format = "0.00"
                    ws.cell(row=row_excel, column=c0 + 6).number_format = "#,##0"
                    ws.cell(row=row_excel, column=c0 + 7).number_format = (
                        '"₹"#,##0;[Red]-"₹"#,##0;-'
                    )
                    ws.cell(row=row_excel, column=c0 + 8).number_format = (
                        '"₹"#,##0;[Red]-"₹"#,##0;-'
                    )
                    ws.cell(row=row_excel, column=c0 + 9).number_format = "#,##0.00"
                    # Colour the Bucket cell
                    bcell = ws.cell(row=row_excel, column=c0 + 1)
                    bcell.fill = bucket_fill.get(bname, fill_meta)
                    bcell.font = font_body_bold
                    # Colour the Tag cell
                    tcell = ws.cell(row=row_excel, column=c0 + 10)
                    _tag_label = rec["tag"]
                    if _tag_label and _tag_label in _tag_color:
                        tcell.fill = PatternFill(
                            "solid",
                            start_color=_tag_color[_tag_label],
                            end_color=_tag_color[_tag_label],
                        )
                        tcell.font = Font(
                            name="Calibri", size=10,
                            bold=True, color="FFFFFF",
                        )
                        tcell.alignment = align_center
                elif rec["kind"] == "subtotal":
                    # Subtotal row — bucket-tinted, bold.
                    bname = rec["bucket"]
                    cells = [
                        "—",
                        rec["label"],
                        "", "",
                        "",
                        rec["avg_tp"],
                        rec["total_vol"],
                        rec["total_value"],
                        rec["gm_value"],
                        rec["tp_mrp_gm"],
                        "",
                    ]
                    for ci, val in enumerate(cells):
                        cell = ws.cell(row=row_excel, column=c0 + ci, value=val)
                        cell.font = font_body_bold
                        cell.fill = fill_subtotal
                        cell.border = cell_border
                        if ci == 0:
                            cell.alignment = align_center
                        elif ci in (1, 2, 3):
                            cell.alignment = align_left
                        else:
                            cell.alignment = align_right
                    ws.cell(row=row_excel, column=c0 + 5).number_format = "0.00"
                    ws.cell(row=row_excel, column=c0 + 6).number_format = "#,##0"
                    ws.cell(row=row_excel, column=c0 + 7).number_format = (
                        '"₹"#,##0;[Red]-"₹"#,##0;-'
                    )
                    ws.cell(row=row_excel, column=c0 + 8).number_format = (
                        '"₹"#,##0;[Red]-"₹"#,##0;-'
                    )
                    ws.cell(row=row_excel, column=c0 + 9).number_format = "#,##0.00"

        # ---- Column widths ----
        # Per-band widths (matched across all 5 bands).
        # 11 cols: #, Bucket, SKU, Brand, Pen%, Avg TP/mo,
        # Total Vol, Total Value (₹), GM Value (₹), TP×MRP×GM, Tag
        col_widths = [5, 13, 28, 16, 8, 11, 11, 16, 16, 13, 12]
        for qi in range(5):
            c0 = 1 + qi * BAND
            for ci, w in enumerate(col_widths):
                col_letter = get_column_letter(c0 + ci)
                ws.column_dimensions[col_letter].width = w
            # Spacer column
            if qi < 4:
                spacer_letter = get_column_letter(c0 + QCOLS)
                ws.column_dimensions[spacer_letter].width = 2

        # ---- Freeze the header rows + first quarter's first cols ----
        ws.freeze_panes = "A8"

        # ---- Legend sheet ----
        ws2 = wb.create_sheet("Legend")
        ws2["A1"] = "Bucket colours"
        ws2["A1"].font = Font(bold=True, size=12)
        for ri, (bname, bfill) in enumerate(
            [
                ("Top (30%)", fill_top),
                ("Mid (30%)", fill_mid),
                ("Bottom (40%)", fill_bot),
            ],
            start=2,
        ):
            ws2.cell(row=ri, column=1, value=bname).fill = bfill
            ws2.cell(row=ri, column=1).font = Font(bold=True)

        ws2["A7"] = "Tag colours"
        ws2["A7"].font = Font(bold=True, size=12)
        for ri, (tname, hex_) in enumerate(_tag_color.items(), start=8):
            cell = ws2.cell(row=ri, column=1, value=tname)
            cell.fill = PatternFill(
                "solid", start_color=hex_, end_color=hex_
            )
            cell.font = Font(bold=True, color="FFFFFF")
        ws2.column_dimensions["A"].width = 22

        ws2["C1"] = "Tag thresholds (vs. previous quarter's avg TP)"
        ws2["C1"].font = Font(bold=True, size=12)
        ws2["C2"] = "Big Gain"
        ws2["D2"] = "≥ +50%"
        ws2["C3"] = "Gainer"
        ws2["D3"] = "+15% … +50%"
        ws2["C4"] = "Stable"
        ws2["D4"] = "−15% … +15%"
        ws2["C5"] = "Loser"
        ws2["D5"] = "−50% … −15%"
        ws2["C6"] = "Big Drop"
        ws2["D6"] = "≤ −50%"
        ws2["C7"] = "New"
        ws2["D7"] = "Prev = 0, Curr > 0"
        ws2["C8"] = "Lost"
        ws2["D8"] = "Prev > 0, Curr = 0"
        ws2.column_dimensions["C"].width = 14
        ws2.column_dimensions["D"].width = 22

        # ---- Write to bytes ----
        bio = io.BytesIO()
        wb.save(bio)
        return bio.getvalue()

    try:
        _xlsx_bytes = _build_quarterly_xlsx(_quarter_payloads)
        st.download_button(
            "📊 Download Q1–Q5 comparison (.xlsx)",
            data=_xlsx_bytes,
            file_name="sku_priority_lister_Q1_Q5.xlsx",
            mime=(
                "application/"
                "vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="spl_v1710_q_xlsx",
            use_container_width=False,
        )
    except Exception as _xlsx_err:
        st.error(
            f"Couldn't build the Excel file: {_xlsx_err}. "
            "If openpyxl isn't installed in your environment, "
            "`pip install openpyxl` and reload."
        )

    # =====================================================
    # (v) BRAND-WISE BEST SKU RECOMMENDATIONS
    #     (ranked by TP × MRP × GM index)
    # =====================================================
    # For each Brand in the working slice, surface the top-N
    # SKUs ranked by **Rank Score** = Throughput × Latest MRP ×
    # GM index, where:
    #   • Throughput = total pieces in scope ÷ active outlets ÷
    #                  # months in scope  (per-outlet-per-month)
    #   • Latest MRP = most recent MRP for the SKU (₹)
    #   • GM index   = from the optional GM file
    # GM Value (= Total Value × GM index) is kept as a reported
    # column for context, but the ranking now follows
    # TP × MRP × GM as requested.
    #
    # Constraints (per user spec):
    #   • Only Large and Medium SKUs are considered (Small SKUs
    #     are excluded). Requires the SKU-Size file. If not
    #     uploaded, an info panel explains how to enable.
    #   • Quarter / months scope selector — narrows the period
    #     used for throughput, pieces, value, GM Value. Default
    #     = All months in the current slice. The user can pick
    #     a custom set of months OR jump to one of the Q1..Q5
    #     presets the Quarterly-comparison block defined above.
    #   • Minimum pieces + minimum penetration sliders below
    #     filter eligibility before ranking.
    #   • Separate ranked list per brand.
    #
    # Output:
    #   • One expandable table per brand on screen.
    #   • A multi-sheet Excel download — one sheet per brand
    #     plus an "All Brands" combined sheet — with per-SKU
    #     TP/MRP, GM index, penetration, pieces, value, GM
    #     value, rank score, and rank columns.
    st.divider()
    st.subheader(
        "⭐ Best-SKU recommendations by Brand "
        "(TP × MRP × GM ranked)"
    )
    st.caption(
        "For each brand, the top SKUs ranked by "
        "**TP × MRP × GM index** "
        "(Throughput × Latest MRP × Gross Margin index). "
        "Only **Large** and **Medium** SKUs are considered. "
        "Pick a quarter / month scope, then refine with the "
        "pieces and penetration filters below. Inherits every "
        "sidebar + tab-level filter applied above."
    )

    if not _sku_size_available or _sku_size_lookup is None:
        st.info(
            "Brand-wise recommendations need the **SKU-Size** "
            "file to filter Large / Medium SKUs. Upload it in "
            "the sidebar (`Optional: SKU-Size file`) and this "
            "section will populate automatically."
        )
    else:
        # ---- Quarter / month scope selector ----
        # The user can narrow the period used for throughput,
        # pieces, value and GM Value to a custom set of months,
        # or jump to one of the Q1..Q5 presets (which mirror the
        # default quarter splits used by the Quarterly
        # comparison block above). This is independent of the
        # tab-level filters — it just narrows the time window
        # within the already-filtered slice.
        #
        # Presets (`_q_defaults`) come from the same
        # `_default_quarter_splits(_pretty_months)` call used
        # above for the Quarterly comparison block, so the
        # quarter buckets here line up 1:1 with what the user
        # sees there.
        _quarter_preset_options = [
            "All months",
            "Q1 (auto)",
            "Q2 (auto)",
            "Q3 (auto)",
            "Q4 (auto)",
            "Q5 (auto)",
            "Custom",
        ]

        qs1, qs2 = st.columns([1, 2])
        with qs1:
            rec_quarter_preset = st.selectbox(
                "Quarter scope",
                options=_quarter_preset_options,
                index=0,
                key="spl_rec_quarter_preset",
                help=(
                    "Restrict the ranking period to a quarter "
                    "(Q1..Q5 use the same default month-splits "
                    "as the Quarterly comparison block above), "
                    "or pick a custom set of months. "
                    "'All months' uses the full slice."
                ),
            )

        # Default month selection follows the preset.
        if rec_quarter_preset == "All months":
            _preset_months = list(_pretty_months)
        elif rec_quarter_preset.startswith("Q1"):
            _preset_months = list(_q_defaults[0])
        elif rec_quarter_preset.startswith("Q2"):
            _preset_months = list(_q_defaults[1])
        elif rec_quarter_preset.startswith("Q3"):
            _preset_months = list(_q_defaults[2])
        elif rec_quarter_preset.startswith("Q4"):
            _preset_months = list(_q_defaults[3])
        elif rec_quarter_preset.startswith("Q5"):
            _preset_months = list(_q_defaults[4])
        else:  # Custom
            _preset_months = list(_pretty_months)

        with qs2:
            # Custom = freely editable; presets = display-only.
            if rec_quarter_preset == "Custom":
                rec_months_pretty = st.multiselect(
                    "Months in scope",
                    options=_pretty_months,
                    default=_preset_months,
                    key="spl_rec_months_custom",
                    help=(
                        "Pieces, value, and throughput are "
                        "computed over these months only. The "
                        "ranking score "
                        "(TP × MRP × GM) uses the same period."
                    ),
                )
            else:
                st.multiselect(
                    "Months in scope",
                    options=_pretty_months,
                    default=_preset_months,
                    key=(
                        f"spl_rec_months_display_"
                        f"{rec_quarter_preset}"
                    ),
                    help=(
                        "Read-only — switch 'Quarter scope' to "
                        "Custom to edit."
                    ),
                    disabled=True,
                )
                rec_months_pretty = list(_preset_months)

        if not rec_months_pretty:
            st.warning(
                "Pick at least one month in scope to see "
                "recommendations."
            )
            return

        # Map pretty month labels (e.g. "Jan25") back to the raw
        # volume column names (e.g. "Jan25_V").
        rec_months_in_data = [
            _pretty_to_raw[m]
            for m in rec_months_pretty
            if m in _pretty_to_raw
        ]
        rec_n_months = len(rec_months_in_data)

        # ---- Per-recommender controls ----
        rc1, rc2, rc3 = st.columns([2, 2, 2])
        with rc1:
            rec_top_n = st.slider(
                "Top N SKUs per brand",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                key="spl_rec_top_n",
                help=(
                    "How many SKUs to recommend per brand "
                    "(after applying the filters below)."
                ),
            )
        with rc2:
            rec_min_vol = st.slider(
                "Minimum total pieces",
                min_value=0,
                max_value=5000,
                value=100,
                step=50,
                key="spl_rec_min_vol",
                help=(
                    "SKUs whose total pieces across the "
                    f"{rec_n_months}-month scope falls below "
                    "this threshold are dropped from the brand "
                    "ranking."
                ),
            )
        with rc3:
            rec_min_pen = st.slider(
                "Minimum penetration %",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                key="spl_rec_min_pen",
                help=(
                    "SKUs whose penetration % (active outlets "
                    "/ total outlets in slice) falls below "
                    "this threshold are dropped from the brand "
                    "ranking."
                ),
            )

        # Optional size scope override — defaults to Large +
        # Medium per the user spec but exposed so the user can
        # narrow it further (e.g. only Large) if desired.
        rec_size_scope = st.multiselect(
            "Size scope",
            options=["Large", "Medium"],
            default=["Large", "Medium"],
            key="spl_rec_size_scope",
            help=(
                "Only SKUs of these sizes are eligible. Small "
                "SKUs are never included in brand "
                "recommendations."
            ),
        )

        # ---- Build a per-SKU recommender table ----
        # Built ONCE here (outside the size-scope branch) so that
        # the Small SKUs section below can reuse it independently
        # of `rec_size_scope` (which only constrains the brand-
        # wise list). Metrics are rebuilt over the chosen month
        # scope so throughput, pieces, value and GM Value all
        # reflect the picked quarter. When 'All months' is chosen,
        # rec_months_in_data == months_in_data and the numbers
        # match the rank_df headline figures.
        _scope_vol = (
            work.groupby(COL_SKU)[rec_months_in_data]
            .sum()
        )
        # Per-SKU total pieces in scope.
        _scope_total_vol = _scope_vol.sum(axis=1)

        # Per-SKU active outlets within the month scope:
        # outlets that sold the SKU in at least one of the
        # picked months.
        _scope_period_vol = (
            work[rec_months_in_data].astype(float).sum(axis=1)
        )
        _scope_active_mask = _scope_period_vol > 0
        _scope_active_outlets = (
            work[_scope_active_mask]
            .groupby(COL_SKU)[COL_OUTLET]
            .nunique()
            .reindex(_scope_vol.index)
            .fillna(0)
            .astype(int)
        )

        # Per-SKU total value in scope = Σ (monthly vol × monthly MRP).
        # Use the sku_monthly_value table built up-tab; if a
        # picked month column isn't present (defensive), treat
        # it as zero.
        _scope_value_cols = [
            c for c in rec_months_in_data
            if c in sku_monthly_value.columns
        ]
        if _scope_value_cols:
            _scope_total_value = (
                sku_monthly_value[_scope_value_cols]
                .reindex(_scope_vol.index)
                .sum(axis=1)
            )
        else:
            _scope_total_value = pd.Series(
                0.0, index=_scope_vol.index
            )

        # Total outlets in the working slice (denominator for
        # penetration). Penetration uses the same denominator
        # as the rest of the tab — total outlets across the
        # full slice — so it's comparable across quarters.
        _scope_total_outlets = int(work[COL_OUTLET].nunique())

        rec_df = pd.DataFrame(index=_scope_vol.index)
        rec_df["total_vol"] = _scope_total_vol.astype(float)
        rec_df["active_outlets"] = (
            _scope_active_outlets.astype(int)
        )
        rec_df["latest_mrp"] = (
            sku_latest_mrp.reindex(rec_df.index)
        )
        rec_df["gm_index"] = (
            sku_gm_index.reindex(rec_df.index)
        )
        rec_df["total_value"] = (
            _scope_total_value.astype(float)
        )

        # Throughput in the scope window — aligned with the
        # month-wise Throughput table shown above in section (iii).
        # That table is built as:
        #     monthly_throughput[sku, month] =
        #         sku_monthly_vol[sku, month] / sku_active_outlets[sku]
        # where `sku_active_outlets` is the count of distinct outlets
        # that sold the SKU in *any* month of the full slice (NOT
        # scope-restricted). The displayed "TP / month" for the
        # brand-wise recommender must therefore equal the simple
        # average of those monthly-throughput values across the
        # picked scope months, so the two views never disagree.
        #
        # Equivalently:
        #     TP / month =
        #         Σ_{m ∈ scope} sku_monthly_vol[sku, m]
        #         ÷ sku_active_outlets_full_slice[sku]
        #         ÷ n_scope_months
        #
        # Note the denominator uses `sku_active_outlets` (the
        # full-slice active outlets defined when rank_df was built
        # above), NOT `_scope_active_outlets`. That is the explicit
        # contract with the month-wise throughput table per user
        # spec: throughputs must match for every month.
        _full_slice_active_outlets = (
            sku_active_outlets
            .reindex(rec_df.index)
            .fillna(0)
            .astype(float)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rec_df["throughput"] = np.where(
                (_full_slice_active_outlets > 0)
                & (rec_n_months > 0),
                rec_df["total_vol"]
                / _full_slice_active_outlets.replace(0, np.nan)
                / max(rec_n_months, 1),
                0.0,
            )
        rec_df["throughput"] = (
            rec_df["throughput"].fillna(0.0)
        )

        # Penetration % (active outlets / total outlets in slice).
        if _scope_total_outlets > 0:
            rec_df["penetration_pct"] = (
                rec_df["active_outlets"]
                / _scope_total_outlets
                * 100.0
            )
        else:
            rec_df["penetration_pct"] = 0.0

        # GM Value reported for context (Total Value × GM index).
        rec_df["gm_value"] = (
            rec_df["total_value"].fillna(0.0)
            * rec_df["gm_index"].fillna(0.0)
        )

        # Ranking score per user spec:
        #   Rank Score = Throughput × Latest MRP × GM index.
        # NaN GM / MRP collapses the score to 0 (those SKUs
        # naturally sink to the bottom and get filtered out
        # by the score > 0 check below).
        rec_df["rank_score"] = (
            rec_df["throughput"].fillna(0.0)
            * rec_df["latest_mrp"].fillna(0.0)
            * rec_df["gm_index"].fillna(0.0)
        )

        # Drop SKUs with no positive ranking signal at all
        # in the scope (no pieces / no MRP / no GM).
        rec_df = rec_df[rec_df["rank_score"] > 0]

        # Per-SKU brand (one brand per SKU in the slice).
        _brand_per_sku_rec = (
            work.groupby(COL_SKU)[COL_BRAND]
            .agg(
                lambda s: (
                    s.dropna().astype(str).iloc[0]
                    if not s.dropna().empty else ""
                )
            )
        )
        rec_df["brand"] = (
            _brand_per_sku_rec.reindex(rec_df.index)
            .fillna("")
            .astype(str)
        )

        # Per-SKU size (Large / Medium / Small / unknown).
        _sku_keys_rec = (
            rec_df.index.astype(str).str.strip().str.lower()
        )
        _size_series_rec = pd.Series(
            _sku_size_lookup.reindex(_sku_keys_rec.values).values,
            index=rec_df.index,
            name="size",
        )
        rec_df["size"] = _size_series_rec

        if not rec_size_scope:
            st.warning(
                "Pick at least one size (Large and/or Medium) "
                "to see brand-wise recommendations. The Small "
                "SKUs ranked list below is unaffected."
            )
            # Stub out the brand-wise outputs so the Small list
            # section further down still has the names it
            # checks (`per_brand_raw`, `rec_df_filtered`,
            # `rec_df_sized`).
            rec_df_sized = rec_df.iloc[0:0].copy()
            rec_df_filtered = rec_df.iloc[0:0].copy()
            brands_present = []
            per_brand_raw = {}
        else:
            # Drop SKUs we can't classify by size — brand-wise
            # recommender is explicit about Large/Medium only.
            rec_df_sized = rec_df[
                rec_df["size"].isin(rec_size_scope)
            ].copy()

            # Apply min-pieces + min-penetration filters.
            rec_df_filtered = rec_df_sized[
                (rec_df_sized["total_vol"] >= rec_min_vol)
                & (rec_df_sized["penetration_pct"] >= rec_min_pen)
                & (rec_df_sized["rank_score"] > 0)
            ].copy()

            # Diagnostic counts for transparency.
            _diag_c1, _diag_c2, _diag_c3 = st.columns(3)
            _diag_c1.metric(
                "SKUs in scope",
                f"{len(rec_df):,}",
                help=(
                    "SKUs surviving the sidebar + tab filters "
                    "with a positive TP × MRP × GM score in "
                    f"the {rec_n_months}-month scope."
                ),
            )
            _diag_c2.metric(
                "After size filter",
                f"{len(rec_df_sized):,}",
                help=(
                    "SKUs after restricting to "
                    f"{', '.join(rec_size_scope)}."
                ),
            )
            _diag_c3.metric(
                "After pieces + pen filters",
                f"{len(rec_df_filtered):,}",
                help=(
                    f"SKUs with ≥ {rec_min_vol:,} pieces and "
                    f"≥ {rec_min_pen}% penetration."
                ),
            )

            if rec_df_filtered.empty:
                st.warning(
                    "No SKUs pass the size + pieces + "
                    "penetration filters. Loosen one of them "
                    "to see brand-wise recommendations."
                )
            else:
                # Rank within each brand by **Rank Score**
                # (TP × MRP × GM).
                rec_df_filtered = rec_df_filtered.sort_values(
                    ["brand", "rank_score"],
                    ascending=[True, False],
                )

                # Brands present, sorted by total Rank Score
                # contribution (biggest brand first).
                _brand_totals = (
                    rec_df_filtered.groupby("brand")["rank_score"]
                    .sum()
                    .sort_values(ascending=False)
                )
                # Brand-level GM Value totals — still reported in
                # the expander headers for context.
                _brand_gm_totals = (
                    rec_df_filtered.groupby("brand")["gm_value"]
                    .sum()
                )
                brands_present = [
                    b for b in _brand_totals.index
                    if str(b).strip() != ""
                ]

                # ---- On-screen: one expandable table per brand ----
                # Build the per-brand "top N" tables once and
                # cache them so the Excel export can reuse them.
                per_brand_tables = {}
                per_brand_raw = {}

                for _bname in brands_present:
                    _bdf = (
                        rec_df_filtered[
                            rec_df_filtered["brand"] == _bname
                        ]
                        .head(rec_top_n)
                        .copy()
                    )
                    if _bdf.empty:
                        continue

                    # Build the display rows.
                    _rows = []
                    for _rank, (_sku, _r) in enumerate(
                        _bdf.iterrows(), start=1
                    ):
                        _rows.append({
                            "Rank": _rank,
                            "SKU": _sku,
                            "Size": _r["size"],
                            "Penetration %": round(
                                float(_r["penetration_pct"]), 2
                            ),
                            "TP / month": round(
                                float(_r["throughput"]), 2
                            ),
                            "Latest MRP (₹)": (
                                round(float(_r["latest_mrp"]), 2)
                                if pd.notna(_r["latest_mrp"])
                                else np.nan
                            ),
                            "GM Index": (
                                round(float(_r["gm_index"]), 4)
                                if pd.notna(_r["gm_index"])
                                else np.nan
                            ),
                            "Total Pieces": int(round(
                                float(_r["total_vol"])
                            )),
                            "Total Value (₹)": float(
                                _r["total_value"]
                            ),
                            "GM Value (₹)": float(
                                _r["gm_value"]
                            ),
                            "Rank Score (TP × MRP × GM)": float(
                                _r["rank_score"]
                            ),
                        })
                    _disp = pd.DataFrame(_rows)
                    per_brand_raw[_bname] = _disp.copy()

                    # Pretty-format rupee columns for screen.
                    _disp_screen = _disp.copy()
                    _disp_screen["Total Value (₹)"] = (
                        _disp_screen["Total Value (₹)"]
                        .apply(_fmt_inr_compact)
                    )
                    _disp_screen["GM Value (₹)"] = (
                        _disp_screen["GM Value (₹)"]
                        .apply(_fmt_inr_compact)
                    )
                    per_brand_tables[_bname] = _disp_screen

                # Render each brand in an expander. Default-open
                # for the first 3 biggest brands (by Rank Score)
                # so the user sees something immediately. The
                # label keeps the GM Value figure for context
                # (it's the metric many readers will be most
                # familiar with), but the ranking inside is by
                # Rank Score = TP × MRP × GM as requested.
                for _idx, _bname in enumerate(brands_present):
                    if _bname not in per_brand_tables:
                        continue
                    _tbl = per_brand_tables[_bname]
                    _brand_gm = float(
                        _brand_gm_totals.get(_bname, 0.0)
                    )
                    _n_in_brand = int(
                        (rec_df_filtered["brand"] == _bname).sum()
                    )
                    with st.expander(
                        f"🏷️ {_bname} — top {len(_tbl)} of "
                        f"{_n_in_brand} eligible SKUs "
                        f"(brand GM Value: "
                        f"{_fmt_inr_compact(_brand_gm)})",
                        expanded=(_idx < 3),
                    ):
                        st.dataframe(
                            _tbl,
                            use_container_width=True,
                            hide_index=True,
                            height=min(
                                480, 45 + 35 * len(_tbl)
                            ),
                        )

                # ---- Excel export ----
                def _build_brand_rec_xlsx(brand_tables_raw):
                    """
                    Build a multi-sheet Excel:
                      • One sheet per brand with the top-N table.
                      • An "All Brands" sheet stacking everything
                        with a leading Brand column.
                      • A "Filters" sheet documenting the filter
                        settings used to generate the report.
                    """
                    from openpyxl import Workbook
                    from openpyxl.styles import (
                        Font, PatternFill, Alignment, Border, Side
                    )
                    from openpyxl.utils import get_column_letter

                    wb = Workbook()
                    # Remove default sheet — we'll add our own.
                    wb.remove(wb.active)

                    thin = Side(
                        border_style="thin", color="BFBFBF"
                    )
                    cell_border = Border(
                        left=thin, right=thin,
                        top=thin, bottom=thin,
                    )
                    fill_header = PatternFill(
                        "solid",
                        start_color="305496",
                        end_color="305496",
                    )
                    fill_brand_row = PatternFill(
                        "solid",
                        start_color="D9E1F2",
                        end_color="D9E1F2",
                    )
                    font_header = Font(
                        bold=True, color="FFFFFF", size=11
                    )
                    font_brand = Font(bold=True, size=11)
                    align_center = Alignment(
                        horizontal="center", vertical="center"
                    )
                    align_right = Alignment(
                        horizontal="right", vertical="center"
                    )
                    align_left = Alignment(
                        horizontal="left", vertical="center"
                    )

                    # Column layout — matches per_brand_raw schema:
                    # Rank, SKU, Size, Penetration %, TP / month,
                    # Latest MRP (₹), GM Index, Total Pieces,
                    # Total Value (₹), GM Value (₹),
                    # Rank Score (TP × MRP × GM)
                    cols = [
                        "Rank", "SKU", "Size", "Penetration %",
                        "TP / month", "Latest MRP (₹)",
                        "GM Index", "Total Pieces",
                        "Total Value (₹)", "GM Value (₹)",
                        "Rank Score (TP × MRP × GM)",
                    ]
                    # Numeric formats per column index.
                    num_fmts = {
                        "Rank": "0",
                        "Penetration %": "0.00",
                        "TP / month": "0.00",
                        "Latest MRP (₹)": '"₹"#,##0.00',
                        "GM Index": "0.0000",
                        "Total Pieces": "#,##0",
                        "Total Value (₹)": (
                            '"₹"#,##0;[Red]-"₹"#,##0;-'
                        ),
                        "GM Value (₹)": (
                            '"₹"#,##0;[Red]-"₹"#,##0;-'
                        ),
                        "Rank Score (TP × MRP × GM)": "#,##0.00",
                    }
                    col_widths = {
                        "Rank": 6, "SKU": 36, "Size": 9,
                        "Penetration %": 13, "TP / month": 12,
                        "Latest MRP (₹)": 14, "GM Index": 10,
                        "Total Pieces": 13,
                        "Total Value (₹)": 16,
                        "GM Value (₹)": 16,
                        "Rank Score (TP × MRP × GM)": 22,
                        "Brand": 22,
                    }

                    def _safe_sheet_name(name):
                        # Excel sheet names: ≤ 31 chars, no
                        # []:*?/\\ characters.
                        bad = set('[]:*?/\\')
                        cleaned = "".join(
                            c for c in str(name) if c not in bad
                        ).strip()
                        if not cleaned:
                            cleaned = "Brand"
                        return cleaned[:31]

                    def _write_table(ws, df_raw, extra_brand_col=False):
                        """Write a brand table to a worksheet."""
                        header_cols = (
                            (["Brand"] if extra_brand_col else [])
                            + cols
                        )
                        # Header row.
                        for ci, cname in enumerate(
                            header_cols, start=1
                        ):
                            cell = ws.cell(
                                row=1, column=ci, value=cname
                            )
                            cell.fill = fill_header
                            cell.font = font_header
                            cell.alignment = align_center
                            cell.border = cell_border
                        # Data rows.
                        for ri, (_, r) in enumerate(
                            df_raw.iterrows(), start=2
                        ):
                            offset = 0
                            if extra_brand_col:
                                bcell = ws.cell(
                                    row=ri, column=1,
                                    value=r.get("Brand", ""),
                                )
                                bcell.font = font_brand
                                bcell.fill = fill_brand_row
                                bcell.alignment = align_left
                                bcell.border = cell_border
                                offset = 1
                            for ci, cname in enumerate(cols):
                                v = r.get(cname)
                                # NaN guard.
                                if (
                                    isinstance(v, float)
                                    and pd.isna(v)
                                ):
                                    v = None
                                cell = ws.cell(
                                    row=ri,
                                    column=ci + 1 + offset,
                                    value=v,
                                )
                                cell.border = cell_border
                                if cname in num_fmts:
                                    cell.number_format = (
                                        num_fmts[cname]
                                    )
                                if cname in (
                                    "SKU", "Size"
                                ):
                                    cell.alignment = align_left
                                elif cname == "Rank":
                                    cell.alignment = align_center
                                else:
                                    cell.alignment = align_right
                        # Column widths.
                        for ci, cname in enumerate(
                            header_cols, start=1
                        ):
                            ws.column_dimensions[
                                get_column_letter(ci)
                            ].width = col_widths.get(cname, 12)
                        # Freeze header row.
                        ws.freeze_panes = "A2"

                    # ---- "All Brands" combined sheet ----
                    all_rows = []
                    for _bname, _draw in brand_tables_raw.items():
                        _draw2 = _draw.copy()
                        _draw2.insert(0, "Brand", _bname)
                        all_rows.append(_draw2)
                    if all_rows:
                        all_df = pd.concat(
                            all_rows, ignore_index=True
                        )
                    else:
                        all_df = pd.DataFrame(
                            columns=["Brand"] + cols
                        )

                    ws_all = wb.create_sheet("All Brands")
                    _write_table(
                        ws_all, all_df, extra_brand_col=True
                    )

                    # ---- One sheet per brand ----
                    _used_names = set()
                    for _bname, _draw in brand_tables_raw.items():
                        sname = _safe_sheet_name(_bname)
                        # Disambiguate duplicates (rare — only
                        # happens if two brand names collapse
                        # after sheet-name sanitization).
                        base = sname
                        _k = 2
                        while sname in _used_names:
                            sname = f"{base[:28]}_{_k}"
                            _k += 1
                        _used_names.add(sname)
                        ws_b = wb.create_sheet(sname)
                        _write_table(
                            ws_b, _draw, extra_brand_col=False
                        )

                    # ---- "Filters" documentation sheet ----
                    ws_f = wb.create_sheet("Filters")
                    ws_f["A1"] = "Brand-wise SKU Recommendations"
                    ws_f["A1"].font = Font(bold=True, size=14)
                    ws_f["A2"] = (
                        "Ranked by Rank Score "
                        "(= Throughput × Latest MRP × GM index)"
                    )
                    ws_f["A2"].font = Font(italic=True, size=10)

                    _filter_rows = [
                        ("Top N per brand", rec_top_n),
                        ("Quarter scope", rec_quarter_preset),
                        ("Minimum total pieces", rec_min_vol),
                        (
                            "Minimum penetration %",
                            rec_min_pen,
                        ),
                        (
                            "Size scope",
                            ", ".join(rec_size_scope),
                        ),
                        (
                            "Months in scope",
                            rec_n_months,
                        ),
                        (
                            "Months included",
                            ", ".join(rec_months_pretty),
                        ),
                        ("Months in tab slice", n_months),
                        (
                            "All months in tab slice",
                            ", ".join(
                                m.replace("_V", "")
                                for m in months_in_data
                            ),
                        ),
                        (
                            "Brands included",
                            ", ".join(brands_present)
                            if brands_present else "—",
                        ),
                        (
                            "Total SKUs in scope (with score)",
                            len(rec_df),
                        ),
                        (
                            "SKUs after size filter",
                            len(rec_df_sized),
                        ),
                        (
                            "SKUs after pieces + pen filters",
                            len(rec_df_filtered),
                        ),
                    ]
                    for ri, (k, v) in enumerate(
                        _filter_rows, start=4
                    ):
                        ws_f.cell(
                            row=ri, column=1, value=k
                        ).font = Font(bold=True)
                        ws_f.cell(row=ri, column=2, value=v)
                    ws_f.column_dimensions["A"].width = 36
                    ws_f.column_dimensions["B"].width = 60

                    # Move "Filters" to be the first sheet for
                    # at-a-glance context.
                    wb.move_sheet("Filters", offset=-len(wb.sheetnames) + 1)

                    bio = io.BytesIO()
                    wb.save(bio)
                    return bio.getvalue()

                if per_brand_raw:
                    try:
                        _rec_xlsx_bytes = _build_brand_rec_xlsx(
                            per_brand_raw
                        )
                        st.download_button(
                            "📊 Download brand-wise "
                            "recommendations (.xlsx)",
                            data=_rec_xlsx_bytes,
                            file_name=(
                                "sku_brand_recommendations.xlsx"
                            ),
                            mime=(
                                "application/"
                                "vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet"
                            ),
                            key="spl_rec_xlsx",
                            use_container_width=False,
                        )
                    except Exception as _rec_err:
                        st.error(
                            f"Couldn't build the brand-wise "
                            f"Excel: {_rec_err}. If openpyxl "
                            "isn't installed in your "
                            "environment, `pip install "
                            "openpyxl` and reload."
                        )

        # =====================================================
        # (vi) SMALL SKUs — separate ranked list
        # =====================================================
        # Mondelez sells a long tail of small-size SKUs (singles
        # / impulse packs). They're excluded from the brand-wise
        # assortment recommendations above because the
        # planogram-fit rule reserves Large + Medium slots for
        # the headline SKUs. But the small SKUs still matter for
        # impulse / checkout placement and need their own ranked
        # list. This section reuses the same quarter scope,
        # ranking formula (TP × MRP × GM index), and pieces +
        # penetration filters from above, but restricts the
        # universe to `size == "Small"` and ranks across all
        # brands so the user can see the strongest impulse SKUs
        # at a glance regardless of brand.
        st.divider()
        st.subheader(
            "🍫 Small SKUs — ranked list (TP × MRP × GM)"
        )
        st.caption(
            "Separate ranking for **Small** SKUs (excluded "
            "from the brand-wise assortment above because they "
            "don't compete for Large/Medium cooler slots). "
            "Uses the same quarter scope, ranking formula, and "
            "pieces + penetration filters as the brand-wise "
            "list. Ranked across all brands."
        )

        # ---- Controls just for the Small list ----
        # Independent Top-N so the user can show a longer tail
        # for small SKUs (where the long-tail story matters more)
        # without lengthening the brand-wise tables above.
        sm_c1, sm_c2 = st.columns([1, 3])
        with sm_c1:
            sm_top_n = st.slider(
                "Top N Small SKUs",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                key="spl_rec_small_top_n",
                help=(
                    "How many Small SKUs to show in the ranked "
                    "list below (after applying the same "
                    "pieces + penetration filters as the brand-"
                    "wise section above)."
                ),
            )
        with sm_c2:
            sm_group_by_brand = st.checkbox(
                "Group by brand (one expandable section per brand)",
                value=False,
                key="spl_rec_small_group_by_brand",
                help=(
                    "Off (default) = one flat table ranked across "
                    "all brands. On = one expandable per brand, "
                    "each with that brand's top Small SKUs."
                ),
            )

        # Build the Small-SKU table from rec_df (already has
        # throughput, latest_mrp, gm_index, total_vol,
        # total_value, penetration_pct, gm_value, rank_score,
        # brand, size — computed over the chosen quarter scope).
        small_df = rec_df[rec_df["size"] == "Small"].copy()

        # Apply the same pieces + penetration filters as the
        # brand-wise block above (so the two lists are
        # consistent on what counts as "eligible").
        small_df_filtered = small_df[
            (small_df["total_vol"] >= rec_min_vol)
            & (small_df["penetration_pct"] >= rec_min_pen)
            & (small_df["rank_score"] > 0)
        ].copy()

        # Diagnostic counts for transparency.
        sm_d1, sm_d2, sm_d3 = st.columns(3)
        sm_d1.metric(
            "Small SKUs in scope",
            f"{len(small_df):,}",
            help=(
                "Small SKUs with a positive TP × MRP × GM "
                f"score in the {rec_n_months}-month scope."
            ),
        )
        sm_d2.metric(
            "After pieces + pen filters",
            f"{len(small_df_filtered):,}",
            help=(
                f"Small SKUs with ≥ {rec_min_vol:,} pieces "
                f"and ≥ {rec_min_pen}% penetration."
            ),
        )
        sm_d3.metric(
            "Shown below",
            f"{min(len(small_df_filtered), sm_top_n):,}",
            help="Top N after ranking by TP × MRP × GM.",
        )

        if small_df_filtered.empty:
            st.warning(
                "No Small SKUs pass the pieces + penetration "
                "filters. Loosen one of them, widen the quarter "
                "scope, or check that the SKU-Size file "
                "classifies any SKUs as Small."
            )
            small_disp_raw = pd.DataFrame()
            small_per_brand_raw = {}
        else:
            # Rank across all brands by Rank Score.
            small_df_filtered = small_df_filtered.sort_values(
                "rank_score", ascending=False
            )

            # Build the display rows (same schema as the brand-
            # wise tables plus a Brand column up front, since
            # this list spans all brands).
            def _small_display_rows(_src_df):
                _rows = []
                for _rank, (_sku, _r) in enumerate(
                    _src_df.iterrows(), start=1
                ):
                    _rows.append({
                        "Rank": _rank,
                        "SKU": _sku,
                        "Brand": _r.get("brand", ""),
                        "Size": _r["size"],
                        "Penetration %": round(
                            float(_r["penetration_pct"]), 2
                        ),
                        "TP / month": round(
                            float(_r["throughput"]), 2
                        ),
                        "Latest MRP (₹)": (
                            round(float(_r["latest_mrp"]), 2)
                            if pd.notna(_r["latest_mrp"])
                            else np.nan
                        ),
                        "GM Index": (
                            round(float(_r["gm_index"]), 4)
                            if pd.notna(_r["gm_index"])
                            else np.nan
                        ),
                        "Total Pieces": int(round(
                            float(_r["total_vol"])
                        )),
                        "Total Value (₹)": float(
                            _r["total_value"]
                        ),
                        "GM Value (₹)": float(
                            _r["gm_value"]
                        ),
                        "Rank Score (TP × MRP × GM)": float(
                            _r["rank_score"]
                        ),
                    })
                return pd.DataFrame(_rows)

            # Flat list: top-N across all brands.
            small_disp_raw = _small_display_rows(
                small_df_filtered.head(sm_top_n)
            )

            # Pretty-format rupee columns for screen.
            _small_screen = small_disp_raw.copy()
            if not _small_screen.empty:
                _small_screen["Total Value (₹)"] = (
                    _small_screen["Total Value (₹)"]
                    .apply(_fmt_inr_compact)
                )
                _small_screen["GM Value (₹)"] = (
                    _small_screen["GM Value (₹)"]
                    .apply(_fmt_inr_compact)
                )

            # Per-brand raw tables — built either way so the
            # Excel export has the data even when the on-screen
            # mode is "flat".
            small_per_brand_raw = {}
            _small_brand_totals = (
                small_df_filtered.groupby("brand")["rank_score"]
                .sum()
                .sort_values(ascending=False)
            )
            for _bname in _small_brand_totals.index:
                if str(_bname).strip() == "":
                    continue
                _bdf_small = (
                    small_df_filtered[
                        small_df_filtered["brand"] == _bname
                    ]
                    .head(sm_top_n)
                )
                if _bdf_small.empty:
                    continue
                small_per_brand_raw[_bname] = (
                    _small_display_rows(_bdf_small)
                )

            # ---- Render ----
            if sm_group_by_brand:
                # Per-brand expanders (use sm_top_n as a cap
                # per brand). Mirror the brand-wise UX: first 3
                # expanded by default.
                for _idx, _bname in enumerate(
                    _small_brand_totals.index
                ):
                    if _bname not in small_per_brand_raw:
                        continue
                    _bdf_tbl = (
                        small_per_brand_raw[_bname].copy()
                    )
                    _bdf_tbl["Total Value (₹)"] = (
                        _bdf_tbl["Total Value (₹)"]
                        .apply(_fmt_inr_compact)
                    )
                    _bdf_tbl["GM Value (₹)"] = (
                        _bdf_tbl["GM Value (₹)"]
                        .apply(_fmt_inr_compact)
                    )
                    _n_in_brand_small = int(
                        (
                            small_df_filtered["brand"]
                            == _bname
                        ).sum()
                    )
                    _brand_score = float(
                        _small_brand_totals.get(_bname, 0.0)
                    )
                    with st.expander(
                        f"🏷️ {_bname} — top "
                        f"{len(_bdf_tbl)} of "
                        f"{_n_in_brand_small} eligible Small "
                        f"SKUs "
                        f"(brand Rank Score: "
                        f"{_brand_score:,.0f})",
                        expanded=(_idx < 3),
                    ):
                        st.dataframe(
                            _bdf_tbl,
                            use_container_width=True,
                            hide_index=True,
                            height=min(
                                480, 45 + 35 * len(_bdf_tbl)
                            ),
                        )
            else:
                # Flat table: top-N across all brands.
                st.dataframe(
                    _small_screen,
                    use_container_width=True,
                    hide_index=True,
                    height=min(
                        640, 45 + 35 * len(_small_screen)
                    ),
                )

        # ---- Excel export for the Small list ----
        # Always offer the download as long as at least one
        # Small SKU survived the filters, so the user can pull
        # the full universe (not just the top-N shown above).
        def _build_small_rec_xlsx(flat_raw, brand_raw):
            """
            Multi-sheet Excel for the Small SKUs list:
              • 'All Small SKUs (ranked)' — every Small SKU
                surviving the filters, ranked by Rank Score
                across all brands. Not capped at top-N so the
                user has the full picture in Excel.
              • One sheet per brand with that brand's eligible
                Small SKUs (also not capped).
              • 'Filters' sheet documenting the scope.
            """
            from openpyxl import Workbook
            from openpyxl.styles import (
                Font, PatternFill, Alignment, Border, Side
            )
            from openpyxl.utils import get_column_letter

            wb = Workbook()
            wb.remove(wb.active)

            thin = Side(
                border_style="thin", color="BFBFBF"
            )
            cell_border = Border(
                left=thin, right=thin,
                top=thin, bottom=thin,
            )
            fill_header = PatternFill(
                "solid",
                start_color="2E7D32",  # green for Small list
                end_color="2E7D32",
            )
            font_header = Font(
                bold=True, color="FFFFFF", size=11
            )
            align_center = Alignment(
                horizontal="center", vertical="center"
            )
            align_right = Alignment(
                horizontal="right", vertical="center"
            )
            align_left = Alignment(
                horizontal="left", vertical="center"
            )

            cols = [
                "Rank", "SKU", "Brand", "Size",
                "Penetration %", "TP / month",
                "Latest MRP (₹)", "GM Index",
                "Total Pieces", "Total Value (₹)",
                "GM Value (₹)",
                "Rank Score (TP × MRP × GM)",
            ]
            num_fmts = {
                "Rank": "0",
                "Penetration %": "0.00",
                "TP / month": "0.00",
                "Latest MRP (₹)": '"₹"#,##0.00',
                "GM Index": "0.0000",
                "Total Pieces": "#,##0",
                "Total Value (₹)": (
                    '"₹"#,##0;[Red]-"₹"#,##0;-'
                ),
                "GM Value (₹)": (
                    '"₹"#,##0;[Red]-"₹"#,##0;-'
                ),
                "Rank Score (TP × MRP × GM)": "#,##0.00",
            }
            col_widths = {
                "Rank": 6, "SKU": 36, "Brand": 18,
                "Size": 8, "Penetration %": 13,
                "TP / month": 12, "Latest MRP (₹)": 14,
                "GM Index": 10, "Total Pieces": 13,
                "Total Value (₹)": 16,
                "GM Value (₹)": 16,
                "Rank Score (TP × MRP × GM)": 22,
            }

            def _safe_sheet_name(name):
                bad = set('[]:*?/\\')
                cleaned = "".join(
                    c for c in str(name) if c not in bad
                ).strip()
                if not cleaned:
                    cleaned = "Brand"
                return cleaned[:31]

            def _write(ws, df_raw):
                for ci, cname in enumerate(cols, start=1):
                    cell = ws.cell(
                        row=1, column=ci, value=cname
                    )
                    cell.fill = fill_header
                    cell.font = font_header
                    cell.alignment = align_center
                    cell.border = cell_border
                for ri, (_, r) in enumerate(
                    df_raw.iterrows(), start=2
                ):
                    for ci, cname in enumerate(cols, start=1):
                        v = r.get(cname)
                        if (
                            isinstance(v, float)
                            and pd.isna(v)
                        ):
                            v = None
                        cell = ws.cell(
                            row=ri, column=ci, value=v
                        )
                        cell.border = cell_border
                        if cname in num_fmts:
                            cell.number_format = (
                                num_fmts[cname]
                            )
                        if cname in ("SKU", "Brand", "Size"):
                            cell.alignment = align_left
                        elif cname == "Rank":
                            cell.alignment = align_center
                        else:
                            cell.alignment = align_right
                for ci, cname in enumerate(cols, start=1):
                    ws.column_dimensions[
                        get_column_letter(ci)
                    ].width = col_widths.get(cname, 12)
                ws.freeze_panes = "A2"

            # All Small SKUs sheet — full universe, not capped.
            # Re-rank (1..N) across the full filtered universe
            # so the rank numbers match what would show if the
            # user widened the slider all the way.
            if not flat_raw.empty:
                # `flat_raw` is already the top-N display. We
                # need the full filtered universe here, so
                # rebuild from `small_df_filtered`.
                _full_rows = []
                for _rank, (_sku, _r) in enumerate(
                    small_df_filtered.iterrows(), start=1
                ):
                    _full_rows.append({
                        "Rank": _rank,
                        "SKU": _sku,
                        "Brand": _r.get("brand", ""),
                        "Size": _r["size"],
                        "Penetration %": round(
                            float(_r["penetration_pct"]), 2
                        ),
                        "TP / month": round(
                            float(_r["throughput"]), 2
                        ),
                        "Latest MRP (₹)": (
                            round(float(_r["latest_mrp"]), 2)
                            if pd.notna(_r["latest_mrp"])
                            else np.nan
                        ),
                        "GM Index": (
                            round(
                                float(_r["gm_index"]), 4
                            )
                            if pd.notna(_r["gm_index"])
                            else np.nan
                        ),
                        "Total Pieces": int(round(
                            float(_r["total_vol"])
                        )),
                        "Total Value (₹)": float(
                            _r["total_value"]
                        ),
                        "GM Value (₹)": float(_r["gm_value"]),
                        "Rank Score (TP × MRP × GM)": float(
                            _r["rank_score"]
                        ),
                    })
                _all_df = pd.DataFrame(_full_rows)
            else:
                _all_df = pd.DataFrame(columns=cols)

            ws_all = wb.create_sheet(
                "All Small SKUs (ranked)"
            )
            _write(ws_all, _all_df)

            # Per-brand sheets.
            _used = set()
            for _bname, _draw in brand_raw.items():
                sname = _safe_sheet_name(_bname)
                base = sname
                _k = 2
                while sname in _used:
                    sname = f"{base[:28]}_{_k}"
                    _k += 1
                _used.add(sname)
                ws_b = wb.create_sheet(sname)
                _write(ws_b, _draw)

            # Filters sheet.
            ws_f = wb.create_sheet("Filters")
            ws_f["A1"] = "Small SKUs — ranked list"
            ws_f["A1"].font = Font(bold=True, size=14)
            ws_f["A2"] = (
                "Ranked by Rank Score "
                "(= Throughput × Latest MRP × GM index)"
            )
            ws_f["A2"].font = Font(italic=True, size=10)
            _frows = [
                ("Top N (on-screen cap only)", sm_top_n),
                (
                    "On-screen mode",
                    (
                        "Grouped by brand"
                        if sm_group_by_brand
                        else "Flat list across all brands"
                    ),
                ),
                ("Quarter scope", rec_quarter_preset),
                ("Minimum total pieces", rec_min_vol),
                ("Minimum penetration %", rec_min_pen),
                (
                    "Months in scope",
                    rec_n_months,
                ),
                (
                    "Months included",
                    ", ".join(rec_months_pretty),
                ),
                ("Months in tab slice", n_months),
                (
                    "Small SKUs in scope (with score)",
                    len(small_df),
                ),
                (
                    "Small SKUs after pieces + pen filters",
                    len(small_df_filtered),
                ),
            ]
            for ri, (k, v) in enumerate(_frows, start=4):
                ws_f.cell(
                    row=ri, column=1, value=k
                ).font = Font(bold=True)
                ws_f.cell(row=ri, column=2, value=v)
            ws_f.column_dimensions["A"].width = 38
            ws_f.column_dimensions["B"].width = 60

            wb.move_sheet(
                "Filters",
                offset=-len(wb.sheetnames) + 1,
            )

            bio = io.BytesIO()
            wb.save(bio)
            return bio.getvalue()

        if not small_df_filtered.empty:
            try:
                _small_xlsx_bytes = _build_small_rec_xlsx(
                    small_disp_raw, small_per_brand_raw
                )
                st.download_button(
                    "📊 Download Small SKUs list (.xlsx)",
                    data=_small_xlsx_bytes,
                    file_name=(
                        "sku_small_recommendations.xlsx"
                    ),
                    mime=(
                        "application/"
                        "vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="spl_rec_small_xlsx",
                    use_container_width=False,
                )
            except Exception as _small_err:
                st.error(
                    f"Couldn't build the Small SKUs Excel: "
                    f"{_small_err}. If openpyxl isn't "
                    "installed in your environment, "
                    "`pip install openpyxl` and reload."
                )

    # =====================================================
    # (vii) TP BY CATEGORY & MONTH  —  reproduces the
    #       "TP_by_Category_and_Month.xlsx" report exactly
    # =====================================================
    # This block rebuilds the multi-sheet "TP by Category &
    # Month" workbook directly from the SKU Priority Lister's
    # working slice (`work`). Because `work` already has the
    # sidebar filters AND every tab-level segment-priority
    # filter applied (VC_CAT / VC_MODEL / PC Type / RE /
    # CHANNEL / Region / ASM / SE_TTY, plus the SKU-tier and
    # SKU-type ranking-scope filters), the report is fully
    # filter-aware: narrowing any of those filters narrows the
    # report's outlet/SKU universe and re-derives every TP cell.
    #
    # Throughput convention — matched to the source workbook:
    #   • All-Coolers monthly TP for SKU s, month m
    #         = pieces(s, m) / active_outlets(s, m)
    #     where active_outlets(s, m) = # distinct outlets in the
    #     filtered slice that sold SKU s in month m (pieces > 0).
    #     This is a PER-SKU, PER-MONTH active-outlet denominator
    #     (a newly-distributed SKU shows a high per-outlet TP),
    #     exactly like the uploaded report.
    #   • Per-segment (VC_CAT) monthly TP for SKU s, month m,
    #     band g = pieces(s, m, outlets with VC_CAT = g)
    #              / active_outlets(s, m)
    #     i.e. the numerator is partitioned by the outlet's
    #     VC_CAT band while the denominator stays the SKU-month
    #     active-outlet count — so the per-band values are
    #     ADDITIVE and sum back to the All-Coolers TP. (Verified
    #     against the source file: 16-20L + 30-35L + … +
    #     380-465L = "All Coolers" for every SKU-month.)
    #
    # SKUs are grouped into the report's categories (Bournville
    # Large/Small, CDM Large/Medium, Milkinis, Silk Large/Plain/
    # Small/Specials, Temptations) via a Brand + SKU-name
    # classifier that reproduces the workbook's "SKU Code key"
    # mapping; SKUs that don't match any known pattern are
    # grouped under an "Other" catch-all so nothing is silently
    # dropped.
    st.divider()
    st.subheader("📑 TP by Category & Month")
    st.caption(
        "Reproduces the **TP_by_Category_and_Month** report from "
        "this slice. Every filter above (sidebar + the tab-level "
        "segment-priority filters) applies, so the report reflects "
        "exactly the SKUs / outlets currently in scope. TP per SKU "
        "per month = pieces ÷ outlets that sold it that month; the "
        "per-cooler-size (VC_CAT) columns split that TP by the "
        "outlet's band and add back to **All Coolers**."
    )

    # ---- Report category model (mirrors the SKU Code key) ----
    _CAT_ORDER = [
        "Bournville Large", "Bournville Small",
        "CDM Large", "CDM Medium",
        "Milkinis",
        "Silk Large", "Silk Plain", "Silk Small", "Silk Specials",
        "Temptations",
        "Other",
    ]
    _CODE_BY_CAT = {
        "Bournville Large": "BL", "Bournville Small": "BS",
        "CDM Large": "CL",        "CDM Medium": "CM",
        "Milkinis": "Milkinis",
        "Silk Large": "SL",       "Silk Plain": "SM",
        "Silk Small": "SS",       "Silk Specials": "SSP",
        "Temptations": "T",       "Other": "—",
    }

    def _classify_report_category(sku_name, brand_value=""):
        """Map a SKU (using its name, with Brand as a fallback
        signal) to one of the report's categories. Pattern rules
        reproduce the uploaded workbook's SKU Code key exactly;
        anything unmatched falls to 'Other' so it still appears
        in the report rather than being dropped."""
        s = str(sku_name).strip().lower()
        b = str(brand_value).strip().lower()

        # --- Bournville (BVL …) ---
        if s.startswith("bvl") or "bournville" in s or b == "bournville":
            return (
                "Bournville Small" if "small" in s
                else "Bournville Large"
            )
        # --- Silk ---
        if s.startswith("silk") or b == "silk":
            if "plain" in s:
                return "Silk Plain"
            if any(k in s for k in (
                "valentine", "walnut", "plum cake", "special"
            )):
                return "Silk Specials"
            if "large" in s:
                return "Silk Large"
            # small / XS / remaining Silk → Silk Small
            return "Silk Small"
        # --- Temptations ---
        if s.startswith("temptation") or b == "temptations":
            return "Temptations"
        # --- Milkinis ---
        if "milkinis" in s or b == "milkinis":
            return "Milkinis"
        # --- CDM (Dairy Milk) ---
        if s.startswith("cdm") or "cdm" in b or "dairy milk" in b:
            # CDM Large = the large-format / high-MRP variants
            # (100g, FN Large, Crackle 75, RA Large, … anything
            # explicitly "large" or the 75/100 gram packs);
            # everything else (40g, FN Small, Crackle 40, RA
            # Small, …) is CDM Medium.
            if (
                "large" in s
                or "100" in s
                or "crackle 75" in s
                or " 75" in s
                or s.endswith("75")
            ):
                return "CDM Large"
            return "CDM Medium"
        return "Other"

    # ---- Months present in the working slice (chronological) ----
    _rep_months_v = list(months_in_data)            # e.g. Jan25_V …
    _rep_months_pretty = [
        m.replace("_V", "") for m in _rep_months_v  # e.g. Jan25 …
    ]

    # ---- Per-SKU per-month pieces (already filtered by `work`) ----
    _rep_pieces = (
        work.groupby(COL_SKU)[_rep_months_v].sum().astype(float)
    )

    # ---- Per-SKU per-month ACTIVE-OUTLET denominator ----
    # active_outlets(s, m) = # outlets that sold SKU s in month m.
    _rep_active = pd.DataFrame(
        0.0, index=_rep_pieces.index, columns=_rep_months_v
    )
    for _m in _rep_months_v:
        _sub = work[work[_m] > 0]
        if not _sub.empty:
            _cnt = _sub.groupby(COL_SKU)[COL_OUTLET].nunique()
            _rep_active[_m] = (
                _cnt.reindex(_rep_pieces.index).fillna(0).astype(float)
            )

    # ---- All-Coolers monthly TP = pieces / active_outlets ----
    _denom = _rep_active.replace(0.0, np.nan)
    _rep_tp = (_rep_pieces / _denom).fillna(0.0)
    _rep_tp.columns = _rep_months_pretty

    # ---- Attach Brand + report category to each SKU ----
    if COL_BRAND in work.columns:
        _sku_brand = (
            work.groupby(COL_SKU)[COL_BRAND]
            .agg(lambda s: s.dropna().astype(str).iloc[0]
                 if s.dropna().size else "")
            .reindex(_rep_tp.index)
            .fillna("")
        )
    else:
        _sku_brand = pd.Series("", index=_rep_tp.index)

    _sku_cat = pd.Series(
        [
            _classify_report_category(_sku, _sku_brand.get(_sku, ""))
            for _sku in _rep_tp.index
        ],
        index=_rep_tp.index,
        name="category",
    )
    _sku_code = _sku_cat.map(_CODE_BY_CAT).fillna("—")

    # ---- VC_CAT bands present (ordered) for the monthly sheets ----
    if COL_VC_CATEGORY in work.columns:
        _rep_bands = order_segment_values(
            COL_VC_CATEGORY,
            sorted(
                work[COL_VC_CATEGORY].dropna().astype(str).unique()
            ),
        )
    else:
        _rep_bands = []

    # =====================================================
    # (a) On-screen: Category-wise TP by month
    # =====================================================
    # Build a tidy table with category header rows, per-SKU
    # rows, "— Total" rows, and a GRAND TOTAL row — same shape
    # as the workbook's "Category wise TP" sheet.
    def _catwise_display_rows():
        rows = []
        grand = np.zeros(len(_rep_months_pretty))
        for _cat in _CAT_ORDER:
            _idx = _sku_cat[_sku_cat == _cat].index
            if len(_idx) == 0:
                continue
            # SKU rows (sorted by total TP across months, desc).
            _block = _rep_tp.loc[_idx]
            _block = _block.loc[
                _block.sum(axis=1).sort_values(ascending=False).index
            ]
            for _sku, _r in _block.iterrows():
                row = {
                    "Code": _sku_code.get(_sku, "—"),
                    "Category / SKU": _sku,
                }
                for _mi, _mp in enumerate(_rep_months_pretty):
                    row[_mp] = round(float(_r[_mp]), 2)
                rows.append(row)
            _ctot = _block.sum(axis=0).values
            grand = grand + _ctot
            trow = {
                "Code": f"{_cat} — Total",
                "Category / SKU": "",
            }
            for _mi, _mp in enumerate(_rep_months_pretty):
                trow[_mp] = round(float(_ctot[_mi]), 2)
            rows.append(trow)
        grow = {"Code": "GRAND TOTAL", "Category / SKU": ""}
        for _mi, _mp in enumerate(_rep_months_pretty):
            grow[_mp] = round(float(grand[_mi]), 2)
        rows.append(grow)
        return pd.DataFrame(rows)

    _catwise_tbl = _catwise_display_rows()
    st.markdown("**Category-wise TP (All Coolers) — by month**")
    st.dataframe(
        _catwise_tbl,
        use_container_width=True,
        hide_index=True,
        height=min(680, 60 + 28 * len(_catwise_tbl)),
    )

    # =====================================================
    # (b) On-screen: pick a month → TP by SKU & cooler size
    # =====================================================
    if _rep_bands:
        st.markdown("**TP by SKU & cooler size (VC_CAT) — one month**")
        _pick_month = st.selectbox(
            "Month",
            options=_rep_months_pretty,
            index=len(_rep_months_pretty) - 1,  # latest by default
            key="spl_catrep_month",
            help=(
                "Shows each SKU's TP split across the VC_CAT cooler-"
                "size bands present in the current slice. The band "
                "columns add up to the All-Coolers TP."
            ),
        )
        _pick_v = f"{_pick_month}_V"

        # Pieces split by VC_CAT for the picked month.
        _seg_pieces = work.pivot_table(
            index=COL_SKU,
            columns=COL_VC_CATEGORY,
            values=_pick_v,
            aggfunc="sum",
            fill_value=0.0,
        )
        _seg_pieces = _seg_pieces.reindex(_rep_tp.index).fillna(0.0)
        _seg_denom = _rep_active[_pick_v].replace(0.0, np.nan)
        _seg_tp = _seg_pieces.div(_seg_denom, axis=0).fillna(0.0)
        # Order/segment columns; ensure every band present.
        for _g in _rep_bands:
            if _g not in _seg_tp.columns:
                _seg_tp[_g] = 0.0
        _seg_tp = _seg_tp[_rep_bands]
        _seg_tp["All Coolers"] = _seg_tp.sum(axis=1)

        def _monthly_display_rows():
            rows = []
            grand = np.zeros(len(_rep_bands) + 1)
            for _cat in _CAT_ORDER:
                _idx = _sku_cat[_sku_cat == _cat].index
                if len(_idx) == 0:
                    continue
                _block = _seg_tp.loc[_idx]
                _block = _block.loc[
                    _block["All Coolers"]
                    .sort_values(ascending=False).index
                ]
                for _sku, _r in _block.iterrows():
                    row = {
                        "Code": _sku_code.get(_sku, "—"),
                        "Category / SKU": _sku,
                    }
                    for _g in _rep_bands:
                        row[_g] = round(float(_r[_g]), 2)
                    row["All Coolers"] = round(
                        float(_r["All Coolers"]), 2
                    )
                    rows.append(row)
                _ctot = _block[_rep_bands + ["All Coolers"]].sum(
                    axis=0
                ).values
                grand = grand + _ctot
                trow = {
                    "Code": f"{_cat} — Total",
                    "Category / SKU": "",
                }
                for _gi, _g in enumerate(_rep_bands):
                    trow[_g] = round(float(_ctot[_gi]), 2)
                trow["All Coolers"] = round(float(_ctot[-1]), 2)
                rows.append(trow)
            grow = {"Code": "GRAND TOTAL", "Category / SKU": ""}
            for _gi, _g in enumerate(_rep_bands):
                grow[_g] = round(float(grand[_gi]), 2)
            grow["All Coolers"] = round(float(grand[-1]), 2)
            rows.append(grow)
            return pd.DataFrame(rows)

        _monthly_tbl = _monthly_display_rows()
        st.dataframe(
            _monthly_tbl,
            use_container_width=True,
            hide_index=True,
            height=min(680, 60 + 28 * len(_monthly_tbl)),
        )
    else:
        st.info(
            "No VC_CAT column in the current slice, so the per-"
            "cooler-size monthly breakdown can't be built. The "
            "category-wise monthly TP table above and the Excel "
            "download still work (without the band split)."
        )

    # =====================================================
    # (c) Download — full multi-sheet workbook (.xlsx)
    # =====================================================
    # Rebuilds the exact sheet set of the uploaded report:
    #   • SKU Code key
    #   • Category wise TP   (SKU × month, All-Coolers TP)
    #   • one sheet per month (SKU × VC_CAT band + All Coolers)
    # all derived from the current filtered slice.
    def _build_tp_category_workbook():
        from openpyxl import Workbook
        from openpyxl.styles import (
            Font, PatternFill, Alignment, Border, Side
        )
        from openpyxl.utils import get_column_letter

        thin = Side(border_style="thin", color="D9D9D9")
        border = Border(
            left=thin, right=thin, top=thin, bottom=thin
        )
        hdr_fill = PatternFill(
            "solid", start_color="1F4E78", end_color="1F4E78"
        )
        hdr_font = Font(bold=True, color="FFFFFF", size=11)
        cat_font = Font(bold=True, color="1F4E78", size=11)
        tot_font = Font(bold=True, size=11)
        tot_fill = PatternFill(
            "solid", start_color="DDEBF7", end_color="DDEBF7"
        )
        grand_font = Font(bold=True, size=11, color="843C0C")
        grand_fill = PatternFill(
            "solid", start_color="FFE699", end_color="FFE699"
        )
        num_fmt = "0.00"

        wb = Workbook()
        wb.remove(wb.active)

        # ---------- Sheet 1: SKU Code key ----------
        ws_key = wb.create_sheet("SKU Code key")
        ws_key["A1"] = "Code Sheet"
        ws_key["A1"].font = Font(bold=True, size=12)
        _kr = 2
        for _cat in _CAT_ORDER:
            if _cat == "Other":
                # only emit Other if some SKU landed there
                if (_sku_cat == "Other").sum() == 0:
                    continue
            _c = ws_key.cell(
                row=_kr, column=1, value=_CODE_BY_CAT[_cat]
            )
            _c.font = Font(bold=True)
            ws_key.cell(row=_kr, column=2, value=_cat)
            _kr += 1
        ws_key.column_dimensions["A"].width = 12
        ws_key.column_dimensions["B"].width = 24

        # ---------- generic category-grouped writer ----------
        def _write_grouped(ws, value_df, value_cols, title,
                           subtitle, totals=True):
            """value_df: index = SKU, columns include value_cols.
            Writes a category-grouped sheet with Code + SKU
            columns then the value columns, with per-category
            '— Total' rows and a GRAND TOTAL."""
            ws["A1"] = title
            ws["A1"].font = Font(bold=True, size=14)
            ws["A2"] = subtitle
            ws["A2"].font = Font(italic=True, size=10)

            hdr_row = 4
            header = ["Code", "SKU"] + list(value_cols)
            for _ci, _h in enumerate(header, start=1):
                _c = ws.cell(row=hdr_row, column=_ci, value=_h)
                _c.font = hdr_font
                _c.fill = hdr_fill
                _c.alignment = Alignment(
                    horizontal="center", vertical="center"
                )
                _c.border = border
            ws.freeze_panes = ws.cell(row=hdr_row + 1, column=3)

            r = hdr_row + 1
            grand = np.zeros(len(value_cols))
            ncols = len(value_cols)
            for _cat in _CAT_ORDER:
                _idx = _sku_cat[_sku_cat == _cat].index
                _idx = [s for s in _idx if s in value_df.index]
                if len(_idx) == 0:
                    continue
                _block = value_df.loc[_idx, list(value_cols)]
                # sort within category by the LAST value column
                # (All Coolers for monthly sheets, latest month
                # for the category-wise sheet) descending.
                _sort_key = _block[value_cols[-1]]
                _block = _block.loc[
                    _sort_key.sort_values(ascending=False).index
                ]
                # Category header row.
                _hc = ws.cell(row=r, column=1, value=_cat)
                _hc.font = cat_font
                r += 1
                _ctot = np.zeros(ncols)
                for _sku, _row in _block.iterrows():
                    ws.cell(
                        row=r, column=1,
                        value=_sku_code.get(_sku, "—"),
                    )
                    ws.cell(row=r, column=2, value=str(_sku))
                    for _vi, _vc in enumerate(value_cols):
                        _v = float(_row[_vc])
                        _ctot[_vi] += _v
                        _cc = ws.cell(
                            row=r, column=3 + _vi,
                            value=round(_v, 2),
                        )
                        _cc.number_format = num_fmt
                        _cc.border = border
                    r += 1
                if totals:
                    _tc = ws.cell(
                        row=r, column=1,
                        value=f"{_cat} — Total",
                    )
                    _tc.font = tot_font
                    _tc.fill = tot_fill
                    ws.cell(row=r, column=2).fill = tot_fill
                    for _vi in range(ncols):
                        _cc = ws.cell(
                            row=r, column=3 + _vi,
                            value=round(float(_ctot[_vi]), 2),
                        )
                        _cc.font = tot_font
                        _cc.fill = tot_fill
                        _cc.number_format = num_fmt
                        _cc.border = border
                    r += 1
                grand = grand + _ctot
                r += 1  # blank separator row between categories
            # GRAND TOTAL.
            _gc = ws.cell(row=r, column=1, value="GRAND TOTAL")
            _gc.font = grand_font
            _gc.fill = grand_fill
            ws.cell(row=r, column=2).fill = grand_fill
            for _vi in range(ncols):
                _cc = ws.cell(
                    row=r, column=3 + _vi,
                    value=round(float(grand[_vi]), 2),
                )
                _cc.font = grand_font
                _cc.fill = grand_fill
                _cc.number_format = num_fmt
                _cc.border = border

            ws.column_dimensions["A"].width = 22
            ws.column_dimensions["B"].width = 26
            for _vi in range(ncols):
                ws.column_dimensions[
                    get_column_letter(3 + _vi)
                ].width = 11

        # ---------- Sheet 2: Category wise TP ----------
        ws_cat = wb.create_sheet("Category wise TP")
        _write_grouped(
            ws_cat,
            _rep_tp,
            _rep_months_pretty,
            "Category-wise Throughput (TP) — All Coolers, by Month",
            (
                "SKUs grouped under each category. TP = pieces ÷ "
                "outlets that sold the SKU that month, over the "
                "current filtered slice."
            ),
        )

        # ---------- Sheets 3..N: one per month ----------
        if _rep_bands:
            for _mp, _mv in zip(_rep_months_pretty, _rep_months_v):
                _seg_pc = work.pivot_table(
                    index=COL_SKU,
                    columns=COL_VC_CATEGORY,
                    values=_mv,
                    aggfunc="sum",
                    fill_value=0.0,
                )
                _seg_pc = _seg_pc.reindex(_rep_tp.index).fillna(0.0)
                _dn = _rep_active[_mv].replace(0.0, np.nan)
                _seg = _seg_pc.div(_dn, axis=0).fillna(0.0)
                for _g in _rep_bands:
                    if _g not in _seg.columns:
                        _seg[_g] = 0.0
                _seg = _seg[_rep_bands]
                _seg["All Coolers"] = _seg.sum(axis=1)

                _ws_m = wb.create_sheet(_mp[:31])
                _write_grouped(
                    _ws_m,
                    _seg,
                    list(_rep_bands) + ["All Coolers"],
                    f"Throughput (TP) by SKU & Cooler Size — {_mp}",
                    (
                        'Each SKU listed under its category. '
                        '"All Coolers" = sum of the cooler-size '
                        "bands present in the current slice."
                    ),
                )

        _bio = io.BytesIO()
        wb.save(_bio)
        return _bio.getvalue()

    try:
        _tp_cat_bytes = _build_tp_category_workbook()
        st.download_button(
            "📥 Download TP by Category & Month (.xlsx)",
            data=_tp_cat_bytes,
            file_name="TP_by_Category_and_Month.xlsx",
            mime=(
                "application/"
                "vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="spl_tp_category_month_xlsx",
            use_container_width=False,
            help=(
                "Multi-sheet workbook: SKU Code key, Category-wise "
                "TP by month, and one TP-by-cooler-size sheet per "
                "month — all from the current filtered slice."
            ),
        )
    except Exception as _tpcat_err:
        st.error(
            f"Couldn't build the TP by Category & Month workbook: "
            f"{_tpcat_err}. If openpyxl isn't installed, "
            "`pip install openpyxl` and reload."
        )


def render_vc_builder():
    """
    VC Planogram Builder — interactive tool to design two
    visicooler planograms side-by-side and compare their total
    throughput / volume / value / GM-value.

    User flow:
      1. Pick the number of Large / Medium / Small slots in each
         of two coolers (independent — they don't have to match).
      2. For each slot, pick a SKU from the dropdown. The
         dropdown only shows SKUs whose size is compatible with
         that slot type (size-fit rule below). The same SKU can
         be placed in multiple slots; its throughput scales by a
         multi-facing multiplier (1× / 1.5× / 1.6× / 1.6× ...).
      3. Live summary panel under each cooler shows Total TP,
         Total Volume, Total Value (TP × MRP), and GM Value
         (TP × MRP × GM), plus a clear per-SKU details table
         with a totals row.
      4. A Suggestions panel ranks the highest-uplift moves
         (add a facing / add new SKU / replace SKU) against
         the chosen objective (Sales / Profit / Transactions
         / All).
      5. An Excel export builds a multi-sheet workbook with
         the summary, both cooler layouts, per-SKU details,
         and the current suggestions.

    Size-fit rule (per user spec):
      • Large SKU  → Large slot only
      • Medium SKU → Large or Medium slot
      • Small SKU  → Medium or Small slot

    Facings rule (per user spec):
      • Cooler A — **count once**: a SKU appearing in any
        number of slots contributes 1× its base value.
        Double / triple facing in A does not add anything.
      • Cooler B — **multi-facing multiplier**:
          facings : 1   2    3+   (saturated)
          mult    : 1.0 1.3  1.5
    The multiplier applies to throughput (and to volume / value /
    gm-value / transactions, since those are derived from
    throughput). A SKU appearing N times in Cooler B contributes
        N_effective = base_tp × _facing_mult(N)
    while in Cooler A it contributes 1 × base_tp regardless
    of how many slots it fills.

    Data source:
      Uses `filtered_df` (sidebar filters applied) and then
      narrows further by tab-level filters (VC_CAT / CHANNEL /
      RE / months). Per-SKU throughput = total volume in the
      narrowed slice ÷ active outlets ÷ number of months
      selected. MRP = `sku_latest_mrp`, GM = `_gm_lookup`. SKUs
      with zero volume in the narrowed slice are dropped from
      the dropdowns.

    Gating: only shown if the SKU-Size file is uploaded
    (`_sku_size_available` flag).
    """
    st.title("🧊 VC Planogram Builder")
    st.caption(
        "Design two visicooler planograms side-by-side and "
        "compare their total throughput, pieces, value, and "
        "GM value. Sidebar filters apply first, then the "
        "cooler-data filters below narrow further before "
        "per-SKU TP / MRP / GM are computed."
    )

    if df is None or df.empty:
        st.info("No data available. Please upload a source file.")
        return

    if not _sku_size_available or _sku_size_lookup is None:
        st.warning(
            "Upload a SKU-Size file in the sidebar to use this "
            "tab (two columns: SKU name, Size = Large / Medium "
            "/ Small)."
        )
        return

    # ---- Build per-SKU metrics from the filtered slice ----
    # We compute throughput on L15M volume (full 15-month window
    # in the slice) by default, but the user can narrow further
    # via the tab-level filters below. The tab filters STACK on
    # top of the sidebar filters: sidebar narrows `filtered_df`
    # first, then these widgets narrow further. Empty selection
    # on any filter = no further narrowing on that dimension.
    if filtered_df is None or filtered_df.empty:
        st.info(
            "Current sidebar filters return no rows. Relax the "
            "filters to use the planogram builder."
        )
        return

    _months_avail_all = [
        m for m in MONTH_COLS if m in filtered_df.columns
    ]
    if not _months_avail_all:
        st.error(
            "Source file has no monthly pieces columns "
            "(Jan25_V … Mar26_V). Cannot compute throughput."
        )
        return

    # ---- Tab-level filters (stack on sidebar filters) ----
    st.subheader("🎛️ Cooler data filters")
    st.caption(
        "Narrow the data slice used to compute per-SKU TP / "
        "Pieces. These stack on top of the sidebar filters — "
        "leave any empty to skip further narrowing on that "
        "dimension."
    )

    # Build option lists from the currently-filtered slice so we
    # never offer a value that has zero rows after sidebar
    # filtering.
    _vc_opts_raw = (
        filtered_df[COL_VC_CATEGORY].dropna().astype(str).unique()
        if COL_VC_CATEGORY in filtered_df.columns else []
    )
    _vc_opts = order_segment_values(
        COL_VC_CATEGORY, sorted(_vc_opts_raw)
    )
    _channel_opts = sorted(
        filtered_df[COL_CHANNEL].dropna().astype(str).unique()
        if COL_CHANNEL in filtered_df.columns else []
    )
    _re_opts = sorted(
        filtered_df[COL_RE].dropna().astype(str).unique()
        if COL_RE in filtered_df.columns else []
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        _vc_sel = st.multiselect(
            "VC_CAT",
            options=_vc_opts,
            default=[],
            key="vcb_filter_vc_cat",
            help=(
                "Volume-capacity band. Empty = all bands in the "
                "current sidebar slice."
            ),
        )
    with fc2:
        _ch_sel = st.multiselect(
            "CHANNEL",
            options=_channel_opts,
            default=[],
            key="vcb_filter_channel",
            help="Empty = all channels.",
        )
    with fc3:
        _re_sel = st.multiselect(
            "RE",
            options=_re_opts,
            default=[],
            key="vcb_filter_re",
            help="Empty = all REs.",
        )

    # Month selector — pretty labels (Jan25, Feb25, ...).
    _pretty_to_raw_vcb = {
        m.replace("_V", ""): m for m in _months_avail_all
    }
    _pretty_months_vcb = list(_pretty_to_raw_vcb.keys())
    _months_pretty_sel = st.multiselect(
        "Months (used to compute Base TP)",
        options=_pretty_months_vcb,
        default=[],
        key="vcb_filter_months",
        help=(
            "Empty = use all available months. Pick a subset "
            "(e.g. festive months) to base the planogram TP on "
            "that seasonal window. Throughput = total pieces in "
            "the selected months ÷ active outlets ÷ # months "
            "selected."
        ),
    )

    # ---- Apply tab filters on top of filtered_df ----
    _vcb_slice = filtered_df
    if _vc_sel:
        _vcb_slice = _vcb_slice[
            _vcb_slice[COL_VC_CATEGORY].astype(str).isin(_vc_sel)
        ]
    if _ch_sel:
        _vcb_slice = _vcb_slice[
            _vcb_slice[COL_CHANNEL].astype(str).isin(_ch_sel)
        ]
    if _re_sel:
        _vcb_slice = _vcb_slice[
            _vcb_slice[COL_RE].astype(str).isin(_re_sel)
        ]

    if _vcb_slice.empty:
        st.warning(
            "These tab filters return no rows. Loosen them "
            "(or clear them) to continue."
        )
        return

    # Resolve month selection → raw column list.
    if _months_pretty_sel:
        _months_avail = [
            _pretty_to_raw_vcb[p] for p in _months_pretty_sel
            if p in _pretty_to_raw_vcb
        ]
    else:
        _months_avail = list(_months_avail_all)
    if not _months_avail:
        st.warning("No months selected.")
        return
    _n_months = len(_months_avail)

    # Small data-slice header so the user can see what's
    # actually flowing into the per-SKU TP calc.
    _slice_outlets = int(_vcb_slice[COL_OUTLET].nunique())
    _slice_rows = len(_vcb_slice)
    st.caption(
        f"Slice: **{_slice_outlets:,}** outlets, "
        f"**{_slice_rows:,}** outlet×SKU rows, "
        f"**{_n_months}** month(s) "
        f"({', '.join(m.replace('_V', '') for m in _months_avail)})."
    )

    # Per-SKU total volume in the tab-narrowed slice & month
    # window.
    _sku_vol = (
        _vcb_slice.groupby(COL_SKU)[_months_avail].sum().sum(axis=1)
    )
    # Per-SKU active outlets (outlets selling the SKU in any of
    # the selected months).
    _sku_outlets = (
        _vcb_slice[_vcb_slice[_months_avail].sum(axis=1) > 0]
        .groupby(COL_SKU)[COL_OUTLET].nunique()
    )
    # Per-SKU brand (for display in dropdowns).
    _sku_brand = (
        _vcb_slice.groupby(COL_SKU)[COL_BRAND].first()
    )
    # Avg monthly TP per SKU.
    _sku_tp = (
        _sku_vol
        / _sku_outlets.replace(0, np.nan)
        / max(_n_months, 1)
    ).fillna(0.0)

    # Per-SKU "transactions" in the slice — defined as the total
    # number of units sold (i.e. the same as Volume / TP × outlets
    # × months). Previously this was a coverage count (number of
    # outlet × month cells with volume > 0); switched to a unit
    # count so "transactions" represents how many units of the SKU
    # left the shelves across the slice. This is a per-SKU scalar
    # in the current slice; cooler-level totals scale it by
    # facings (linear for Cooler A, multi-facing curve for Cooler
    # B — see _cooler_totals below).
    _sku_txn = _sku_vol.astype(float).fillna(0.0)

    # Latest MRP per SKU (rightmost non-null monthly MRP).
    _sku_mrp = pd.Series(0.0, index=_sku_vol.index)
    if _mrp_available and _mrp_lookup is not None:
        _sku_keys_vc = (
            _sku_vol.index.astype(str).str.strip().str.lower()
        )
        _mrp_cols_present = [
            c for c in MRP_MONTH_COLS if c in _mrp_lookup.columns
        ]
        if _mrp_cols_present:
            _mrp_aligned = (
                _mrp_lookup[_mrp_cols_present]
                .reindex(_sku_keys_vc.values)
            )
            _mrp_aligned.index = _sku_vol.index
            # Rightmost non-null = latest available MRP.
            _sku_mrp = (
                _mrp_aligned.ffill(axis=1).iloc[:, -1].fillna(0.0)
            )

    # GM index per SKU.
    _sku_gm = pd.Series(0.0, index=_sku_vol.index)
    if _gm_available and _gm_lookup is not None:
        _sku_keys_vc2 = (
            _sku_vol.index.astype(str).str.strip().str.lower()
        )
        _sku_gm = (
            _gm_lookup.reindex(_sku_keys_vc2.values).fillna(0.0)
        )
        _sku_gm.index = _sku_vol.index

    # Size per SKU (from uploaded SKU-Size file, joined on lower-
    # cased name). SKUs not in the size file → "Unknown" and are
    # not offered in any slot.
    _sku_keys_vc3 = (
        _sku_vol.index.astype(str).str.strip().str.lower()
    )
    _sku_size = (
        _sku_size_lookup.reindex(_sku_keys_vc3.values)
    )
    _sku_size.index = _sku_vol.index
    _sku_size = _sku_size.fillna("Unknown")

    # Per-slot-type eligibility lists. Sort by TP descending so
    # the most useful SKUs are at the top of each dropdown.
    # Per user spec (SKU → which slot sizes it can go into):
    #   Large SKU  → Large slot only
    #   Medium SKU → Large or Medium slot
    #   Small SKU  → Medium or Small slot
    # Inverted to slot → eligible SKU sizes for the dropdown
    # filter below:
    #   Large slot  accepts Large + Medium SKUs
    #   Medium slot accepts Medium + Small SKUs
    #   Small slot  accepts Small SKUs only
    _slot_eligible = {
        "Large":  ["Large", "Medium"],
        "Medium": ["Medium", "Small"],
        "Small":  ["Small"],
    }

    def _sku_options_for_slot(slot_type):
        """Return a list of SKU names eligible for this slot
        type, sorted by descending TP. Format as
        '<sku> — TP <x>, MRP <y>, GM <z>' for the dropdown
        label, but return the bare SKU name as the value."""
        eligible_sizes = _slot_eligible.get(slot_type, [])
        mask = _sku_size.isin(eligible_sizes)
        skus = _sku_vol.index[mask]
        # Sort by avg monthly TP, descending.
        tp_for_sort = _sku_tp.reindex(skus).fillna(0.0)
        ordered = tp_for_sort.sort_values(ascending=False).index
        return list(ordered)

    def _facing_mult(n, custom_double=None):
        """Multi-facing TP multiplier (Cooler B / "second list"
        only). Per user spec:
            1 facing  → 1.0×
            2 facings → 1.3× (default — overridable per SKU
                              via `custom_double`)
            3+ facings → 1.5×  (saturates)
        Cooler A does NOT use this — see _cooler_totals.

        `custom_double` (optional float): if provided and n == 2,
        the 2-facing multiplier is replaced by this value. Lets
        the user set a per-SKU double-facing multiplier instead
        of the hard-coded 1.3.
        """
        if n <= 0:
            return 0.0
        if n == 1:
            return 1.0
        if n == 2:
            if custom_double is not None:
                try:
                    return float(custom_double)
                except (TypeError, ValueError):
                    return 1.3
            return 1.3
        return 1.5  # 3+ saturates at 1.5×

    def _sku_label(sku):
        """Friendly dropdown label for a SKU.

        Per user spec: show only the bare SKU name in the
        dropdown — no TP / MRP / GM / size / brand tags.
        """
        if sku is None or sku == "":
            return "— empty —"
        return str(sku)

    # ---- Skeleton picker UI ----
    st.subheader("🧱 Cooler skeletons")
    st.caption(
        "Pick how many Large / Medium / Small slots each cooler "
        "has. Then fill the slots below. Size-fit rule: Large "
        "SKU → Large slot only; Medium SKU → Large or Medium; "
        "Small (countline) SKU → Medium or Small."
    )

    sk1, sk2 = st.columns(2)
    with sk1:
        st.markdown("**Cooler A**")
        a_large = st.number_input(
            "Large slots (A)", min_value=0, max_value=40,
            value=6, step=1, key="vcb_a_large",
        )
        a_medium = st.number_input(
            "Medium slots (A)", min_value=0, max_value=40,
            value=10, step=1, key="vcb_a_medium",
        )
        a_small = st.number_input(
            "Small slots (A)", min_value=0, max_value=40,
            value=10, step=1, key="vcb_a_small",
        )
    with sk2:
        st.markdown("**Cooler B**")
        b_large = st.number_input(
            "Large slots (B)", min_value=0, max_value=40,
            value=6, step=1, key="vcb_b_large",
        )
        b_medium = st.number_input(
            "Medium slots (B)", min_value=0, max_value=40,
            value=10, step=1, key="vcb_b_medium",
        )
        b_small = st.number_input(
            "Small slots (B)", min_value=0, max_value=40,
            value=10, step=1, key="vcb_b_small",
        )

    # ---- Cooler A preset: pre-built planogram ----
    # One-click loader that wipes Cooler A and drops in a fixed
    # 29-slot layout (7 Large + 10 Medium + 12 Small, with one
    # empty Medium slot). Names below are matched against the
    # SKUs present in the current slice (case- and whitespace-
    # insensitive, with a substring fallback so e.g. "CDM 40"
    # finds "CDM 40 60g" if that's how the data has it).
    #
    # Slots are numbered top-to-bottom on each shelf, matching
    # how `_render_cooler` lays them out (Large first, then
    # Medium, then Small).
    _COOLER_A_PRESET = [
        # (shelf_size, slot_ordinal, sku_name_or_None_for_empty)
        ("Large",  1,  "Silk FN Large"),
        ("Large",  2,  "Silk Oreo Large"),
        ("Large",  3,  "Silk Mousse Large"),
        ("Large",  4,  "CDM 100"),
        ("Large",  5,  "CDM 100"),
        ("Large",  6,  "CDM FN Large"),
        ("Large",  7,  "BVL RC Small"),
        ("Medium", 1,  "Silk Plain 100"),
        ("Medium", 2,  "Silk Mousse Small"),
        ("Medium", 3,  "Silk Hazelnut Small"),
        ("Medium", 4,  "Silk RA Small"),
        ("Medium", 5,  "Silk Oreo Small"),
        ("Medium", 6,  "CDM 40"),
        ("Medium", 7,  "CDM 10"),
        ("Medium", 8,  "Silk XS"),
        ("Medium", 9,  "Silk XS"),
        ("Medium", 10, None),
        ("Small",  1,  "CDM Crackle 40"),
        ("Small",  2,  "CDM RA Small"),
        ("Small",  3,  "Crispello 10"),
        ("Small",  4,  "Crispello 30"),
        ("Small",  5,  "FS 10"),
        ("Small",  6,  "FS 20"),
        ("Small",  7,  "Fuse 20"),
        ("Small",  8,  "Fuse 35"),
        ("Small",  9,  "FS 3D Small"),
        ("Small",  10, "FS Oreo Small"),
        ("Small",  11, "FS Oreo Large"),
        ("Small",  12, "FS 3D Large"),
    ]

    def _resolve_preset_sku(name, slice_skus):
        """Match a preset SKU name against the SKUs in the
        current slice. Tries exact (case-insensitive, whitespace-
        normalised) first, then a substring containment fallback.
        Returns the matched SKU exactly as it appears in the data,
        or None if no match is found."""
        if not name:
            return None
        _norm = lambda s: " ".join(str(s).strip().lower().split())
        target = _norm(name)
        # Build normalised → original map once.
        candidates = {_norm(s): s for s in slice_skus}
        # 1. Exact normalised match.
        if target in candidates:
            return candidates[target]
        # 2. Substring fallback — preset name appears inside a
        #    data SKU name (e.g. preset "CDM 40" → data "CDM 40 60g").
        contains = [
            orig for norm_s, orig in candidates.items()
            if target in norm_s
        ]
        if len(contains) == 1:
            return contains[0]
        if len(contains) > 1:
            # Multiple matches — pick the shortest name (least
            # likely to be a superstring with extra descriptors).
            return min(contains, key=lambda s: len(s))
        return None

    def _apply_cooler_a_preset():
        """Button callback: wipe Cooler A's current arrangement
        and write the preset into session_state so the classic
        dropdowns pick it up on the next rerun."""
        # 1. Force the slot counts to fit the preset.
        st.session_state["vcb_a_large"] = 7
        st.session_state["vcb_a_medium"] = 10
        st.session_state["vcb_a_small"] = 12

        # 2. Resolve SKU names against what's actually in the
        #    current slice (the SKU palette is built from the
        #    same source).
        _slice_skus = list(_sku_vol.index.astype(str))
        _resolved = []
        _unmatched = []
        for shelf, ord_, name in _COOLER_A_PRESET:
            if name is None:
                _resolved.append((shelf, ord_, None))
                continue
            hit = _resolve_preset_sku(name, _slice_skus)
            if hit is None:
                _unmatched.append(name)
                _resolved.append((shelf, ord_, None))
            else:
                _resolved.append((shelf, ord_, hit))

        # 3. Write classic-view slot keys.
        for shelf, ord_, sku in _resolved:
            slot_key = f"vcb_a_slot_{shelf}_{ord_}"
            st.session_state[slot_key] = (
                sku if sku else "— empty —"
            )

        # 4. Stash the unmatched list for a post-rerun toast.
        st.session_state["_cooler_a_preset_unmatched"] = _unmatched
        st.session_state["_cooler_a_preset_applied"] = True

    _preset_col1, _preset_col2 = st.columns([1, 3])
    with _preset_col1:
        st.button(
            "📋 Load preset → Cooler A",
            key="vcb_a_load_preset",
            on_click=_apply_cooler_a_preset,
            help=(
                "Wipes Cooler A and loads the 29-slot preset "
                "planogram (7 Large + 10 Medium + 12 Small, "
                "with one empty Medium slot). SKU names are "
                "matched against the current slice — any names "
                "that don't appear in the data will be reported "
                "below the button."
            ),
        )
    with _preset_col2:
        if st.session_state.pop("_cooler_a_preset_applied", False):
            _unmatched = st.session_state.pop(
                "_cooler_a_preset_unmatched", []
            )
            if _unmatched:
                st.warning(
                    "Cooler A preset loaded, but these SKU "
                    "names weren't found in the current slice "
                    "(slot left empty): "
                    + ", ".join(_unmatched)
                )
            else:
                st.success("Cooler A preset loaded.")

    st.divider()

    def _render_cooler(label, key_prefix, n_large, n_medium, n_small):
        """Render one cooler's slot pickers and return a list of
        (slot_type, sku) tuples for every filled slot."""
        st.markdown(f"### {label}")

        # Build (slot_type, ordinal) sequence: Large first
        # (top shelves), then Medium, then Small (countline).
        slots = (
            [("Large", i + 1) for i in range(n_large)]
            + [("Medium", i + 1) for i in range(n_medium)]
            + [("Small", i + 1) for i in range(n_small)]
        )
        if not slots:
            st.info("No slots configured. Add slots above.")
            return []

        placements = []
        # Group by slot_type with subheaders for visual clarity.
        prev_type = None
        _shelf_label = {
            "Large":  "Big / Large shelf",
            "Medium": "Medium shelf",
            "Small":  "Countline shelf",
        }
        for slot_type, ordinal in slots:
            if slot_type != prev_type:
                st.markdown(f"**{_shelf_label[slot_type]}**")
                prev_type = slot_type
            opts = _sku_options_for_slot(slot_type)
            display_opts = ["— empty —"] + opts

            # ---- Preserve previously-selected SKU across
            # filter changes ----
            # When the user narrows the data slice (changing
            # VC_CAT / CHANNEL / RE / Months), the dropdown
            # option list is rebuilt from the new slice. If
            # the SKU previously chosen for this slot is no
            # longer in the slice, Streamlit would silently
            # reset the selectbox to its first option,
            # wiping the user's selection. We don't want
            # that — the user's planogram should survive
            # filter tweaks. So we look up the prior value
            # in session_state and, if it's a real SKU that
            # we lost, splice it back into `display_opts` so
            # Streamlit keeps it selected. The SKU's stats
            # for the *current* slice will be zero (since
            # it's not in the slice), but the moment the
            # user broadens the filters again, its stats
            # come back automatically.
            _slot_key = f"{key_prefix}_slot_{slot_type}_{ordinal}"
            _prev = st.session_state.get(_slot_key)
            if (
                _prev
                and _prev != "— empty —"
                and _prev not in opts
            ):
                # Append at the end so the natural ranking
                # (TP-descending) of in-slice options is
                # unchanged.
                display_opts = display_opts + [_prev]

            sel = st.selectbox(
                f"{slot_type} slot {ordinal}",
                options=display_opts,
                format_func=lambda x: (
                    "— empty —" if x == "— empty —" else _sku_label(x)
                ),
                key=_slot_key,
                label_visibility="visible",
            )
            if sel and sel != "— empty —":
                placements.append((slot_type, sel))
        return placements

    # ---- Render both coolers using the classic dropdowns ----
    cc1, cc2 = st.columns(2)
    with cc1:
        a_placements = _render_cooler(
            "🧊 Cooler A", "vcb_a",
            int(a_large), int(a_medium), int(a_small),
        )
    with cc2:
        b_placements = _render_cooler(
            "🧊 Cooler B", "vcb_b",
            int(b_large), int(b_medium), int(b_small),
        )

    # ---- Out-of-slice notice ----
    # When the current filters exclude a SKU that's still
    # placed in a slot, we keep it selected (so the user
    # doesn't lose work) but its stats for the current slice
    # are zero. Surface that here so the totals further down
    # aren't confusing.
    _placed_skus = {s for _t, s in (a_placements + b_placements)}
    _in_slice = set(_sku_vol.index)
    _out_of_slice = sorted(_placed_skus - _in_slice)
    if _out_of_slice:
        st.info(
            f"ℹ️ {len(_out_of_slice)} placed SKU(s) are kept in "
            f"their slots but aren't in the current data slice, "
            f"so they contribute 0 to the totals below. Broaden "
            f"the filters above to bring their stats back. SKUs: "
            f"{', '.join(_out_of_slice[:5])}"
            + (f" (+{len(_out_of_slice) - 5} more)"
               if len(_out_of_slice) > 5 else "")
        )

    # ---- Per-SKU custom double-facing multiplier (Cooler B) ----
    # When an SKU appears exactly 2 times in Cooler B, expose a
    # numeric input so the user can override the default 1.3×
    # double-facing multiplier just for that SKU. Saved in
    # session_state under a per-SKU key so it survives reruns
    # and filter changes. Cooler A still counts each SKU once
    # so the override only matters for Cooler B.
    _b_facings = {}
    for _t, _s in b_placements:
        _b_facings[_s] = _b_facings.get(_s, 0) + 1
    _b_double_skus = sorted(s for s, n in _b_facings.items() if n == 2)

    # Map of SKU → custom 2-facing multiplier (only populated
    # for SKUs the user has set explicitly). Passed into
    # _cooler_totals and the suggestions engine below.
    _custom_doubles_b = {}

    if _b_double_skus:
        with st.expander(
            f"🎚️ Per-SKU double-facing multiplier "
            f"(Cooler B) — {len(_b_double_skus)} SKU(s) at 2 facings",
            expanded=False,
        ):
            st.caption(
                "Default 2-facing multiplier is **1.3×**. Override it "
                "per SKU below — leave at 1.3 to keep the default. "
                "3+ facings still saturate at 1.5×."
            )
            _cols_per_row = 3
            for _row_start in range(
                0, len(_b_double_skus), _cols_per_row
            ):
                _row_skus = _b_double_skus[
                    _row_start: _row_start + _cols_per_row
                ]
                _cols = st.columns(_cols_per_row)
                for _ci, _sku in enumerate(_row_skus):
                    _key = f"vcb_b_double_mult__{_sku}"
                    # Seed the default 1.3 once per SKU so the
                    # number_input picks it up via session_state
                    # on its first render. Subsequent renders
                    # respect any user edits.
                    if _key not in st.session_state:
                        st.session_state[_key] = 1.3
                    with _cols[_ci]:
                        _val = st.number_input(
                            f"{_sku}",
                            min_value=0.0,
                            max_value=3.0,
                            step=0.05,
                            format="%.2f",
                            key=_key,
                            help=(
                                "Multiplier applied to this SKU's "
                                "Base TP when it has exactly 2 "
                                "facings in Cooler B."
                            ),
                        )
                        _custom_doubles_b[_sku] = float(_val)

    # ---- Totals computation per cooler ----
    def _cooler_totals(placements, use_multifacing,
                       custom_doubles=None):
        """Aggregate per-SKU facings → effective TP / vol /
        value / gm-value / transactions.

        Per user spec:
          • Cooler A (use_multifacing=False) — "count once":
            a SKU is counted exactly once no matter how many
            slots it occupies. Double / triple facing in
            Cooler A does NOT increase its contribution.
          • Cooler B (use_multifacing=True) — multi-facing
            multiplier:
                1 facing  → 1.0×
                2 facings → 1.3× (overridable per SKU via
                                  `custom_doubles[sku]`)
                3+ facings → 1.5×  (saturates)
        This lets the user compare a "single-listing"
        planogram (A) against a multi-facing planogram (B).
        """
        custom_doubles = custom_doubles or {}
        if not placements:
            return {
                "n_filled":     0,
                "n_unique":     0,
                "tp_total":     0.0,
                "value_total":  0.0,
                "gm_value_total": 0.0,
                "txn_total":    0.0,
                "rows":         pd.DataFrame(),
            }
        # Count facings per SKU.
        facings = {}
        for _slot_type, sku in placements:
            facings[sku] = facings.get(sku, 0) + 1

        rows = []
        for sku, n in facings.items():
            if use_multifacing:
                mult = _facing_mult(
                    n, custom_double=custom_doubles.get(sku)
                )
            else:
                # Per user spec for Cooler A ("first list"):
                # a SKU is **counted once** regardless of how
                # many slots it occupies (double / triple
                # facing still contributes 1×). So mult is 1.0
                # for any n ≥ 1.
                mult = 1.0 if n >= 1 else 0.0
            base_tp = float(_sku_tp.get(sku, 0.0))
            base_mrp = float(_sku_mrp.get(sku, 0.0))
            base_gm = float(_sku_gm.get(sku, 0.0))
            base_txn = float(_sku_txn.get(sku, 0.0))
            eff_tp = base_tp * mult
            # Effective monthly volume in a "typical" outlet =
            # effective TP (volume per outlet per month).
            # Value and GM-value follow.
            eff_value = eff_tp * base_mrp
            eff_gm_value = eff_value * base_gm
            # Transactions scale by the same multiplier — more
            # facings → more outlet-month sales events
            # (diminishing for Cooler B, linear for Cooler A).
            eff_txn = base_txn * mult
            rows.append({
                "SKU":          sku,
                "Brand":        _sku_brand.get(sku, ""),
                "Size":         _sku_size.get(sku, "?"),
                "Facings":      n,
                "Mult":         mult,
                "Base TP":      base_tp,
                "Eff TP":       eff_tp,
                "MRP":          base_mrp,
                "GM":           base_gm,
                "Value":        eff_value,
                "GM Value":     eff_gm_value,
                "Base Txns":    base_txn,
                "Eff Txns":     eff_txn,
            })
        rows_df = pd.DataFrame(rows).sort_values(
            "GM Value", ascending=False
        ).reset_index(drop=True)
        return {
            "n_filled":      sum(facings.values()),
            "n_unique":      len(facings),
            "tp_total":      float(rows_df["Eff TP"].sum()),
            "value_total":   float(rows_df["Value"].sum()),
            "gm_value_total":float(rows_df["GM Value"].sum()),
            "txn_total":     float(rows_df["Eff Txns"].sum()),
            "rows":          rows_df,
        }

    # Cooler A → linear (no multi-facing rule), Cooler B →
    # multi-facing curve. Per user spec.
    a_tot = _cooler_totals(a_placements, use_multifacing=False)
    b_tot = _cooler_totals(
        b_placements, use_multifacing=True,
        custom_doubles=_custom_doubles_b,
    )

    # ---- Summary panels ----
    st.divider()
    st.subheader("📊 Cooler totals")
    st.caption(
        "All totals are reported on a per-outlet per-month basis "
        "(same units as Throughput). **Cooler A counts each SKU "
        "once** regardless of how many slots it fills — extra "
        "facings don't add anything. **Cooler B applies the "
        "multi-facing multiplier** (1× / 1.3× default / 1.5× "
        "saturating) — so doubling facings yields more, but with "
        "diminishing returns. The **2-facing multiplier** can be "
        "overridden per SKU above when any SKU has 2 facings in "
        "Cooler B."
    )

    def _fmt_inr_compact_vc(v):
        v = float(v)
        sign = "-" if v < 0 else ""
        v = abs(v)
        if v >= 1e7:
            return f"{sign}₹{v / 1e7:,.2f} Cr"
        if v >= 1e5:
            return f"{sign}₹{v / 1e5:,.2f} L"
        return f"{sign}₹{v:,.0f}"

    sum_a, sum_b = st.columns(2)
    for col, tot, label in (
        (sum_a, a_tot, "Cooler A"),
        (sum_b, b_tot, "Cooler B"),
    ):
        with col:
            st.markdown(f"#### {label}")
            m1, m2 = st.columns(2)
            m1.metric("Filled slots", f"{tot['n_filled']:,}")
            m2.metric("Unique SKUs", f"{tot['n_unique']:,}")
            m3, m4 = st.columns(2)
            m3.metric(
                "Total TP / Pieces",
                f"{tot['tp_total']:,.1f}",
                help=(
                    "Units sold per outlet per month — sum of "
                    "effective TPs across SKUs."
                ),
            )
            m4.metric(
                "Total Value (TP × MRP)",
                _fmt_inr_compact_vc(tot["value_total"]),
            )
            st.metric(
                "GM Value (TP × MRP × GM)",
                _fmt_inr_compact_vc(tot["gm_value_total"]),
            )
            st.metric(
                "Total transactions",
                f"{tot['txn_total']:,.0f}",
                help=(
                    "Total units of the placed SKUs sold across "
                    "the slice (i.e. how many units left the "
                    "shelves), scaled by facings (linear for "
                    "Cooler A, multi-facing curve for Cooler B)."
                ),
            )

            if tot["rows"].empty:
                st.info("No SKUs placed yet.")
            else:
                # ---- Build a friendlier SKU-details table ----
                # Original column names like "Eff TP", "Mult",
                # "Eff Txns" were unclear. We rename everything
                # to plain English, add a totals row at the
                # bottom, and use clean column formatting so
                # the user can read the table at a glance.
                _src = tot["rows"]
                _show = pd.DataFrame({
                    "SKU":           _src["SKU"],
                    "Brand":         _src["Brand"],
                    "Size":          _src["Size"],
                    "Facings":       _src["Facings"].astype(int),
                    "Facing mult.":  _src["Mult"],
                    "Pieces / mo":   _src["Eff TP"],
                    "MRP (₹)":       _src["MRP"],
                    "GM index":      _src["GM"],
                    "Sales (₹)":     _src["Value"],
                    "Profit (₹)":    _src["GM Value"],
                    "Transactions":  _src["Eff Txns"],
                })

                # Append a totals row for quick read-off.
                _totals_row = pd.DataFrame([{
                    "SKU":           "▶ TOTAL",
                    "Brand":         "",
                    "Size":          "",
                    "Facings":       int(_show["Facings"].sum()),
                    "Facing mult.":  np.nan,
                    "Pieces / mo":   float(_show["Pieces / mo"].sum()),
                    "MRP (₹)":       np.nan,
                    "GM index":      np.nan,
                    "Sales (₹)":     float(_show["Sales (₹)"].sum()),
                    "Profit (₹)":    float(_show["Profit (₹)"].sum()),
                    "Transactions":  float(_show["Transactions"].sum()),
                }])
                _show = pd.concat(
                    [_show, _totals_row], ignore_index=True
                )

                st.dataframe(
                    _show,
                    use_container_width=True,
                    hide_index=True,
                    height=min(420, 60 + 35 * len(_show)),
                    column_config={
                        "SKU": st.column_config.TextColumn(
                            "SKU",
                            help="Stock-keeping unit placed in this cooler.",
                        ),
                        "Brand": st.column_config.TextColumn(
                            "Brand",
                        ),
                        "Size": st.column_config.TextColumn(
                            "Size",
                            help="Large / Medium / Small — drives which shelf the SKU can go on.",
                        ),
                        "Facings": st.column_config.NumberColumn(
                            "Facings",
                            format="%d",
                            help="Number of slots this SKU occupies in the cooler.",
                        ),
                        "Facing mult.": st.column_config.NumberColumn(
                            "Facing mult.",
                            format="%.2fx",
                            help=(
                                "Throughput multiplier from facings. "
                                "Cooler A = 1.0x always (count once). "
                                "Cooler B = 1.0/1.3/1.5x for 1/2/3+ "
                                "facings — the 2-facing value can be "
                                "overridden per SKU."
                            ),
                        ),
                        "Pieces / mo": st.column_config.NumberColumn(
                            "Pieces / mo",
                            format="%.2f",
                            help="Estimated units sold per outlet per month after the facing multiplier.",
                        ),
                        "MRP (₹)": st.column_config.NumberColumn(
                            "MRP (₹)",
                            format="₹%d",
                            help="Latest MRP for this SKU from the source file.",
                        ),
                        "GM index": st.column_config.NumberColumn(
                            "GM index",
                            format="%.2f",
                            help="Gross-margin index from the uploaded GM file.",
                        ),
                        "Sales (₹)": st.column_config.NumberColumn(
                            "Sales (₹)",
                            format="₹%d",
                            help="Pieces × MRP — estimated revenue per outlet per month.",
                        ),
                        "Profit (₹)": st.column_config.NumberColumn(
                            "Profit (₹)",
                            format="₹%d",
                            help="Sales × GM index — estimated gross profit per outlet per month.",
                        ),
                        "Transactions": st.column_config.NumberColumn(
                            "Transactions",
                            format="%.0f",
                            help="Sales events (outlet × month cells with pieces > 0) scaled by facings.",
                        ),
                    },
                )

    # ---- Suggestions engine ----
    # Looks at what's currently in each cooler and proposes
    # concrete next moves to lift the user's chosen objective
    # (Sales / Profit / Transactions / All-of-them). Two move
    # families:
    #   1. "Add a facing" — take a SKU already in the cooler
    #      with headroom on its multi-facing curve and bump
    #      its facings (only meaningful for Cooler B; Cooler A
    #      counts once).
    #   2. "Replace a SKU" — swap the weakest placed SKU in a
    #      slot with the strongest unplaced SKU eligible for
    #      that slot type.
    # The "uplift" we report is the marginal change in the
    # chosen metric if the move were applied — straight
    # arithmetic on Volume/Sales/Profit/Txns.
    st.divider()
    st.subheader("💡 Suggestions")
    st.caption(
        "Pick which cooler to optimise and what to optimise for. "
        "The engine looks at every placed SKU and every eligible "
        "unplaced SKU and proposes the moves with the highest "
        "expected uplift."
    )

    sg_c1, sg_c2, sg_c3 = st.columns([1, 1, 1])
    with sg_c1:
        _sg_cooler = st.radio(
            "Cooler",
            options=["Cooler A", "Cooler B"],
            horizontal=True,
            key="vcb_sg_cooler",
        )
    with sg_c2:
        _sg_goal = st.selectbox(
            "Optimise for",
            options=["Sales", "Profit", "Transactions", "All"],
            index=1,
            key="vcb_sg_goal",
            help=(
                "Sales = Pieces × MRP. "
                "Profit = Sales × GM index. "
                "Transactions = sales events. "
                "All = equal-weighted blend of all three."
            ),
        )
    with sg_c3:
        _sg_top_n = st.number_input(
            "Top N suggestions",
            min_value=3, max_value=20, value=8, step=1,
            key="vcb_sg_topn",
        )

    # Pick the source placements / multifacing flag for the
    # selected cooler.
    if _sg_cooler == "Cooler A":
        _sg_placements = a_placements
        _sg_multifacing = False
        _sg_n_large = int(a_large)
        _sg_n_medium = int(a_medium)
        _sg_n_small = int(a_small)
    else:
        _sg_placements = b_placements
        _sg_multifacing = True
        _sg_n_large = int(b_large)
        _sg_n_medium = int(b_medium)
        _sg_n_small = int(b_small)

    def _sku_metric_per_facing(sku, current_facings, use_multifacing):
        """Marginal contribution to (sales, profit, txns) if
        we go from `current_facings` to `current_facings + 1`
        facings of `sku` in a cooler with the given facing
        rule. For Cooler A (count-once) the marginal is the
        full base metric for a brand-new SKU (0 → 1) and
        zero thereafter.

        For Cooler B, if the user has set a custom double-facing
        multiplier for this SKU, use it whenever a 2-facing state
        is involved in the before/after.
        """
        base_tp = float(_sku_tp.get(sku, 0.0))
        base_mrp = float(_sku_mrp.get(sku, 0.0))
        base_gm = float(_sku_gm.get(sku, 0.0))
        base_txn = float(_sku_txn.get(sku, 0.0))
        if use_multifacing:
            _cd = _custom_doubles_b.get(sku) if _sg_multifacing else None
            mult_before = _facing_mult(current_facings, custom_double=_cd)
            mult_after = _facing_mult(
                current_facings + 1, custom_double=_cd
            )
        else:
            mult_before = 1.0 if current_facings >= 1 else 0.0
            mult_after = 1.0 if (current_facings + 1) >= 1 else 0.0
        d_mult = mult_after - mult_before
        d_vol = base_tp * d_mult
        d_sales = d_vol * base_mrp
        d_profit = d_sales * base_gm
        d_txns = base_txn * d_mult
        return d_vol, d_sales, d_profit, d_txns

    def _score(d_sales, d_profit, d_txns, goal):
        """Single-number score for ranking moves under the
        user's chosen goal."""
        if goal == "Sales":
            return d_sales
        if goal == "Profit":
            return d_profit
        if goal == "Transactions":
            return d_txns
        # "All" — normalise each component to a 0-1 scale
        # relative to its base SKU stats so they sum
        # meaningfully. We use a softer blend: equal-weighted
        # rank-style sum after dividing by typical magnitudes
        # in this slice.
        _norm_sales = (
            d_sales / max(float(_sku_mrp.max()) * float(_sku_tp.max()), 1.0)
        )
        _norm_profit = (
            d_profit / max(
                float(_sku_mrp.max()) * float(_sku_tp.max())
                * float(_sku_gm.max()), 1.0
            )
        )
        _norm_txns = d_txns / max(float(_sku_txn.max()), 1.0)
        return _norm_sales + _norm_profit + _norm_txns

    # Count current facings per SKU in this cooler.
    _cur_facings = {}
    for _t, _s in _sg_placements:
        _cur_facings[_s] = _cur_facings.get(_s, 0) + 1
    _cur_skus = set(_cur_facings.keys())

    # Slot occupancy by shelf — which shelves have empty
    # slots, and which placed SKU is the weakest one we'd
    # consider swapping out.
    _slot_counts = {
        "Large":  _sg_n_large,
        "Medium": _sg_n_medium,
        "Small":  _sg_n_small,
    }
    _placed_by_shelf = {"Large": [], "Medium": [], "Small": []}
    for _t, _s in _sg_placements:
        _placed_by_shelf[_t].append(_s)

    suggestions = []

    # -------- Move type 1: add a facing (B) / add SKU to
    # empty slot (A and B) --------
    # For each shelf, if there is an empty slot, propose
    # adding the best unplaced eligible SKU. If the shelf is
    # full but the cooler uses multi-facing (Cooler B), we
    # consider bumping facings on the currently-best-performing
    # placed SKU instead.
    for shelf_name, n_slots in _slot_counts.items():
        filled = len(_placed_by_shelf[shelf_name])
        empty_slots = n_slots - filled
        eligible_sizes = _slot_eligible.get(shelf_name, [])

        # Best unplaced eligible SKU for empty slots.
        if empty_slots > 0:
            mask = _sku_size.isin(eligible_sizes)
            candidates = [
                s for s in _sku_vol.index[mask]
                if s not in _cur_skus
            ]
            # Score each: marginal from 0 → 1 facing.
            scored = []
            for s in candidates:
                d_vol, d_sales, d_profit, d_txns = (
                    _sku_metric_per_facing(s, 0, _sg_multifacing)
                )
                sc = _score(d_sales, d_profit, d_txns, _sg_goal)
                scored.append((
                    sc, s, d_vol, d_sales, d_profit, d_txns
                ))
            scored.sort(key=lambda r: r[0], reverse=True)
            for sc, s, d_vol, d_sales, d_profit, d_txns in scored[:3]:
                if sc <= 0:
                    continue
                suggestions.append({
                    "Move":      "Add new SKU",
                    "Where":     f"{shelf_name} shelf (empty slot)",
                    "SKU":       s,
                    "From":      "—",
                    "To":        "1 facing",
                    "Δ Pieces":  d_vol,
                    "Δ Sales":   d_sales,
                    "Δ Profit":  d_profit,
                    "Δ Txns":    d_txns,
                    "_score":    sc,
                })

        # For Cooler B (and only B), suggest bumping facings on
        # already-placed SKUs that haven't saturated yet.
        if _sg_multifacing:
            for s in _placed_by_shelf[shelf_name]:
                cur = _cur_facings.get(s, 0)
                if cur >= 3:
                    continue  # already saturated at 1.5x
                # Need a free slot of this shelf type (or a
                # larger one that accepts this SKU's size).
                # For simplicity we only allow bumping when
                # the same shelf has empty slots.
                if empty_slots <= 0:
                    continue
                d_vol, d_sales, d_profit, d_txns = (
                    _sku_metric_per_facing(s, cur, True)
                )
                sc = _score(d_sales, d_profit, d_txns, _sg_goal)
                if sc <= 0:
                    continue
                suggestions.append({
                    "Move":      "Add a facing",
                    "Where":     f"{shelf_name} shelf",
                    "SKU":       s,
                    "From":      f"{cur} facing(s)",
                    "To":        f"{cur + 1} facing(s)",
                    "Δ Pieces":  d_vol,
                    "Δ Sales":   d_sales,
                    "Δ Profit":  d_profit,
                    "Δ Txns":    d_txns,
                    "_score":    sc,
                })

    # -------- Move type 2: replace a SKU --------
    # For each shelf, take the weakest placed SKU (lowest
    # score per facing) and propose replacing one of its
    # facings with the strongest unplaced eligible SKU.
    for shelf_name, n_slots in _slot_counts.items():
        placed = _placed_by_shelf[shelf_name]
        if not placed:
            continue
        eligible_sizes = _slot_eligible.get(shelf_name, [])

        # Per-placed-SKU score per facing (with current count).
        weakest = None
        weakest_score = None
        for s in set(placed):
            cur = _cur_facings.get(s, 0)
            # Marginal *lost* if we remove one facing.
            d_vol, d_sales, d_profit, d_txns = (
                _sku_metric_per_facing(s, cur - 1, _sg_multifacing)
            )
            sc = _score(d_sales, d_profit, d_txns, _sg_goal)
            if weakest is None or sc < weakest_score:
                weakest = s
                weakest_score = sc

        if weakest is None:
            continue

        # Strongest unplaced eligible candidate.
        mask = _sku_size.isin(eligible_sizes)
        candidates = [
            s for s in _sku_vol.index[mask]
            if s not in _cur_skus
        ]
        best_new = None
        best_new_score = None
        best_new_metrics = None
        for s in candidates:
            d_vol, d_sales, d_profit, d_txns = (
                _sku_metric_per_facing(s, 0, _sg_multifacing)
            )
            sc = _score(d_sales, d_profit, d_txns, _sg_goal)
            if best_new is None or sc > best_new_score:
                best_new = s
                best_new_score = sc
                best_new_metrics = (
                    d_vol, d_sales, d_profit, d_txns
                )

        if best_new is None:
            continue

        # Net uplift = score_new - score_lost.
        net_score = best_new_score - weakest_score
        if net_score <= 0:
            continue

        d_vol_lost, d_sales_lost, d_profit_lost, d_txns_lost = (
            _sku_metric_per_facing(
                weakest, _cur_facings[weakest] - 1, _sg_multifacing
            )
        )
        d_vol_new, d_sales_new, d_profit_new, d_txns_new = (
            best_new_metrics
        )
        suggestions.append({
            "Move":      "Replace SKU",
            "Where":     f"{shelf_name} shelf",
            "SKU":       f"{weakest} → {best_new}",
            "From":      f"1 facing of {weakest}",
            "To":        f"1 facing of {best_new}",
            "Δ Pieces":  d_vol_new - d_vol_lost,
            "Δ Sales":   d_sales_new - d_sales_lost,
            "Δ Profit":  d_profit_new - d_profit_lost,
            "Δ Txns":    d_txns_new - d_txns_lost,
            "_score":    net_score,
        })

    # Sort by score, take top N.
    suggestions.sort(key=lambda r: r["_score"], reverse=True)
    suggestions = suggestions[: int(_sg_top_n)]

    if not suggestions:
        st.info(
            "No positive-uplift moves found for the current "
            "planogram. Try adding more empty slots, choosing a "
            "different goal, or relaxing the data filters."
        )
    else:
        sg_df = pd.DataFrame(suggestions).drop(columns=["_score"])
        st.dataframe(
            sg_df,
            use_container_width=True,
            hide_index=True,
            height=min(420, 60 + 35 * len(sg_df)),
            column_config={
                "Move": st.column_config.TextColumn(
                    "Move",
                    help="What kind of change is being proposed.",
                ),
                "Where": st.column_config.TextColumn("Where"),
                "SKU": st.column_config.TextColumn("SKU"),
                "From": st.column_config.TextColumn("From"),
                "To": st.column_config.TextColumn("To"),
                "Δ Pieces": st.column_config.NumberColumn(
                    "Δ Pieces", format="%.2f",
                    help="Change in units / outlet / month.",
                ),
                "Δ Sales": st.column_config.NumberColumn(
                    "Δ Sales", format="₹%d",
                    help="Change in revenue / outlet / month.",
                ),
                "Δ Profit": st.column_config.NumberColumn(
                    "Δ Profit", format="₹%d",
                    help="Change in gross profit / outlet / month.",
                ),
                "Δ Txns": st.column_config.NumberColumn(
                    "Δ Txns", format="%.0f",
                    help="Change in transactions.",
                ),
            },
        )
        st.caption(
            "Ranked by your selected objective. Apply moves manually "
            "in the slot pickers above and the totals will update."
        )

    # ---- Export full planogram to Excel ----
    st.divider()
    st.subheader("⬇️ Export planogram")
    st.caption(
        "Download the entire planogram — both coolers' slot "
        "layouts, the slot-wise details with totals (one row "
        "per occupied slot), and the current suggestions — as "
        "a multi-sheet Excel file."
    )

    def _build_planogram_workbook():
        """Assemble the full planogram into an Excel workbook
        (bytes) with one sheet per logical section."""
        from openpyxl import Workbook
        from openpyxl.styles import (
            Font, PatternFill, Alignment, Border, Side
        )
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        # Remove default empty sheet — we add named ones below.
        wb.remove(wb.active)

        # Shared styles.
        _header_font = Font(bold=True, color="FFFFFF", size=11)
        _header_fill = PatternFill(
            "solid", fgColor="4472C4"
        )
        _total_font = Font(bold=True)
        _total_fill = PatternFill(
            "solid", fgColor="D9E1F2"
        )
        _thin = Side(border_style="thin", color="BFBFBF")
        _border = Border(
            left=_thin, right=_thin, top=_thin, bottom=_thin
        )
        _center = Alignment(horizontal="center", vertical="center")

        def _write_df(ws, df, start_row=1, totals=False):
            """Write a dataframe to a worksheet with header
            styling. Optionally style the last row as totals."""
            # Header.
            for j, col in enumerate(df.columns, start=1):
                c = ws.cell(row=start_row, column=j, value=col)
                c.font = _header_font
                c.fill = _header_fill
                c.alignment = _center
                c.border = _border
            # Body.
            for i, (_, row) in enumerate(
                df.iterrows(), start=start_row + 1
            ):
                for j, col in enumerate(df.columns, start=1):
                    val = row[col]
                    # Convert pd.NA / NaN to empty.
                    if pd.isna(val):
                        val = ""
                    c = ws.cell(row=i, column=j, value=val)
                    c.border = _border
                    if totals and i == start_row + len(df):
                        c.font = _total_font
                        c.fill = _total_fill
            # Auto width.
            for j, col in enumerate(df.columns, start=1):
                width = max(
                    [len(str(col))]
                    + [
                        len(str(v)) if v is not None else 0
                        for v in df[col].tolist()
                    ]
                )
                ws.column_dimensions[get_column_letter(j)].width = (
                    min(40, max(10, width + 2))
                )

        # ---- Sheet 1: Summary ----
        ws_sum = wb.create_sheet("Summary")
        _summary_rows = []
        for label, tot in (
            ("Cooler A", a_tot),
            ("Cooler B", b_tot),
        ):
            _summary_rows.append({
                "Cooler":          label,
                "Filled slots":    int(tot["n_filled"]),
                "Unique SKUs":     int(tot["n_unique"]),
                "Pieces / mo":     round(float(tot["tp_total"]), 2),
                "Sales (₹)":       round(float(tot["value_total"]), 2),
                "Profit (₹)":      round(float(tot["gm_value_total"]), 2),
                "Transactions":    round(float(tot["txn_total"]), 2),
            })
        _write_df(
            ws_sum, pd.DataFrame(_summary_rows), start_row=1
        )
        # Add slot-skeleton info below.
        _sk_start = len(_summary_rows) + 4
        ws_sum.cell(
            row=_sk_start - 1, column=1,
            value="Slot skeleton"
        ).font = Font(bold=True, size=12)
        _skel_rows = [
            {
                "Cooler": "Cooler A",
                "Large slots": int(a_large),
                "Medium slots": int(a_medium),
                "Small slots": int(a_small),
                "Total slots": int(a_large + a_medium + a_small),
            },
            {
                "Cooler": "Cooler B",
                "Large slots": int(b_large),
                "Medium slots": int(b_medium),
                "Small slots": int(b_small),
                "Total slots": int(b_large + b_medium + b_small),
            },
        ]
        _write_df(
            ws_sum, pd.DataFrame(_skel_rows),
            start_row=_sk_start
        )
        # Slice info.
        _info_row = _sk_start + len(_skel_rows) + 3
        ws_sum.cell(
            row=_info_row, column=1, value="Data slice"
        ).font = Font(bold=True, size=12)
        ws_sum.cell(
            row=_info_row + 1, column=1,
            value=f"Outlets: {_slice_outlets:,}"
        )
        ws_sum.cell(
            row=_info_row + 2, column=1,
            value=f"Rows: {_slice_rows:,}"
        )
        ws_sum.cell(
            row=_info_row + 3, column=1,
            value=(
                f"Months ({_n_months}): "
                f"{', '.join(m.replace('_V', '') for m in _months_avail)}"
            ),
        )
        ws_sum.cell(
            row=_info_row + 4, column=1,
            value=(
                "Cooler A counts each SKU once. "
                "Cooler B applies multi-facing multiplier "
                "(1.0x / 1.3x default / 1.5x for 1 / 2 / 3+ "
                "facings; 2-facing value can be overridden per SKU)."
            ),
        )

        # ---- Sheet 2 & 3: Slot layouts ----
        for label, placements, n_l, n_m, n_s in (
            ("Cooler A layout", a_placements,
             int(a_large), int(a_medium), int(a_small)),
            ("Cooler B layout", b_placements,
             int(b_large), int(b_medium), int(b_small)),
        ):
            ws = wb.create_sheet(label)
            # Bucket placements by shelf.
            by_shelf = {"Large": [], "Medium": [], "Small": []}
            for slot_type, sku in placements:
                by_shelf[slot_type].append(sku)
            layout_rows = []
            for shelf, n_slots in (
                ("Large", n_l), ("Medium", n_m), ("Small", n_s)
            ):
                filled = by_shelf[shelf]
                for i in range(n_slots):
                    sku = filled[i] if i < len(filled) else ""
                    if sku:
                        layout_rows.append({
                            "Shelf":        shelf,
                            "Slot #":       i + 1,
                            "SKU":          sku,
                            "Brand":        _sku_brand.get(sku, ""),
                            "Size":         _sku_size.get(sku, ""),
                            "Base TP":      round(float(_sku_tp.get(sku, 0.0)), 2),
                            "MRP (₹)":      round(float(_sku_mrp.get(sku, 0.0)), 2),
                            "GM index":     round(float(_sku_gm.get(sku, 0.0)), 3),
                        })
                    else:
                        layout_rows.append({
                            "Shelf":        shelf,
                            "Slot #":       i + 1,
                            "SKU":          "(empty)",
                            "Brand":        "",
                            "Size":         "",
                            "Base TP":      "",
                            "MRP (₹)":      "",
                            "GM index":     "",
                        })
            if layout_rows:
                _write_df(
                    ws, pd.DataFrame(layout_rows), start_row=1
                )
            else:
                ws.cell(
                    row=1, column=1,
                    value="No slots configured."
                )

        # ---- Sheet 4 & 5: Slot-wise details (per cooler) ----
        # One row per **occupied slot** (not per unique SKU). A SKU
        # that occupies N slots contributes N rows; per-slot values
        # (Volume / mo, Sales, Profit, Transactions) are the SKU's
        # effective totals divided by N, so the column sums in the
        # totals row equal the cooler-level totals shown in the
        # Summary sheet.
        #   Cooler A:  mult = 1.0 → per-slot Volume = base_TP / N
        #   Cooler B:  mult ∈ {1.0, 1.3, 1.5} (or custom 2-facing
        #              override) → per-slot Volume = base_TP × mult / N
        # Rows are ordered Large → Medium → Small to match the
        # corresponding "Cooler X layout" sheet.
        for label, tot, placements in (
            ("Cooler A slots", a_tot, a_placements),
            ("Cooler B slots", b_tot, b_placements),
        ):
            ws = wb.create_sheet(label)
            if tot["rows"].empty or not placements:
                ws.cell(
                    row=1, column=1,
                    value="No SKUs placed."
                )
                continue
            # Index per-SKU effective totals so we can divide by
            # facings to get the per-slot contribution.
            _src = tot["rows"].set_index("SKU")
            _shelf_order = {"Large": 0, "Medium": 1, "Small": 2}
            _slot_rows = []
            # Track slot # within each shelf in placements order.
            _shelf_counters = {"Large": 0, "Medium": 0, "Small": 0}
            for slot_type, sku in placements:
                _shelf_counters[slot_type] += 1
                slot_no = _shelf_counters[slot_type]
                if sku not in _src.index:
                    # Shouldn't happen — placements are built from
                    # the same SKU set — but guard anyway.
                    continue
                _r = _src.loc[sku]
                n_facings = int(_r["Facings"])
                if n_facings <= 0:
                    continue
                _slot_rows.append({
                    "Shelf":         slot_type,
                    "Slot #":        slot_no,
                    "SKU":           sku,
                    "Brand":         _r["Brand"],
                    "Size":          _r["Size"],
                    "SKU facings":   n_facings,
                    "Facing mult.":  round(float(_r["Mult"]), 2),
                    "Pieces / mo":   round(float(_r["Eff TP"]) / n_facings, 2),
                    "MRP (₹)":       round(float(_r["MRP"]), 2),
                    "GM index":      round(float(_r["GM"]), 3),
                    "Sales (₹)":     round(float(_r["Value"]) / n_facings, 2),
                    "Profit (₹)":    round(float(_r["GM Value"]) / n_facings, 2),
                    "Transactions":  round(float(_r["Eff Txns"]) / n_facings, 2),
                })
            if not _slot_rows:
                ws.cell(
                    row=1, column=1,
                    value="No SKUs placed."
                )
                continue
            _details = pd.DataFrame(_slot_rows)
            # Stable order: Large → Medium → Small, then Slot #.
            _details["_shelf_ord"] = _details["Shelf"].map(_shelf_order)
            _details = (
                _details.sort_values(["_shelf_ord", "Slot #"])
                        .drop(columns=["_shelf_ord"])
                        .reset_index(drop=True)
            )
            # Append slot-wise totals row.
            _totals_xlsx = pd.DataFrame([{
                "Shelf":         "TOTAL",
                "Slot #":        len(_details),
                "SKU":           "",
                "Brand":         "",
                "Size":          "",
                "SKU facings":   "",
                "Facing mult.":  "",
                "Pieces / mo":   round(float(_details["Pieces / mo"].sum()), 2),
                "MRP (₹)":       "",
                "GM index":      "",
                "Sales (₹)":     round(float(_details["Sales (₹)"].sum()), 2),
                "Profit (₹)":    round(float(_details["Profit (₹)"].sum()), 2),
                "Transactions":  round(float(_details["Transactions"].sum()), 2),
            }])
            _details = pd.concat(
                [_details, _totals_xlsx], ignore_index=True
            )
            _write_df(ws, _details, start_row=1, totals=True)

        # ---- Sheet 6: Suggestions ----
        ws_sg = wb.create_sheet("Suggestions")
        if suggestions:
            _sg_xlsx = pd.DataFrame(suggestions).drop(
                columns=["_score"]
            )
            # Round numeric columns.
            for _c in ("Δ Pieces", "Δ Sales", "Δ Profit", "Δ Txns"):
                _sg_xlsx[_c] = _sg_xlsx[_c].astype(float).round(2)
            # Header note.
            ws_sg.cell(
                row=1, column=1,
                value=(
                    f"Suggestions for {_sg_cooler} — "
                    f"optimising for {_sg_goal}"
                ),
            ).font = Font(bold=True, size=12)
            _write_df(ws_sg, _sg_xlsx, start_row=3)
        else:
            ws_sg.cell(
                row=1, column=1,
                value=(
                    f"No positive-uplift moves found for "
                    f"{_sg_cooler} (goal: {_sg_goal})."
                ),
            )

        # Serialise to bytes.
        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        return bio.getvalue()

    try:
        _xlsx_bytes = _build_planogram_workbook()
        _fname = (
            f"planogram_"
            f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        st.download_button(
            "📥 Download planogram as Excel",
            data=_xlsx_bytes,
            file_name=_fname,
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True,
            help=(
                "Multi-sheet workbook: Summary, Cooler A / B "
                "layouts, slot-wise details with totals "
                "(one row per occupied slot), and the current "
                "suggestions."
            ),
        )
    except Exception as _xlsx_err:
        st.error(
            f"Could not build Excel export: {_xlsx_err}. "
            "If openpyxl isn't installed in your environment, "
            "`pip install openpyxl` and reload."
        )

    # ---- Quick-view slot map ----
    # A compact visual at the bottom that, at a glance, shows
    # which SKU landed in which slot for each cooler. Each slot
    # is drawn as a labeled rectangle, grouped by shelf type
    # (Large on top, Medium in middle, Small/countline at
    # bottom — same vertical order as a real visicooler). The
    # total estimated volume (= Total Effective TP, units per
    # outlet per month) is shown right above the chart for
    # quick context.
    st.divider()
    st.subheader("🗺️ Slot map — what's where")
    st.caption(
        "Quick visual: which SKU sits in which slot, per "
        "cooler. Shelves are stacked top-to-bottom (Large → "
        "Medium → Small/countline) just like the actual "
        "visicooler. Empty slots are shown faded."
    )

    # Total estimated volume callouts (effective TP = volume
    # per outlet per month, summed across SKUs with the
    # multi-facing multiplier applied).
    tv1, tv2 = st.columns(2)
    tv1.metric(
        "Cooler A · Total est. pieces",
        f"{a_tot['tp_total']:,.1f}",
        help=(
            "Sum of effective TPs across placed SKUs in "
            "Cooler A. Units = pieces per outlet per month."
        ),
    )
    tv2.metric(
        "Cooler B · Total est. pieces",
        f"{b_tot['tp_total']:,.1f}",
        help=(
            "Sum of effective TPs across placed SKUs in "
            "Cooler B. Units = pieces per outlet per month."
        ),
    )

    def _short_sku(name, max_len=22):
        """Shorten a SKU name for tight cell labels."""
        if name is None:
            return ""
        s = str(name).strip()
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    # Stable colour per SKU across both coolers so the eye
    # can track the same SKU side-by-side. We seed from the
    # union of placed SKUs (sorted for determinism).
    _all_skus = sorted({
        sku for _t, sku in (a_placements + b_placements)
    })
    _palette = px.colors.qualitative.Set3 + \
        px.colors.qualitative.Pastel + \
        px.colors.qualitative.Set2
    _sku_color = {
        sku: _palette[i % len(_palette)]
        for i, sku in enumerate(_all_skus)
    }

    def _slot_map_figure(placements, n_large, n_medium,
                         n_small, title):
        """Render a compact slot-grid figure for one cooler.

        Layout: three shelf rows stacked top-to-bottom (Large,
        Medium, Small). Slots within a shelf are laid out
        left-to-right. Each slot is a coloured rectangle
        labeled with the (shortened) SKU name. Empty slots
        are drawn faded with no label.
        """
        import plotly.graph_objects as go

        # Build a flat ordered list of (shelf, idx_in_shelf)
        # entries matching `placements` ordering (which is
        # Large-first → Medium → Small).
        shelves = [
            ("Large", n_large),
            ("Medium", n_medium),
            ("Small", n_small),
        ]
        # Map slot_type → list of SKUs filled (in order).
        by_type = {"Large": [], "Medium": [], "Small": []}
        for slot_type, sku in placements:
            by_type[slot_type].append(sku)

        fig = go.Figure()
        # Track shelf y-positions (top = Large).
        # Each shelf row has height = 1. Place Large at y=2,
        # Medium at y=1, Small at y=0 (so Large draws on top).
        y_map = {"Large": 2, "Medium": 1, "Small": 0}

        # Max slots in any shelf controls x-extent. Use at
        # least 1 to avoid degenerate axes when a cooler has
        # no slots of a given type.
        max_slots = max(
            [n_large, n_medium, n_small, 1]
        )

        for shelf_name, n_slots in shelves:
            y_row = y_map[shelf_name]
            filled = by_type[shelf_name]
            for i in range(n_slots):
                x0, x1 = i, i + 0.92  # small gap between slots
                y0, y1 = y_row + 0.08, y_row + 0.92
                if i < len(filled):
                    sku = filled[i]
                    color = _sku_color.get(sku, "#cccccc")
                    label = _short_sku(sku)
                    # Per-SKU stats for hover.
                    base_tp = float(_sku_tp.get(sku, 0.0))
                    mrp = float(_sku_mrp.get(sku, 0.0))
                    size = _sku_size.get(sku, "?")
                    hover = (
                        f"<b>{sku}</b><br>"
                        f"Shelf: {shelf_name}<br>"
                        f"Slot: {i + 1}<br>"
                        f"Size: {size}<br>"
                        f"Base TP: {base_tp:,.2f}<br>"
                        f"MRP: ₹{mrp:,.0f}"
                    )
                else:
                    color = "rgba(220,220,220,0.35)"
                    label = ""
                    hover = (
                        f"<b>Empty</b><br>"
                        f"Shelf: {shelf_name}<br>"
                        f"Slot: {i + 1}"
                    )

                # Rectangle.
                fig.add_shape(
                    type="rect",
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color="rgba(60,60,60,0.5)",
                              width=1),
                    fillcolor=color,
                    layer="below",
                )
                # Invisible scatter point at the centre for
                # hover + the text label.
                fig.add_trace(go.Scatter(
                    x=[(x0 + x1) / 2],
                    y=[(y0 + y1) / 2],
                    mode="text",
                    text=[label],
                    textfont=dict(size=10, color="#222"),
                    hoverinfo="text",
                    hovertext=[hover],
                    showlegend=False,
                ))

        # Shelf labels on the left (as y-axis tick labels).
        # Empty shelves still get a row so the layout is
        # consistent across coolers.
        tickvals = []
        ticktext = []
        for shelf_name, n_slots in shelves:
            tickvals.append(y_map[shelf_name] + 0.5)
            ticktext.append(
                f"{shelf_name}<br>"
                f"<span style='font-size:10px;color:#888'>"
                f"({n_slots} slots)</span>"
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            xaxis=dict(
                range=[-0.1, max_slots + 0.1],
                showgrid=False, zeroline=False,
                showticklabels=False, fixedrange=True,
            ),
            yaxis=dict(
                range=[-0.1, 3.1],
                tickvals=tickvals,
                ticktext=ticktext,
                showgrid=False, zeroline=False,
                fixedrange=True,
            ),
            height=260,
            margin=dict(l=70, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    map_c1, map_c2 = st.columns(2)
    with map_c1:
        if not a_placements and (
            int(a_large) + int(a_medium) + int(a_small)
        ) == 0:
            st.info("Cooler A has no slots configured.")
        else:
            st.plotly_chart(
                _slot_map_figure(
                    a_placements,
                    int(a_large), int(a_medium), int(a_small),
                    "Cooler A",
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with map_c2:
        if not b_placements and (
            int(b_large) + int(b_medium) + int(b_small)
        ) == 0:
            st.info("Cooler B has no slots configured.")
        else:
            st.plotly_chart(
                _slot_map_figure(
                    b_placements,
                    int(b_large), int(b_medium), int(b_small),
                    "Cooler B",
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # ---- Footer help ----
    with st.expander("ℹ️ How are the totals computed?"):
        st.markdown(
            """
- **Base TP per SKU** = Σ(pieces in selected months) ÷ active outlets ÷ # months selected. Computed from the **sidebar-filtered slice further narrowed by the tab-level filters above** (VC_CAT / CHANNEL / RE / months). Picking a subset of months bases throughput on that seasonal window instead of all 15 months.
- **Base Transactions per SKU** = count of (outlet × month) cells with pieces > 0 for that SKU in the slice — same definition the rest of the dashboard uses.
- **Facings rule**:
    - **Cooler A — count once**: a SKU is counted exactly once no matter how many slots it fills. Double or triple facing in Cooler A does NOT increase any total.
    - **Cooler B — multi-facing curve**:
        - 1 facing → **1.0×**
        - 2 facings → **1.3× (default — overridable per SKU via the "Per-SKU double-facing multiplier" expander above)**
        - 3+ facings → **1.5×** (saturates)
- **Effective TP** = Base TP × multiplier. **Pieces** is reported on the same per-outlet-per-month basis as Throughput, so it equals Effective TP numerically.
- **Value** = Effective TP × Latest MRP. **GM Value** = Value × GM index. **Effective Transactions** = Base Transactions × multiplier.
- **Size-fit rule**: Large SKU → Large slot only; Medium SKU → Large or Medium; Small SKU → Medium or Small. (Equivalently: Large slots accept Large + Medium SKUs; Medium slots accept Medium + Small; Small slots accept Small only.)
- SKUs missing from the filtered slice (zero pieces) won't appear in the dropdowns even if they're in the SKU-Size file, because we have no TP / MRP / GM signal to scale them.
            """
        )



# =========================================================
# TOP-LEVEL TIME-PERIOD SHEETS
# =========================================================

# The SKU Priority Lister tab is only shown if the user has
# uploaded the (optional) Gross Margin file. The tab ranks
# SKUs by Throughput × Latest MRP × Gross Margin, so without
# a GM file the whole ranking would collapse — we keep the
# tab hidden until the file is there.
# The VC Planogram Builder tab is shown only if the SKU-Size
# file is uploaded — without size info we can't enforce the
# slot-fit rules.
_period_tab_label = f"📅 Custom Period ({PERIOD_LABEL})"
_tab_labels = [_period_tab_label]
if _gm_available:
    _tab_labels.append("⭐ SKU Priority Lister")
if _sku_size_available:
    _tab_labels.append("🧊 VC Planogram Builder")

period_tabs = st.tabs(_tab_labels)

with period_tabs[0]:
    render_period_dashboard(
        PERIOD_COL_NAME, PERIOD_MONTHS, "sel", PERIOD_LABEL
    )

# Priority Lister sits at tab index 1 when GM file is uploaded.
if _gm_available:
    with period_tabs[1]:
        render_sku_priority_lister()

# VC Planogram Builder sits at the end of the tab list, after
# the Priority Lister (if present). Its tab index is 2 when
# GM is uploaded, otherwise 1.
if _sku_size_available:
    _vc_builder_idx = 2 if _gm_available else 1
    with period_tabs[_vc_builder_idx]:
        render_vc_builder()
