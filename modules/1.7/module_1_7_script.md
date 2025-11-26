# Module 1.7: Understanding the M5 Dataset
## Video Tutorial Script — Updated for New HTML Structure

---

## Opening

> **[On camera or voiceover]**

Welcome to Module 1.7 — Understanding the M5 Dataset.

In Module 1.6, we ran first-contact checks on the M5 data. We loaded it, cleaned it, aggregated to weekly, and assessed whether it could support our 5Q Framework.

But we used a helper function — `tsf.load_m5()` — that did a lot of work behind the scenes. Today, we're going to look under the hood.

> **[Open the Module 1.7 HTML file in browser]**

I've created an interactive explorer for the M5 dataset. Open the HTML file and follow along — I'll tell you where to click as we go.

You'll see five tabs that take you on a journey:
- **📊 Raw Data** — The source files as downloaded from Kaggle
- **📚 Vocabulary** — The four classifications you need to know
- **🔍 Explore** — Deep dive into each file's fields
- **🛠️ Transform** — How TSForge converts raw to forecast-ready
- **🚀 Cheat Sheet** — Your go-to reference

Let's start at the beginning.

---

## Tab 1: Raw Data — The M5 Data Structure

> **[In the HTML: You should be on the "📊 Raw Data" tab]**

### What is M5?

The M5 dataset comes from Walmart and contains ~5 years of daily sales data for thousands of items across multiple stores. Your goal: predict future sales for 3,049 products across 10 stores in 3 US states (CA, TX, WI).

> **[In the HTML: Read the "Why This Matters" callout]**

Understanding how these tables connect and what each field represents is not just academic — it's the difference between a model that works in notebooks and one that survives production. **Most forecasting failures happen at the data preparation stage, not the modeling stage.**

### The Three Raw Files

> **[In the HTML: Look at the three file cards]**

These are the **raw, untransformed files** exactly as downloaded from Kaggle. No preprocessing has been applied yet.

| File | What It Contains | Shape |
|------|------------------|-------|
| `sales_train.csv` | Historical daily sales (the target) + hierarchy IDs | 30,490 × 1,947 |
| `calendar.csv` | Date information, events, SNAP schedules | 1,969 × 14 |
| `sell_prices.csv` | Weekly pricing — handle with care! | 6,841,121 × 4 |

Each file has a specific role. Sales is your target. Calendar is your known-at-time features. Prices is where the leakage trap lives.

### Data Hierarchies

> **[In the HTML: Scroll down to see the hierarchy diagrams]**

**Product Hierarchy:**
```
Category (3) → Department (7) → Item (3,049)
   FOODS   →    FOODS_3    →  FOODS_3_090
```

**Location Hierarchy:**
```
State (3) → Store (10)
   CA    →   CA_3
```

**Combined:** 3,049 items × 10 stores = 30,490 unique item-store combinations.

> **[In the HTML: Click "Learn the vocabulary that makes this click" to continue]**

---

## Tab 2: Vocabulary — The Four Classifications

> **[In the HTML: Click the "📚 Vocabulary" tab]**

Before we explore the files, you need to understand four terms. These classifications determine what you can safely use as features.

### The Four Types

> **[In the HTML: Look at the four classification cards]**

**✓ STATIC** — Never changes over time
- Examples: item_id, store_id, dept_id, cat_id, state_id
- Always safe to use as features

**📅 KNOWN-AT-TIME** — Changes over time, but known in advance
- Examples: weekday, month, events, SNAP schedules
- Safe because they come from calendars and schedules

**⚠️ DYNAMIC (Unknown)** — Changes over time, NOT known at forecast time
- Examples: sell_price, actual weather
- High leakage risk — use lagged values only

**🎯 TARGET** — What you're predicting
- Examples: daily sales (d_1, d_2, ... d_1941)
- Can use past values as lag features, never future

### The Critical Question

> **[In the HTML: Read "The Critical Question" box]**

Before using any feature, ask: **"Will I actually have this data when I need to make the forecast?"**

If the answer is no — or even "maybe not" — you need to either lag the feature or exclude it.

> **[In the HTML: Click "Now let's see what's actually in each file" to continue]**

---

## Tab 3: Explore — What's in Each File?

> **[In the HTML: Click the "🔍 Explore" tab]**

Now let's examine every field in each file. You'll see detailed tables with:
- Field name and type
- Classification (Static, Known-At-Time, Dynamic, Target)
- Description and examples
- Whether it's known at forecast time
- Leakage risk level

### sales_train.csv

> **[In the HTML: Look at the sales_train.csv section]**

**Shape:** 30,490 rows × 1,947 columns

This file has two parts:
1. **6 identifier/hierarchy columns** — All STATIC, always safe
2. **1,941 daily sales columns (d_1 to d_1941)** — Your TARGET

> **[In the HTML: Note the warnings about wide format and target leakage]**

**Key insight:** The raw data is in WIDE format — each day is a column. This needs to be transformed to LONG format for forecasting.

### calendar.csv

> **[In the HTML: Scroll to the calendar.csv section]**

**Shape:** 1,969 rows × 14 columns

Everything here is safe! All fields are either date identifiers or known-at-time features.

> **[In the HTML: Note the warning about wm_yr_wk being used for joining]**

The `d` column maps to the d_X columns in sales_train. The `wm_yr_wk` column is how you join prices.

### sell_prices.csv — THE LEAKAGE TRAP

> **[In the HTML: Scroll to the sell_prices.csv section]**

**Shape:** 6,841,121 rows × 4 columns

> **[In the HTML: Note this file has the most warnings]**

This is where most people make mistakes. Prices change week to week, and **you do NOT know next week's actual price when making a forecast**.

**Common mistake:**
```python
# ❌ WRONG: Joining future prices
df = df.merge(prices, on=['item_id', 'store_id', 'wm_yr_wk'])
```

This gives you next week's actual price to predict next week's sales. That's leakage!

**What you CAN do:**
1. Use lagged prices (last week's price)
2. Use planned promotional prices
3. Forecast prices separately

> **[In the HTML: Click "You understand the data. Now let's transform it" to continue]**

---

## Tab 4: Transform — From Raw to Forecast-Ready

> **[In the HTML: Click the "🛠️ Transform" tab]**

Now you understand the raw structure. Here's how TSForge transforms it.

### The One-Line Solution

> **[In the HTML: Look at the TSForge callout box at the top]**

```python
from tsforge.data import load_m5

df = load_m5()
```

That's it. One line. But what does it actually do?

### What load_m5() Handles

> **[In the HTML: Read the list of what load_m5() does]**

1. **Creates unique_id** — Combines item_id + store_id
2. **Melts wide → long** — Converts d_X columns to rows
3. **Joins calendar** — Maps d_X to actual dates
4. **Renames to y** — Standard target column name
5. **Preserves static features** — Keeps hierarchy columns

### The Interactive Schema Mapping

> **[In the HTML: Hover over fields in the transformation visualizer]**

Try hovering over fields on the left (raw) or right (transformed) side. You'll see how each raw field maps to the output schema.

**Raw M5 Fields → TSForge Transformed Schema**

| Raw Field | → | Transformed Field |
|-----------|---|-------------------|
| item_id + store_id | → | unique_id |
| d_X columns + calendar.date | → | ds |
| d_X values | → | y |
| dept_id, cat_id, state_id | → | (preserved as static features) |

### Before and After

> **[In the HTML: Look at the Before/After data tables]**

**BEFORE (Wide format):**
- 30,490 rows × 1,947 columns
- Each row is one item-store
- Days are columns (d_1, d_2, d_3, ...)

**AFTER (Long format):**
- 59,181,490 rows × ~6 columns
- Each row is one observation
- Standard schema: unique_id, ds, y

> **[In the HTML: Click "Use the cheat sheet to validate your work" to continue]**

---

## Tab 5: Cheat Sheet — Your Go-To Reference

> **[In the HTML: Click the "🚀 Cheat Sheet" tab]**

This is your reference page. Bookmark it.

### Complete M5 Column Reference

> **[In the HTML: Look at the four column classification boxes]**

Every single column in M5 is classified here:

- **STATIC (5 total):** item_id, dept_id, cat_id, store_id, state_id
- **KNOWN-AT-TIME (11 total):** date, wday, weekday, month, year, event_*, snap_*
- **DYNAMIC Unknown (2 total):** sell_price, wm_yr_wk
- **TARGET (1,941 total):** d_1 through d_1941

### Decision Tree: Can I Use This Column?

> **[In the HTML: Look at the three decision cards]**

**✅ YES — Use it directly**
- Any STATIC column
- Any KNOWN-AT-TIME column from calendar.csv
- Date-derived features (weekday, month, year)

**⚠️ MAYBE — Use with care**
- sell_price — Use LAGGED values or PLANNED prices only
- wm_yr_wk — Only for joining; validate temporal ordering
- d_X (past values) — Can create LAG features, verify temporal split

**❌ NO — Do not use**
- sell_price for FUTURE weeks
- d_X where X ≥ forecast date
- Any value requiring "peeking" into the future

### Common Leakage Patterns to Avoid

> **[In the HTML: Look at the code examples]**

The cheat sheet shows four patterns:
1. ❌ Joining prices without time constraint
2. ❌ Using current period sales as feature
3. ❌ Rolling window with center=True
4. ✅ Correct lag feature creation with shift()

---

## Key Takeaways

> **[Summary slide]**

**The five tabs tell a story:**
- **📊 Raw Data** — The 3 files as downloaded from Kaggle (wide format, d_X columns)
- **📚 Vocabulary** — Static, Known-At-Time, Dynamic, Target
- **🔍 Explore** — Every field classified with leakage warnings
- **🛠️ Transform** — How load_m5() converts to forecast-ready format
- **🚀 Cheat Sheet** — Complete reference + decision tree

**The format transformation:**
- Raw: Wide format (30,490 rows × 1,947 columns)
- Transformed: Long format (59M rows × 6 columns)
- Schema: (unique_id, ds, y) + static features

**The critical question:**
> "Will I actually have this data when I need to make the forecast?"

---

## Next Steps

> **[Preview upcoming modules]**

Now that you understand the M5 structure:

- **Module 1.8:** Diagnostics — profile volatility, seasonality, trend across all series
- **Module 1.9:** Portfolio Analysis with GenAI — scale understanding to 30K series
- **Module 1.10:** Data Preparation — merge calendar and prices safely, avoiding leakage

We've laid the foundation. You know what the data contains, how it's structured, and where the traps are. Next, we'll actually work with these files.

See you in Module 1.8.

---

*End of Script*
