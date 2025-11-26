# Module 1.6: First Contact with the Data
## Script 

---

## Opening

> **[On camera or voiceover before showing notebook]**

Welcome to Module 1.6 — First Contact with the Data.

### The M5 Dataset

We're using the M5 dataset — about 5 years of daily Walmart sales across 30,000 item-store combinations. It's publicly available, it's real retail data, and the challenges it presents — intermittent demand, hierarchical structure, promotional effects — are exactly what you'll face in industry.

We'll walk through the M5 files and structure in detail in Module 1.7. Today we're focused on the process of first contact — the checks you run on any new dataset before doing anything else.

### What We're NOT Doing

This is **not** an attempt to compete with the M5 Kaggle winners. Those solutions were optimized for a leaderboard — complex ensembles, features engineered for a fixed test set. They're not forecasting systems.

We're focused on **process over leaderboard**:
- We're aggregating to **weekly** because that's how business planning actually works
- We're building **transferable skills** that apply to any forecasting problem
- We're building **production habits** — workflows that could run in the real world

Forget WRMSSE rankings. We're optimizing for understanding.

### First Contact

Before we build any models, we need to answer: **Can this data support a forecasting solution?**

That's what first contact is about — systematic checks that take five minutes but save hours of debugging. And it connects directly to the 5Q Framework.

Let me show you.

---

## 5Q Checklist

> **[Show the checklist table on screen]**

Here's how first contact maps to our 5Q Framework.

For Q1 — Target — we verify the target is numeric and check for NAs. Can we actually use this as a forecasting target?

For Q2 — Metric — we're looking for zeros and intermittency. High zeros means MAPE won't work.

For Q3 — Horizon and Level — we need enough date range for the patterns we want to capture.

For Q4 — Data and Drivers — we check for missing weeks, duplicates, and dtype issues that would corrupt our modeling.

For Q5 — Ownership — does the data match what stakeholders expect?

We'll check each box as we work through the module.

---

## Section 1: Setup

> **[Switch to notebook, run setup cell]**

Let's start with our standard setup. Pandas, numpy, matplotlib for plotting. We're using TSForge, which wraps the utility functions we'll use.

> **[Run cell, show "✓ Setup complete"]**

Good. Let's load some data.

---

## Section 2: Load Data

> **[Show the markdown cell]**

We're loading M5 with `messify=True`. This deliberately corrupts the data to simulate real-world quality issues.

What does messify do?

- Converts 15% of zeros to NA — simulates missing sales reports
- Adds duplicate rows — simulates ETL bugs where a job ran twice
- Corrupts the dtypes — dates and numbers become strings

Why do this? Because clean Kaggle data teaches you nothing about production reality. I want you to see what real data looks like and learn to handle it.

> **[Run the load cell]**

About 47 million rows. Daily data for 30,000 item-store combinations over 5 years.

---

## Section 3: Pre-Aggregation Checks (Daily Data)

> **[Show the pre-aggregation markdown]**

Here's the most important concept in this module: **order of operations**.

Before we aggregate daily data to weekly, we must fix certain issues. And the order matters.

If we don't fix string dtypes, the groupby sum will fail. If we don't remove duplicates, the same day's sales gets summed twice into the weekly total.

Get this order wrong and you get silent failures. The code runs, but the numbers are wrong.

---

### 3.1 Fix Data Types & Handle NAs

> **[Show the dtype explanation markdown]**

Messification converted `ds` and `y` to strings. NaN values became the literal four-character string "nan" — not a null, just text.

> **[Run the dtype check cell]**

See? Both columns show `object` dtype. Pandas-speak for "probably strings."

> **[Show the coerce explanation]**

The fix uses `errors='coerce'` — this converts anything unparseable into a proper NaN. So the string "nan" becomes a real null value.

> **[Run the fix dtypes cell]**

Good.

> **[Show the NA check markdown]**

Now we check remaining NAs. Important note: we're NOT filling these with zeros yet. That's a Module 1.10 task, after we fill gaps properly.

Why wait? Because if we fill NAs now, we lose the ability to distinguish between "actually sold zero" and "data was missing." That distinction matters for gap-filling and imputation strategies.

> **[Run the NA check cell]**

About 9% of target values are NA. We'll note this and handle it properly in data preparation.

---

### 3.2 Check for Null IDs

> **[Show the null IDs markdown]**

Rows with null ID columns can't be properly aggregated. If `item_id` is null, which item does this row belong to? It's orphan data.

In production, this happens when data pipelines fail partially, or when new products are added without proper master data setup.

> **[Run the null IDs check cell]**

We're checking all ID columns — everything except `ds` and `y`. If any have nulls, we need to decide: drop those rows, or investigate the source.

This dataset should be clean, but always check. I've seen production datasets where 5% of rows had null IDs due to a bad join.

---

### 3.3 Check for Weird Dates

> **[Show the weird dates markdown]**

Look for dates that shouldn't exist:
- Dates before 1900 — probably parsing errors
- Future dates — data leakage or timestamp bugs
- Outlier dates far from the main range — maybe a typo like 2061 instead of 2016

> **[Run the weird dates check cell]**

We're looking at the unique dates sorted. The earliest should be around January 2011, the latest around June 2016 for M5.

Also useful: group by date and sum sales, sorted latest to earliest. If you see a random date in 2020 with a tiny amount of sales, that's an outlier to investigate.

This is a sanity check that catches obvious problems. In production, I've seen datasets with dates in the year 9999 — that's a system default that means "no date set."

---

### 3.4 Data Info & Memory

> **[Show the info markdown]**

Now let's see what we're working with. The `.info()` method gives us dtypes, non-null counts, and memory usage.

> **[Run the info cell]**

A few things to notice:

First, memory usage. With `memory_usage='deep'`, pandas calculates the actual memory including string contents. This dataset is probably a few gigabytes. Important to know if you're working on a laptop with 8GB RAM.

Second, dtypes. After our fixes, `ds` should show `datetime64` and `y` should show `float64`. If they still show `object`, something went wrong.

Third, non-null counts. Quick sanity check that our NA counts match what we saw earlier.

---

### 3.5 Check for Duplicates

> **[Show the duplicates markdown]**

A duplicate means the same row — excluding the target — appears multiple times.

Think about what happens if we miss this. Item X at store Y sold 5 units on Monday. But that row is duplicated. When we aggregate to weekly, Monday contributes 10 units instead of 5.

Your numbers are wrong. Your forecasts are wrong. And you might never notice.

> **[Show the columns cell]**

First let's see what columns we have. We'll check for duplicates on all columns except the target.

> **[Run the duplicate check cell]**

We're using `keep=False` so we can see ALL duplicated rows, not just the second occurrence.

> **[Run the value counts cell]**

About 290 duplicates out of 47 million. Tiny percentage, but it matters.

> **[Run the drop duplicates cell]**

Gone. In production, I've seen datasets where 10-20% of rows were duplicated due to a bad join. Always check.

---

### 3.6 Daily Data Summary

> **[Show the summary markdown]**

Before we aggregate, confirm the daily data looks right.

> **[Run the head cell]**

One row per item-store-day. Hierarchy columns preserved. Dates look reasonable. Target is numeric.

Now we can aggregate.

---

## Section 4: Aggregate to Weekly

> **[Show the weekly markdown]**

Why weekly? Three reasons.

First, it aligns with business planning. Replenishment cycles, labor scheduling, promotional planning — these happen weekly. Daily forecasts are often more precision than the business can use.

Second, it reduces noise. Daily retail sales are erratic. Weekly totals smooth this out and make patterns visible.

Third, it's practical. We go from 47 million rows to about 6 million. Faster iteration.

> **[Show the groupby explanation]**

We group by all non-target, non-date columns and sum the target. This way we don't need to know the specific hierarchy column names — the code works on any dataset.

> **[Run the week column cell]**

The week column uses `to_period('W-SAT')` which gives weeks ending Saturday — meaning weeks start on Sunday. That's the Walmart fiscal week convention.

> **[Run the aggregation cell]**

Down to about 6.8 million rows.

---

### Memory After Aggregation

> **[Run the post-aggregation info cell]**

Let's see memory after aggregation. Should be significantly smaller.

Notice we went from ~47M rows to ~6.8M rows — about 7x reduction. Memory should drop proportionally. This is one of the benefits of weekly aggregation during development.

---

### 4.1 Date Range

> **[Show the date range markdown]**

Rule of thumb: you need 2-3x your forecast horizon for meaningful patterns. For a 12-week forecast, you want at least 36 weeks of history.

> **[Run the min date, max date, and weeks cells]**

We have about 280 weeks — over 5 years. Plenty for annual patterns and a 12-week horizon.

This checks our Q3 box — sufficient history at the item-store-week level.

---

## Section 5: First Contact Summary

> **[Show the summary markdown]**

Running these checks manually is educational, but in practice you want to automate it. Here's a function that runs all the checks in one call.

> **[Walk through the function]**

Let me explain what this function checks:

**Required columns** — does the DataFrame have `ds` and `y`?

**Data types** — is `ds` datetime? Is `y` numeric?

**NAs** — how many nulls in the date column, ID columns, and target? Note that target NAs are informational — we'll handle them in Module 1.10.

**Impossible dates** — anything before 1900 or in the future?

**Duplicates** — any repeated rows after excluding the target?

**Summary stats** — shape, series count, date range, memory.

> **[Run the function call]**

There's our summary. All checks in one place. Use this function on every new dataset.

The output tells you:
- Whether each check passed (✓) or failed (✗)
- Informational items (ℹ) that aren't failures but need attention
- Summary stats for the dataset
- Any issues that need resolution

If you see all checks passed, you're ready to proceed. If not, fix the issues first.

---

## Section 6: Save Output

> **[Show the save markdown]**

Let's save the cleaned, aggregated data for the next module.

Important: we're saving NAs as-is, not filling them with zeros. The imputation decision happens in Module 1.10 after we properly fill gaps. This preserves the distinction between "actually zero" and "data missing."

> **[Run the save cell]**

Saved to `1_6_output.parquet`. About 6.8 million rows with all hierarchy columns preserved.

Parquet format is efficient — compressed, columnar, preserves dtypes. Module 1.7 picks up right here.

---

## Section 7: Next Steps

> **[Show the next steps table]**

Here's what's coming:

**Module 1.7** — Understanding M5 structure. What are the files? Which fields are target, dynamic, or static? What's known at forecast time?

**Module 1.8** — Diagnostics. We'll profile the portfolio for volatility, seasonality, and stability.

**Module 1.9** — Portfolio analysis with GenAI. How do you understand 30,000 series? AI as an analytical partner.

**Module 1.10** — Data preparation. Fill gaps, merge calendar features, handle imputation. This is where we'll properly handle those NAs we noted today.

**Module 1.11** — Plotting and visual diagnostics. See the data before modeling.

---

## Closing

> **[On camera or voiceover]**

That's first contact. Ten minutes of checks that tell you whether you have a workable forecasting problem.

The key insights from today:

**Order matters.** Fix dtypes, then check for issues, then remove duplicates, then aggregate. Get this wrong and you get silent failures.

**Don't impute too early.** We noted the NAs but didn't fill them. That decision waits until we understand the gap structure.

**Automate the checks.** The `first_contact_check()` function runs everything in one call. Use it on every new dataset.

Make this a habit. First contact before anything else. It's how you earn the right to model.

See you in Module 1.7.

---

*End of Script*
