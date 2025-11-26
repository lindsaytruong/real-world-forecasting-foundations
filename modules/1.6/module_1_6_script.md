# Module 1.6: First Contact with the Data
## Video Tutorial Script — Detailed Narration

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

## 5Q Framework: First Contact Checklist

> **[Show the checklist on screen]**

Here's how first contact maps to our 5Q Framework.

For Q1 — Target — we need to verify the target is clear, numeric, and clean. Can we actually use this as a forecasting target?

For Q2 — Metric — We want to select a metric that is mathematically sound and fits the business objective. When computing metrics, we're looking for issues like missingness, outliers, or duplicates that would bias our evaluation. If we can't trust our metrics, we can't trust our models.

For Q3 — Horizon and Level — we need enough history at the right frequency and granularity. You can't forecast 12 weeks ahead if you only have 10 weeks of data.

For Q4 — Data and Drivers — we're looking for early behavioral signals. How much intermittency? How volatile? These shape what methods will work.

And Q5 — Ownership — we're documenting who owns this data, where it comes from, how often it refreshes.

We'll check each box as we work through the module. By the end, you'll know exactly where this dataset stands.

---

## Section 1: Setup

> **[Switch to notebook, run setup cell]**

Let's start with our standard setup. Pandas, numpy, matplotlib for plotting. We're using TSForge, which wraps a lot of the utility functions we'll use today.

> **[Run cell, show "✓ Setup complete"]**

Good. Let's load some data.

---

## Section 2: Load Data

> **[Show the markdown cell with messify parameters]**

Now, here's something important. We're loading the M5 dataset with `messify=True`. 

What does that mean? We're deliberately corrupting the data to simulate real-world quality issues.

Look at this table:

- We're converting 15% of zeros to NA — this simulates missing sales reports. In production, sometimes data just doesn't show up.
- We're adding 150 duplicate rows — this simulates ETL bugs. Someone ran a job twice, or there's a join that's creating duplicates.
- We're corrupting the dtypes — dates and numbers become strings. This happens all the time when data passes through different systems.

Why do this? Because clean Kaggle data teaches you nothing about production reality. I want you to see what real data looks like and learn to handle it.

> **[Run the load cell]**

You can see it's loading about 47 million rows. That's daily data for around 30,000 item-store combinations over about 5 years.

---

## Section 3: Pre-Aggregation Checks

> **[Show the pre-aggregation markdown cell]**

Now, here's the most important concept in this module: **order of operations**.

Before we can aggregate this daily data to weekly, we have to fix certain issues. And the order matters.

Look at this table — if we don't fix string dtypes, the groupby sum will fail or produce garbage. If we don't handle NAs properly, they'll get dropped or propagate incorrectly. If we don't remove duplicates, the same day's sales will be summed twice into the weekly total.

So the order is: Fix dtypes, then fill NAs, then remove duplicates, then aggregate.

Get this order wrong and you get silent failures. The code runs, but the numbers are wrong. That's the worst kind of bug.

---

### 3.1 Fix Data Types & Handle NAs

> **[Show the markdown explanation]**

Messification converted our `ds` and `y` columns to strings. And here's a subtle thing — NaN values became the literal string `"nan"`. Not a null, not a missing value — the three-character string N-A-N.

> **[Run the dtype check cell]**

See? Both columns show `object` dtype. That's pandas-speak for "I don't know what this is, probably strings." As an additional tip, object dtypes are very memory heavy and will slow down your workflow. In pandas avoid object dtypes when possible and use `category` for categorical data.

> **[Show the to_numeric explanation]**

Now we fix it. The key is `errors='coerce'` — this converts anything that can't be parsed into a proper NaN. So the string "nan" becomes a real null value again.

> **[Run the fix dtypes cell]**

Good. Now look at this next part.

> **[Show the domain decision markdown]**

This is a domain decision. In retail, when a sales value is missing, it almost always means "no sales were recorded" — which is effectively zero. The store was open, nobody bought that item.

But in other domains, this isn't true. If you're forecasting sensor data and you have a missing value, that might mean the sensor failed — you don't know what the actual reading was. In finance, a missing price might mean the market was closed.

So we're making an explicit choice here: fill NAs with zero. Document this decision. It matters.

> **[Run the fill NAs cell]**

About 9% of our values were NA. Now they're zeros.

This checks our Q1 box — Target. Our target is now numeric and clean.

---

### 3.2 Check for Duplicates

> **[Show the duplicates explanation]**

A duplicate means the same item-store-date combination appears more than once.

Think about what happens if we don't catch this. Say item X at store Y sold 5 units on Monday. But that row is duplicated in our data. When we aggregate to weekly, we sum all the daily values — and now that Monday contributes 10 units instead of 5.

Your weekly totals are wrong. Your forecasts are wrong. And you might never notice.

> **[Run the duplicate check cell]**

We found about 290 duplicates out of 47 million rows. That's a tiny percentage, but it matters. In production, I've seen datasets where 10-20% of rows were duplicated due to a bad join. Always check.

> **[Run the drop duplicates cell]**

Gone. This checks part of our Q2 box — we've removed an issue that would bias our evaluation.

---

### 3.3 Daily Data Summary

> **[Run the summary cell]**

Before we aggregate, let's confirm what we have. About 47 million rows, 7 columns, date range from January 2011 to June 2016.

One row per item-store-day. Hierarchy columns are preserved. This looks right.

Now we can aggregate.

---

## Section 4: Aggregate to Weekly

> **[Show the "why weekly" markdown]**

Why weekly? Three reasons.

First, it aligns with business planning. Replenishment cycles, labor scheduling, promotional planning — these typically happen at the weekly level. Daily forecasts are often more precision than the business can actually use.

Second, it reduces noise. Daily retail sales are erratic. Weekends look different from weekdays. Weekly totals smooth this out and make patterns more visible.

Third, it's practical. We go from 47 million rows to about 6 million. That's 7x fewer rows, which means faster iteration during development.

> **[Show the performance notes markdown]**

A couple of technical notes on this groupby. `as_index=False` gives us a DataFrame directly without needing to call reset_index. `observed=True` skips empty category combinations — big speedup when you have sparse hierarchies.

> **[Run the aggregation cell]**

Down to about 6.8 million rows. Week starts on Tuesday to align with Walmart's fiscal calendar.

---

## Section 5: Post-Aggregation Checks

> **[Show the post-aggregation markdown with 5Q link table]**

Now we check the data we'll actually model. And notice how each check connects back to our 5Q Framework.

Series count connects to Q3 — it determines our compute requirements and cross-validation strategy.

Date range also connects to Q3 — we need enough history for the patterns we want to capture.

Missing weeks connects to both Q2 and Q3 — gaps break lag features and affect model training.

Zeros percentage connects to Q4 — high intermittency is a behavioral signal that shapes our model choice.

Let's go through each one.

---

### 5.1 Series Count

> **[Show the explanation markdown]**

How many unique item-store combinations do we have? This determines computational scale.

> **[Run the series count cell]**

About 30,000 series. The median series has around 220 weeks of data, with some shorter ones.

> **[Run the plot cell]**

You can see most series have close to the full history, but there's a tail of shorter series. These might be new products that were introduced partway through.

30,000 series is manageable — we can run models on the full dataset. But during development, we might want to sample for faster iteration.

---

### 5.2 Date Range

> **[Show the rule of thumb markdown]**

Here's a rule of thumb: you need 2-3x your forecast horizon for meaningful patterns. If you're forecasting 12 weeks ahead, you want at least 36 weeks of history — and more is better, especially if you need to capture annual seasonality.

> **[Run the date range cell]**

We have about 280 weeks — that's over 5 years. Plenty of history for annual patterns.

This checks our Q3 box — we have sufficient history at the item-store-week level.

---

### 5.3 Missing Weeks

> **[Show the thresholds markdown]**

Missing weeks create gaps that break time-based features. If you're calculating a 4-week lag and week 3 is missing, what do you do?

Here are the thresholds I use:

Less than 5% missing — you're fine. Minor forward-fill or interpolation will handle it.

5 to 15% missing — investigate the cause. Plan your imputation strategy carefully.

More than 15% missing — this series might not be usable. Consider excluding it or using special handling.

> **[Run the missing weeks cell]**

Median is around 5%, but the max is over 30%. So most series are in good shape, but some have significant gaps.

> **[Run the plot cell]**

You can see the distribution is right-skewed. Most series are healthy, but there's a tail of problematic ones. We'll address these in the data preparation module.

This further informs our Q2 assessment — missingness could bias our evaluation if we don't handle it properly.

---

### 5.4 Zeros (Intermittency)

> **[Show the intermittency explanation markdown]**

High percentage of zeros means intermittent demand. This is common in retail — many products just don't sell every week at every store.

Why does it matter? Two reasons.

First, metric choice. MAPE — Mean Absolute Percentage Error — is undefined when the actual value is zero. You're dividing by zero. So if you have lots of zeros, you can't use MAPE. Use RMSE or WRMSSE instead.

Second, model choice. Standard methods like ARIMA and ETS struggle with intermittent series. You might need specialized methods like Croston, or tree-based models that handle zeros naturally.

> **[Run the zeros cell]**

About 18% zeros. That's moderate intermittency — not extreme, but enough that we need to be thoughtful about metrics and methods.

This checks our Q4 box — we've identified a key behavioral signal that will shape our modeling decisions.

---

## Section 6: Automated First Contact Check

> **[Show the explanation markdown]**

Running these checks manually is educational, but in practice you want to automate it. TSForge has a function that runs all these checks with a single call.

> **[Run the automated check cell]**

There's our summary. 30,000 series, 280 weeks of history, 5% median missing, 18% zeros, weekly frequency confirmed.

Use this function on every new dataset. It takes seconds and tells you immediately whether you have a workable forecasting problem.

---

## Section 7: Save Output for Next Module

> **[Show the save output markdown]**

Before we wrap up, let's save this cleaned data for the next module. We've done the work of fixing dtypes, filling NAs, removing duplicates, and aggregating to weekly. No need to repeat that.

> **[Run the save cell]**

Saved to `1_6_output.parquet`. About 6.8 million rows with all hierarchy columns preserved. Module 1.7 will pick up right here.

---

## Section 8: Putting It All Together

> **[Show the 5Q summary table]**

Let's go back to our 5Q checklist and see where we stand.

Q1 — Target — checked. Our target is numeric, cleaned from string corruption.

Q2 — Metric — checked. We identified 5% median missing weeks and 18% zeros. MAPE isn't suitable for this dataset.

Q3 — Horizon and Level — checked. We have 280+ weeks at the item-store-week level. Sufficient for a 12-week horizon.

Q4 — Data and Drivers — partially checked. We identified moderate intermittency. We still need to assess volatility and seasonality — that's coming in the diagnostics module.

Q5 — Ownership — still open. We need to document the data source, refresh cadence, and stakeholders. That's an organizational task, not a technical one.

> **[Show the key decisions table]**

Here are the key decisions we made and why:

We filled NAs with zero because in retail, missing typically means no sales recorded.

We aggregated to weekly because it aligns with business planning cycles.

We kept all 30,000 series because it's a manageable scale. We can sample for rapid iteration during development.

> **[Show the order of operations]**

And here's the reference card. Memorize this order:

Load, fix dtypes, fill NAs, remove duplicates, aggregate, run checks.

Getting this order wrong causes silent failures. Make it automatic. Make it a habit.

---

## Section 9: Next Steps

> **[Show the findings table]**

Based on what we found today, here's what we need to address in upcoming modules:

High missing weeks in some series — we'll handle that in Module 1.10 on data preparation.

Moderate intermittency — this will influence our model selection in Module 2.

Thirty thousand series — we might benefit from hierarchical approaches, which we'll explore in Module 1.7.

> **[Show the coming up list]**

Here's what's coming:

Module 1.7 — we'll dig into the M5 dataset structure. What are the files? Which fields are target, dynamic, or static? What's known at forecast time?

Module 1.8 — Diagnostics. We'll profile the entire portfolio for volatility, seasonality, drift, and stability.

Module 1.9 — Portfolio Analysis with GenAI. How do you understand 30,000 series? You use AI as an analytical partner.

Module 1.10 — Data Preparation. Fill gaps, merge features, and enforce known-at-time discipline.

---

## Closing

> **[On camera or voiceover]**

That's first contact. Five to ten minutes of checks that tell you whether you have a workable forecasting problem.

The key insight from today: every check connects to a business question. We're not just running diagnostics for the sake of it — we're verifying that the data can support the 5Q Framework.

Make this a habit. Run first contact on every new dataset before you do anything else. It's how you earn the right to model.

See you in Module 1.7.

---

*End of Script*
