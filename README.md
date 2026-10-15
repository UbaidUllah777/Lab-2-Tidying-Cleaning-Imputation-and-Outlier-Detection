# Lab 2 – Tidying, Cleaning, Imputation, and Outlier Detection

**Machine Learning Programming (PROG8245)**

This repository contains the group implementation of Lab 2, focused on the data-preprocessing work that typically occurs before machine-learning modelling. The project demonstrates data inspection, reshaping, cleaning, missing-value treatment, imputation, outlier detection, visualization, and documentation across four different datasets.

The completed work is organized in a single notebook, `week5Lab.ipynb`, with each group member responsible for one major section.

---

## Project Objectives

The lab demonstrates how to:

- inspect dataset dimensions, columns, data types, missing values, and duplicates;
- distinguish between wide and long/tidy data;
- reshape data using `pandas.melt()`;
- clean and standardize string-based fields;
- convert variables to appropriate numeric and datetime types;
- evaluate missing-data strategies using `dropna()`, `fillna()`, mean, median, mode, and `SimpleImputer`;
- detect outliers visually using box plots and scatter plots;
- detect outliers statistically using Z-scores and the Interquartile Range (IQR);
- justify data-cleaning and preprocessing decisions using evidence from the data;
- produce a reproducible and well-documented Python workflow.

---

## Team Contributions

| Section | Dataset | Main Focus | Team Member |
|---|---|---|---|
| **Section 1** | PEW Research Global Restrictions on Religion | Data tidying and reshaping | **Koushik** |
| **Section 2** | Billboard Top 100 Rankings | Advanced tidying, string cleaning, missing rankings, date enhancement, trend analysis | **Ubaid Ullah** |
| **Section 3** | Vehicles / Cars | Data cleaning, missing-data handling, imputation, duplicate checks | **Ved** |
| **Section 4** | CDC Diabetes Health Indicators | Outlier detection, treatment, visualization, and feature engineering | **Sukh** |

Each section is integrated into the shared `week5Lab.ipynb` notebook so the complete group submission can be reviewed and executed in one place.

---

## Section 1 – PEW Research Dataset

**Team member:** Koushik  
**Dataset:** `data/PublicDataset_ReligiousRestrictions_2007to2022.csv`

This section uses the PEW Research Global Restrictions on Religion dataset covering 2007–2022.

### Key work completed

- loaded and inspected the dataset;
- identified identifier and measured variables;
- selected detailed Global Restrictions Index (GRI) and Social Hostilities Index (SHI) indicators;
- converted selected indicator columns from wide format to long format using `melt()`;
- preserved country, region, and year information as identifier variables;
- created a visualization of recorded indicator values;
- documented the transformation and completed the required reflection questions.

The original dataset contains **3,164 rows and 81 columns**. Seventy-one selected indicator columns are reshaped into a long-format dataset containing **224,644 rows and 6 columns**.

---

## Section 2 – Billboard Dataset

**Team member:** Ubaid Ullah  

**Dataset:** `data/billboard.csv`

This section analyzes Billboard Top 100 song rankings for the year 2000 and demonstrates advanced data tidying.

### Key work completed

- validated the raw dataset structure;
- identified song-level identifiers and weekly ranking columns;
- reshaped `wk1` through `wk76` into `Week` and `Rank` variables using `melt()`;
- applied Pandas string operations including:
  - `.str.strip()`
  - `.str.lower()`
  - `.str.replace()`
  - `.str.extract()`
- converted week numbers and chart rankings to numeric types;
- analyzed and handled missing ranking observations;
- validated Billboard rankings against the valid 1–100 range;
- converted entry dates to datetime values;
- calculated weekly chart dates using timedeltas;
- visualized the ranking history of **3 Doors Down – Kryptonite**;
- completed validation and reflection questions.

The raw dataset contains **317 songs and 81 columns**. Reshaping 76 weekly ranking columns produces **24,092 potential song-week observations**. After removing observations without a recorded ranking, **5,307 valid ranking records** remain.

The reusable `BillboardTidier` class is implemented in:

`src/billboard_tidier.py`

### Billboard data provenance

Primary dataset source:

- tidyverse / tidyr raw Billboard dataset  
  https://raw.githubusercontent.com/tidyverse/tidyr/main/data-raw/billboard.csv
- tidyr documentation  
  https://tidyr.tidyverse.org/reference/billboard.html

The tidyr documentation identifies the original source as the Whitburn Project.

---

## Section 3 – Vehicles / Cars Dataset

**Team member:** Ved  
**Dataset:** `data/vehicles.csv`

This section focuses on practical data cleaning and missing-value treatment. The notebook loads the first **1,000 vehicle records** for the lab analysis.

### Key work completed

- inspected dataset shape, columns, data types, and sample observations;
- investigated suspicious or incomplete records;
- calculated missing-value counts and percentages;
- demonstrated `isnull()`, `isna()`, `notnull()`, and `notna()`;
- checked numeric fields and applied numeric conversion where appropriate;
- identified and handled duplicate records;
- compared multiple missing-value strategies:
  - dropping rows;
  - dropping columns;
  - mean imputation;
  - median imputation;
  - mode imputation;
- visualized a numeric distribution to support the imputation decision;
- applied Scikit-Learn's `SimpleImputer`;
- compared the dataset before and after cleaning;
- completed the required reflection questions.

This section emphasizes that a cleaning strategy should be selected based on the meaning, distribution, and quality of the data rather than applying one approach automatically.

---

## Section 4 – Diabetes Dataset

**Team member:** Sukh  
**Dataset:** CDC Diabetes Health Indicators, retrieved from UCI using `ucimlrepo`

The notebook retrieves the dataset programmatically with:

```python
from ucimlrepo import fetch_ucirepo
cdc_diabetes = fetch_ucirepo(id=891)