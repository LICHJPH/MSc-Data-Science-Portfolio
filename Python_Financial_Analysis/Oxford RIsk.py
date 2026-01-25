import requests
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import linregress
import matplotlib.pyplot as plt
# 1. Define your Supabase details
SUPABASE_URL = "https://pvgaaikztozwlfhyrqlo.supabase.co/rest/v1/assets?select=*"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2Z2FhaWt6dG96d2xmaHlycWxvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NDE2MjUsImV4cCI6MjA2MzQxNzYyNX0.iAqMXnJ_sJuBMtA6FPNCRcYnKw95YkJvY3OhCIZ77vI"

# 2. Set up headers
headers = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}"
}

# 3. Send GET request
resp = requests.get(SUPABASE_URL, headers=headers)
resp.raise_for_status()  # will raise an error if the call failed

# 4. Load into a DataFrame for analysis
assets = pd.DataFrame(resp.json())

# 5. Inspect
print(assets.head())
print(f"Total records retrieved: {len(assets)}")



#6. load the file
personality = pd.read_csv('/Users/cola/Downloads/personality.csv')

#7. merge the data
combined = pd.merge(personality, assets, on='_id', how='inner')
combined
combined.to_csv('combined.csv', index=False)


#8. Filter only GBP-denominated assets
gbp_assets = combined[combined['asset_currency'] == 'GBP'].copy()

# 9. Sum each person’s GBP holdings across all their GBP rows
gbp_totals = (
    gbp_assets
    .groupby('_id', as_index=False)
    .agg(total_gbp_assets=('asset_value', 'sum'))
)

# 10. Identify the individual with the highest total
top_row = gbp_totals.loc[gbp_totals['total_gbp_assets'].idxmax()]

print(f"Individual ID with highest GBP assets: {top_row['_id']}")
print(f"Total GBP assets: £{top_row['total_gbp_assets']:.2f}")


# 11.  Look up their risk-tolerance score
risk_score = (
    combined
    .loc[combined['_id'] == top_row['_id'], 'risk_tolerance']
    .iloc[0]
)
print(f"Risk tolerance: {risk_score}")

# Exploratory Data Analysis
# Summary statistics
print("\nSummary statistics for numeric variables:")
print(combined.describe())

# Correlation matrix
# This will automatically ignore non-numeric dtypes(Asses currency)
corr = combined.corr(numeric_only=True)
print("\nCorrelation matrix:")
print(corr)

#Distribution of key scores
#loop and generate the distribution for each personality trait
for col in ['composure', 'impact_desire', 'confidence', 'risk_tolerance', 'impulsivity']:
    plt.figure()
    plt.hist(combined[col], bins=20)
    plt.title(f'Distribution of {col.replace("_", " ").title()}')
    plt.xlabel(col.replace("_", " ").title())
    plt.ylabel('Frequency')
    plt.show()



#Scatter: total GBP assets vs. risk tolerance
#    First compute total GBP per individual
gbp_only = combined[combined['asset_currency']=='GBP']
gbp_totals = gbp_only.groupby('_id')['asset_value'].sum().reset_index()

#    Merge back for plotting
merged_for_plot = gbp_totals.merge(
    combined[['_id','risk_tolerance']].drop_duplicates('_id'),
    on='_id'
)

plt.figure()
plt.scatter(
    merged_for_plot['asset_value'],
    merged_for_plot['risk_tolerance']
)
plt.title('Total GBP Assets vs. Risk Tolerance')
plt.xlabel('Total GBP Assets (£)')
plt.ylabel('Risk Tolerance')
plt.show()

#Asset-allocation breakdown
print("\nCounts by asset allocation:")
print(combined['asset_allocation'].value_counts())

#Currency breakdown
print("\nCounts by currency:")
print(combined['asset_currency'].value_counts())


#plot the gbp assets allocation vs each persnality traits

def plot_gbp_asset_vs_trait(df, trait):
    """
    Scatter‐plot total GBP asset_value vs. average of `trait`,
    broken down by asset_allocation.
    """
    # 0. Keep GBP assets only
    gbp = df[df['asset_currency'] == 'GBP']

    # 1. Group & aggregate on GBP subset
    summary = (
        gbp
        .groupby('asset_allocation', as_index=False)
        .agg(
            total_gbp_value=('asset_value', 'sum'),
            avg_trait=(trait, 'mean')
        )
    )

    #Plot
    plt.figure()
    plt.scatter(summary['total_gbp_value'], summary['avg_trait'])
    for _, row in summary.iterrows():
        plt.annotate(
            row['asset_allocation'],
            (row['total_gbp_value'], row['avg_trait']),
            textcoords="offset points",
            xytext=(5,5),
            ha='left'
        )
    plt.title(f'Total GBP Assets vs. Avg. {trait.replace("_"," ").title()}')
    plt.xlabel('Total GBP Asset Value (£)')
    plt.ylabel(f'Average {trait.replace("_"," ").title()}')
    plt.show()

#correlation between each personality traits and total gbp assets

# Run for each behavioural trait:
for t in ['confidence','composure','impulsivity','impact_desire','risk_tolerance']:
    plot_gbp_asset_vs_trait(combined, t)





traits = ['confidence','composure','impulsivity','impact_desire','risk_tolerance']
results = []
for trait in traits:
    # merge totals & trait
    df_trait = gbp_totals.merge(
        combined[['_id', trait]].drop_duplicates('_id'),
        on='_id'
    )
    r, p = pearsonr(df_trait['asset_value'], df_trait[trait])
    results.append((trait, r, p))

# Display
for trait, r, p in results:
    print(f"{trait:15s}  r = {r:.3f}, p-value = {p:.3e}")


# (a) Filter to GBP and sum per person & per asset class
gbp = combined[combined['asset_currency']=='GBP']
asset_sums = (
    gbp
    .groupby(['_id','asset_allocation'])['asset_value']
    .sum()
    .unstack(fill_value=0)    # now each asset_allocation is its own column
)

# (b) Extract one row per person of their trait scores
traits = (
    combined
    .loc[:,['_id','confidence','risk_tolerance','composure','impulsivity','impact_desire']]
    .drop_duplicates('_id')
    .set_index('_id')
)

# (c) Join sums + traits into one DataFrame
df = asset_sums.join(traits, how='inner')

# (d) Run Pearson’s r & p-value for every combination
results = []
for asset_type in asset_sums.columns:
    for trait in traits.columns:
        r, p = pearsonr(df[asset_type], df[trait])
        results.append((asset_type, trait, r, p))

# (e) Turn into a nice table and sort by |r|
res_df = pd.DataFrame(
    results,
    columns=['asset_type','trait','r','p_value']
)
res_df['abs_r'] = res_df['r'].abs()
res_df = res_df.sort_values('abs_r', ascending=False).drop(columns='abs_r')

print(res_df.to_string(index=False))

#Main question, Does GBP track the rest? So does it represent the rest of the currency
currency_sums = (
    combined
    .groupby(['_id','asset_currency'])['asset_value']
    .sum()
    .unstack(fill_value=0)
)

currency_sums['others_total'] = (
    currency_sums
    .drop(columns=['GBP'])
    .sum(axis=1))
# 2. Run a simple linear regression of others_total ~ GBP
lr = linregress(currency_sums['GBP'], currency_sums['others_total'])

print("Linear fit: others_total = intercept + slope * GBP")
print(f"  slope       = {lr.slope:.3f}")
print(f"  intercept   = {lr.intercept:.3f}")
print(f"  R-squared   = {lr.rvalue**2:.3f}")
print(f"  p-value     = {lr.pvalue:.3e}")




plt.figure()
plt.scatter(currency_sums['GBP'], currency_sums['others_total'], label='Data')
# plot regression line
x = currency_sums['GBP']
plt.plot(x, lr.intercept + lr.slope * x, color='red', label='Fit')
plt.title('Other-Currency Total vs. GBP Holdings')
plt.xlabel('GBP Total (£)')
plt.ylabel('Sum of Other Currencies (£)')
plt.legend()
plt.tight_layout()
plt.show()