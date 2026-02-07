# === TWITTER BRAND SENTIMENT ANALYSIS ===
# Brand Perception & Engagement Analysis for Social Media

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("📊 Loading Twitter Brand Sentiment Data...")

# Load datasets
df_train = pd.read_csv('twitter_training.csv', names=['ID', 'Brand', 'Sentiment', 'Tweet'])
df_val = pd.read_csv('twitter_validation.csv', names=['ID', 'Brand', 'Sentiment', 'Tweet'])

# Combine for comprehensive analysis
df = pd.concat([df_train, df_val], ignore_index=True)

print(f"✓ Total tweets analyzed: {len(df):,}")
print(f"✓ Brands tracked: {df['Brand'].nunique()}")
print(f"✓ Date range: Social media monitoring dataset\n")

# === DATA OVERVIEW ===
print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(df.info())
print("\n" + "="*60)

# === KEY METRICS CALCULATION ===
print("\n📈 CALCULATING KEY METRICS...\n")

# Overall sentiment distribution
sentiment_dist = df['Sentiment'].value_counts()
sentiment_pct = (sentiment_dist / len(df) * 100).round(2)

print("Overall Sentiment Distribution:")
for sent, count in sentiment_dist.items():
    pct = sentiment_pct[sent]
    print(f"  {sent:12s}: {count:6,} tweets ({pct:5.2f}%)")

# Top brands analysis
top_brands = df['Brand'].value_counts().head(15)
print(f"\n🏆 Top 15 Most Mentioned Brands:")
for brand, count in top_brands.items():
    print(f"  {brand:30s}: {count:6,} mentions")

# === BRAND HEALTH METRICS ===
def calculate_brand_metrics(df):
    """Calculate comprehensive brand health metrics"""
    brand_metrics = []
    
    for brand in df['Brand'].unique():
        brand_data = df[df['Brand'] == brand]
        total = len(brand_data)
        
        # Sentiment counts
        pos = len(brand_data[brand_data['Sentiment'] == 'Positive'])
        neg = len(brand_data[brand_data['Sentiment'] == 'Negative'])
        neu = len(brand_data[brand_data['Sentiment'] == 'Neutral'])
        irr = len(brand_data[brand_data['Sentiment'] == 'Irrelevant'])
        
        # Calculate metrics
        sentiment_score = ((pos - neg) / total * 100) if total > 0 else 0
        positive_ratio = (pos / total * 100) if total > 0 else 0
        negative_ratio = (neg / total * 100) if total > 0 else 0
        engagement = total  # Total mentions as proxy for engagement
        
        brand_metrics.append({
            'Brand': brand,
            'Total_Mentions': total,
            'Positive': pos,
            'Negative': neg,
            'Neutral': neu,
            'Irrelevant': irr,
            'Sentiment_Score': round(sentiment_score, 2),
            'Positive_Ratio': round(positive_ratio, 2),
            'Negative_Ratio': round(negative_ratio, 2),
            'Brand_Health': 'Excellent' if sentiment_score > 20 else ('Good' if sentiment_score > 0 else ('Poor' if sentiment_score < -20 else 'Fair'))
        })
    
    return pd.DataFrame(brand_metrics).sort_values('Total_Mentions', ascending=False)

brand_metrics_df = calculate_brand_metrics(df)

print("\n"+ "="*80)
print("BRAND HEALTH ANALYSIS - Top 10 Brands")
print("="*80)
print(brand_metrics_df.head(10).to_string(index=False))

# Save metrics
brand_metrics_df.to_csv('brand_health_metrics.csv', index=False)
print("\n✓ Saved: brand_health_metrics.csv")

# === VISUALIZATION 1: Overall Sentiment Distribution ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Twitter Brand Sentiment Analysis Dashboard', fontsize=18, fontweight='bold', y=0.995)

# 1.1 Sentiment Distribution Pie Chart
colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6', 'Irrelevant': '#f39c12'}
sentiment_colors = [colors.get(sent, '#3498db') for sent in sentiment_dist.index]

axes[0,0].pie(sentiment_dist.values, labels=sentiment_dist.index, autopct='%1.1f%%',
              colors=sentiment_colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0,0].set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold', pad=15)

# 1.2 Top 10 Brands by Mentions
top10_brands = brand_metrics_df.head(10)
bars = axes[0,1].barh(range(len(top10_brands)), top10_brands['Total_Mentions'], color='#3498db')
axes[0,1].set_yticks(range(len(top10_brands)))
axes[0,1].set_yticklabels(top10_brands['Brand'])
axes[0,1].set_xlabel('Total Mentions', fontweight='bold')
axes[0,1].set_title('Top 10 Most Mentioned Brands', fontsize=14, fontweight='bold', pad=15)
axes[0,1].invert_yaxis()
for i, v in enumerate(top10_brands['Total_Mentions']):
    axes[0,1].text(v + 100, i, f'{v:,}', va='center', fontsize=10)

# 1.3 Brand Sentiment Comparison (Top 10)
top10_for_sentiment = top10_brands.set_index('Brand')[['Positive', 'Negative', 'Neutral']]
top10_for_sentiment.plot(kind='barh', stacked=True, ax=axes[1,0], 
                         color=['#2ecc71', '#e74c3c', '#95a5a6'])
axes[1,0].set_xlabel('Number of Tweets', fontweight='bold')
axes[1,0].set_title('Sentiment Breakdown by Brand (Top 10)', fontsize=14, fontweight='bold', pad=15)
axes[1,0].legend(title='Sentiment', loc='lower right')

# 1.4 Brand Health Score (Top 15)
top15_health = brand_metrics_df.head(15).sort_values('Sentiment_Score')
colors_health = ['#2ecc71' if x > 0 else '#e74c3c' for x in top15_health['Sentiment_Score']]
axes[1,1].barh(range(len(top15_health)), top15_health['Sentiment_Score'], color=colors_health)
axes[1,1].set_yticks(range(len(top15_health)))
axes[1,1].set_yticklabels(top15_health['Brand'])
axes[1,1].set_xlabel('Sentiment Score (%)', fontweight='bold')
axes[1,1].set_title('Brand Sentiment Score (Positive% - Negative%)', fontsize=14, fontweight='bold', pad=15)
axes[1,1].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[1,1].invert_yaxis()

plt.tight_layout()
plt.savefig('twitter_sentiment_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: twitter_sentiment_dashboard.png")
plt.show()

# === VISUALIZATION 2: Category/Industry Analysis ===
# Group brands by industry/category
tech_brands = ['Microsoft', 'Google', 'Apple', 'Amazon', 'Nvidia']
gaming_brands = ['CallOfDuty', 'FIFA', 'CS-GO', 'Fortnite', 'ApexLegends', 'PUBG', 'NBA2K', 
                'Overwatch', 'GTA', 'RDR', 'Hearthstone', 'Dota2', 'LeagueOfLegends']
telecom_brands = ['Verizon', 'Comcast', 'ATT']
social_brands = ['Facebook', 'Twitter', 'Instagram']

def categorize_brand(brand):
    if brand in tech_brands:
        return 'Technology'
    elif brand in gaming_brands:
        return 'Gaming'
    elif brand in telecom_brands:
        return 'Telecom'
    elif brand in social_brands:
        return 'Social Media'
    else:
        return 'Other'

df['Category'] = df['Brand'].apply(categorize_brand)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Industry Sentiment Analysis', fontsize=16, fontweight='bold')

# Category distribution
category_counts = df['Category'].value_counts()
axes[0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0].set_title('Tweet Distribution by Industry', fontsize=13, fontweight='bold', pad=15)

# Sentiment by category
category_sentiment = pd.crosstab(df['Category'], df['Sentiment'], normalize='index') * 100
category_sentiment.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6'])
axes[1].set_xlabel('Industry Category', fontweight='bold')
axes[1].set_ylabel('Percentage (%)', fontweight='bold')
axes[1].set_title('Sentiment Distribution by Industry', fontsize=13, fontweight='bold', pad=15)
axes[1].legend(title='Sentiment')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('industry_sentiment_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: industry_sentiment_analysis.png")
plt.show()

# === KEY INSIGHTS REPORT ===
print("\n" + "="*80)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

# Top performing brands
top_positive = brand_metrics_df.nlargest(5, 'Sentiment_Score')
print("\n🌟 TOP 5 BRANDS BY SENTIMENT SCORE:")
for idx, row in top_positive.iterrows():
    print(f"  {row['Brand']:20s} | Score: {row['Sentiment_Score']:+6.2f}% | Mentions: {row['Total_Mentions']:,}")

# Brands needing attention
bottom_negative = brand_metrics_df.nsmallest(5, 'Sentiment_Score')
print("\n⚠️  BRANDS REQUIRING ATTENTION (Lowest Sentiment):")
for idx, row in bottom_negative.iterrows():
    print(f"  {row['Brand']:20s} | Score: {row['Sentiment_Score']:+6.2f}% | Mentions: {row['Total_Mentions']:,}")

# High engagement brands
high_engagement = brand_metrics_df.nlargest(5, 'Total_Mentions')
print("\n🔥 HIGHEST ENGAGEMENT BRANDS:")
for idx, row in high_engagement.iterrows():
    pos_rate = row['Positive_Ratio']
    neg_rate = row['Negative_Ratio']
    print(f"  {row['Brand']:20s} | {row['Total_Mentions']:6,} mentions | Positive: {pos_rate:.1f}% | Negative: {neg_rate:.1f}%")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. twitter_sentiment_dashboard.png - Main dashboard")
print("  2. industry_sentiment_analysis.png - Industry breakdown")
print("  3. brand_health_metrics.csv - Detailed metrics")
print("\n💡 Ready for presentation and reporting!")