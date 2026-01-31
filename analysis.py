import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, axes

# Load the dataset
df = pd.read_csv('AI_Resume_Screening.csv')
print(df.head())
print(df.info())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates(subset=['Name'], keep='first')
print(df['Recruiter Decision'].value_counts())

Description : {'Resume_ID: Unique numerical identifier for each candidate record.', 
               'Name: Full name of the candidate.',
               'Skills: List of technical tools and competencies (e.g., Python, SQL).',
                'Experience (Years): Total number of years in the professional workforce.',
                'Education: Highest academic degree attained (e.g., B.Sc, PhD).',
                'Certifications: Professional credentials held (e.g., AWS, Google ML).',
                'Job Role: The specific position being applied for (e.g., AI Researcher).',
                'Recruiter Decision: Final screening outcome and target variable (Hire/Reject).',
                'Salary Expectation ($): Desired annual compensation in US Dollars.',
                'Projects Count: Number of professional or personal projects completed.',
                'AI Score (0-100): Automated score indicating how well the resume matches the job.'}


missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

df['Certifications'] = df['Certifications'].fillna('None')
cols_to_drop = ['Resume_ID', 'Name']

df = df.drop(columns=cols_to_drop)

# 4. Reset the index
df = df.reset_index(drop=True)

# 5. Save and view the result
print(df.head())
df.to_csv('Processed_AI_Resume_Data.csv', index=False)

# --- 1. Key Numerical and Categorical Variables ---
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numerical Features:", num_features)
cat_features = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical Features:", cat_features)

# 1. Visualization of Numerical Features
for col in num_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Distribution of {col}')
    plt.show()

# Outlier Treatment using IQR method
for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

# Cap the outliers
df[col] = np.clip(df[col], lower_bound, upper_bound)
print(f"Outliers in {col} treated using IQR method.")

# Visualization after Outlier Treatment
for col in num_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Distribution of {col} after Outlier Treatment')
    plt.show()

# --- 2. Overall distribution of shortlisted vs non-shortlisted ---
# Recruiter Decision: 'Hire' (Shortlisted) vs 'Reject' (Non-shortlisted)
decision_counts = df['Recruiter Decision'].value_counts()
decision_pct = df['Recruiter Decision'].value_counts(normalize=True) * 100

# 2. Visualization
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='Recruiter Decision', palette='viridis')
plt.title('Distribution of Recruiter Decisions')
plt.savefig('distribution_plot.png')

#Univariate Analysis

# 1. Plot Numerical Distributions
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('numerical_distributions.png')
plt.show()

# 2. Plot Categorical Frequencies
plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=df, x=col, palette='magma', order=df[col].value_counts().index)
    plt.title(f'Frequency of {col}')
    plt.xticks(rotation=15)
plt.show()

#3.Insights from individual feature analysis:
#The dataset is highly curated and balanced across roles and degrees, making it ideal for a general-purpose screening model, though its high AI Score skew confirms that the AI Score is the primary 'gatekeeper' for the final hiring decision.

#--- 3. Bivariate Analysis ---

# 1. AI Score vs Recruiter Decision (Boxplot)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Recruiter Decision', y='AI Score (0-100)', palette='Set2')
plt.savefig('ai_score_vs_decision.png')
plt.grid(True)
plt.show()
#There is a massive gap; hired candidates average 92.4, while rejected ones average only 47.3.

# 2. Experience vs Recruiter Decision (Mean Bar Plot)
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Recruiter Decision', y='Experience (Years)', palette='coolwarm')
plt.title('2. Avg Experience: Hire vs Reject', fontsize=14, fontweight='bold')
plt.savefig('experience_vs_decision.png')
plt.grid(True)
plt.show()
#Hired candidates have an average of 5.76 years, compared to just 1.17 years for rejected ones.

# 3. Education vs Recruiter Decision (Stacked Proportions)
plt.figure(figsize=(8, 6))
edu_pct = pd.crosstab(df['Education'], df['Recruiter Decision'], normalize='index') * 100
edu_pct.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], ax=plt.gca())
plt.title('3. Hiring Rate by Education Level', fontsize=14, fontweight='bold')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.savefig('education_vs_decision.png')
plt.grid(True)
plt.show()
#Candidates with PhDs have the highest hire rate at 85%, followed by Masters at 70%, and Bachelors at 55%.

# Additional Bivariate Plots
# 4. Number of Skills vs Decision (Bar Plot)

df['Skills_Count'] = df['Skills'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(data=df, x='Recruiter Decision', y='Skills_Count', palette='autumn', ax=ax1)
ax1.set_title('4. Skills Count vs Recruiter Decision', fontsize=14, fontweight='bold')
plt.grid(True)
plt.savefig('skills_vs_decision.png')
#Hired candidates have an average of 3.05 skills, while rejected ones average 2.87.

# 5. Projects Count vs Decision (Bar Plot)
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(data=df, x='Recruiter Decision', y='Projects Count', palette='winter', ax=ax2)
ax2.set_title('5. Projects Count vs Recruiter Decision', fontsize=14, fontweight='bold')
plt.grid(True)
plt.savefig('projects_vs_decision.png')
#Hired candidates average 5.65 projects, whereas rejected candidates average only 2.91.

# 6. Salary Expectation vs Decision (Boxplot)
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x='Recruiter Decision', y='Salary Expectation ($)', palette='spring', ax=ax3)
ax3.set_title('6. Salary Expectation vs Recruiter Decision', fontsize=14, fontweight='bold')
plt.savefig('salary_vs_decision.png')
plt.grid(True)
plt.show()
#The median salary expectation for hired candidates ($80,325) is nearly identical to rejected ones ($76,392).

# 7. AI Score across different Job Roles (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Job Role', y='AI Score (0-100)', palette='Set3')
plt.title('7. AI Score Distribution by Job Role', fontsize=14, fontweight='bold')
plt.xticks(rotation=15)
# Comment: AI Scores are consistent across roles, showing the screening tool is standardized.
plt.savefig('ai_score_by_role.png')
plt.grid(True)
plt.show()

# 8. Correlation between Experience and AI Score (Scatter Plot with Trend Line)
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Experience (Years)', y='AI Score (0-100)', 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('8. Correlation: Experience vs AI Score', fontsize=14, fontweight='bold')
# Comment: A slight positive correlation exists; more experienced candidates tend to score higher.
plt.savefig('experience_ai_correlation.png')
plt.grid(True)
plt.show()

# 9. Certifications among Shortlisted Candidates (Countplot)
plt.figure(figsize=(10, 6))
# Filtering for only hired candidates
hired_df = df[df['Recruiter Decision'] == 'Hire']
sns.countplot(data=hired_df, y='Certifications', palette='viridis', 
              order=hired_df['Certifications'].value_counts().index)
plt.title('9. Most Common Certifications (Hired Candidates)', fontsize=14, fontweight='bold')
# Comment: 'Deep Learning Specialization' and 'AWS Certified' are the top credentials for hires.
plt.savefig('certifications_hired.png')
plt.grid(True)
plt.show()

# 10. Education Level vs AI Score (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Education', y='AI Score (0-100)', palette='pastel')
plt.title('10. Impact of Education on AI Score', fontsize=14, fontweight='bold')
# Comment: AI Scores are generally high across all levels, showing that a degree 
# doesn't necessarily guarantee a higher automated score than skills do.
plt.savefig('edu_vs_aiscore.png')
plt.grid(True)
plt.show()

# 11. Experience across Job Roles (Boxplot) 
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Job Role', y='Experience (Years)', palette='cool')
plt.title('11. Experience Distribution by Job Role', fontsize=14, fontweight='bold')
plt.xticks(rotation=15)
# Comment: Most roles have a broad range of experience (0-10 years), 
# suggesting these positions are open to various seniority levels.
plt.savefig('exp_by_role.png')
plt.grid(True)
plt.show()

#12 . Projects Count by Recruiter Decision (Violin Plot) 
plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x='Recruiter Decision', y='Projects Count', palette='muted', inner="quart")
plt.title('13. Projects Distribution: Hired vs Rejected', fontsize=14, fontweight='bold')
# Comment: Hired candidates have a much wider distribution and higher median project count.
plt.savefig('projects_violin.png')
plt.grid(True)  
plt.show()

# 13. Numerical Correlation with Target Variable
# Convert Target to numeric for correlation calculation
df_corr = df.copy()
df_corr['Target'] = df_corr['Recruiter Decision'].map({'Hire': 1, 'Reject': 0})

# Select only numerical columns for correlation
numerical_cols = df_corr.select_dtypes(include=['number']).columns
correlation_matrix = df_corr[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['Target']].sort_values(by='Target', ascending=False), 
            annot=True, cmap='coolwarm', center=0)
plt.title('14. Feature Correlation with Shortlisting', fontsize=14, fontweight='bold')
plt.savefig('feature_correlation.png')
plt.grid(True)
plt.show()

#15. Overall Patterns (Pairplot)
# Visualizing the interaction between the top 3 predictors
sns.pairplot(df, vars=['AI Score (0-100)', 'Experience (Years)', 'Projects Count'], 
             hue='Recruiter Decision', palette='husl', diag_kind='kde')
plt.suptitle('15. Overall Patterns for Shortlisting Prediction', y=1.02, fontsize=16)
plt.savefig('overall_patterns.png')
plt.grid(True)
plt.show()