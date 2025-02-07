# =============================================================================
# Road Accident Analysis - Complete Code Example
# =============================================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For inline plotting in Jupyter Notebook (if using Jupyter)
%matplotlib inline

# =============================================================================
# 0. Load Dataset and Basic Preprocessing
# =============================================================================

# Load the dataset (update the file path as necessary)
df = pd.read_csv("road-accident-data.csv")

# Display initial information
print("First five rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# -----------------------------------------------------------------------------
# Convert Date and Time columns, if they exist
# -----------------------------------------------------------------------------
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.day_name()

if 'Time' in df.columns:
    # Assuming Time is in HH:MM format
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour

# =============================================================================
# 1. Frequency of Accidents Over Time
# =============================================================================

print("\n================== 1. Frequency of Accidents Over Time ==================")

# 1a. Total number of accidents
total_accidents = df.shape[0]
print("Total number of accidents recorded:", total_accidents)

# 1b. Distribution over Years, Months, Days, and Hours
# Accidents by Year
if 'Year' in df.columns:
    plt.figure(figsize=(10, 6))
    accidents_per_year = df['Year'].value_counts().sort_index()
    sns.barplot(x=accidents_per_year.index, y=accidents_per_year.values, palette='Blues_d')
    plt.title("Number of Accidents per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Accidents")
    plt.show()

# Accidents by Month
if 'Month' in df.columns:
    plt.figure(figsize=(10, 6))
    accidents_per_month = df['Month'].value_counts().sort_index()
    sns.barplot(x=accidents_per_month.index, y=accidents_per_month.values, palette='Greens_d')
    plt.title("Number of Accidents per Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Accidents")
    plt.show()

# Accidents by Day of the Week
if 'DayOfWeek' in df.columns:
    plt.figure(figsize=(10, 6))
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(x='DayOfWeek', data=df, order=days_order, palette='Purples_d')
    plt.title("Number of Accidents by Day of the Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Number of Accidents")
    plt.show()

# Accidents by Hour of the Day
if 'Hour' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Hour', data=df, palette='Oranges_d')
    plt.title("Number of Accidents by Hour of the Day")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Accidents")
    plt.show()

# 1c. Trends and Patterns (e.g., Daily time-series)
if 'Date' in df.columns:
    daily_accidents = df.groupby('Date').size()
    plt.figure(figsize=(14, 7))
    daily_accidents.plot(kind='line', color='navy')
    plt.title("Daily Accident Frequency Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Accidents")
    plt.show()

# =============================================================================
# 2. Geographical Distribution
# =============================================================================

print("\n================== 2. Geographical Distribution ==================")

# 2a. Locations with the highest frequency (using City/Intersection/Road_Segment)
if 'City' in df.columns:
    plt.figure(figsize=(12, 6))
    top_cities = df['City'].value_counts().head(10)  # Top 10 cities
    sns.barplot(x=top_cities.index, y=top_cities.values, palette='Reds_d')
    plt.title("Top 10 Cities with Highest Accident Frequency")
    plt.xlabel("City")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()

if 'Intersection' in df.columns:
    plt.figure(figsize=(12, 6))
    top_intersections = df['Intersection'].value_counts().head(10)
    sns.barplot(x=top_intersections.index, y=top_intersections.values, palette='Reds_d')
    plt.title("Top 10 Intersections with Highest Accident Frequency")
    plt.xlabel("Intersection")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()

# 2b. Accidents across various Regions or Zones
if 'Region' in df.columns:
    plt.figure(figsize=(12, 6))
    region_counts = df['Region'].value_counts()
    sns.barplot(x=region_counts.index, y=region_counts.values, palette='coolwarm')
    plt.title("Accident Distribution by Region/Zone")
    plt.xlabel("Region/Zone")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()

# 2c. Specific Hotspots using geographical coordinates
if ('Latitude' in df.columns) and ('Longitude' in df.columns):
    plt.figure(figsize=(10, 8))
    if 'Severity' in df.columns:
        sns.scatterplot(x='Longitude', y='Latitude', data=df, hue='Severity', palette='viridis', alpha=0.6)
    else:
        sns.scatterplot(x='Longitude', y='Latitude', data=df, color='blue', alpha=0.6)
    plt.title("Geographical Hotspots of Accidents")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Severity")
    plt.show()

# =============================================================================
# 3. Accident Severity Analysis
# =============================================================================

print("\n================== 3. Accident Severity Analysis ==================")

if 'Severity' in df.columns:
    # 3a. Distribution of Accident Severities
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Severity', data=df, order=df['Severity'].value_counts().index, palette='Set2')
    plt.title("Distribution of Accident Severities")
    plt.xlabel("Accident Severity")
    plt.ylabel("Count")
    plt.show()

    # 3b. Percentage of Fatal and Serious Accidents
    severity_counts = df['Severity'].value_counts()
    total_severity = severity_counts.sum()
    fatal_percentage = (severity_counts.get('Fatal', 0) / total_severity) * 100
    serious_percentage = (severity_counts.get('Serious', 0) / total_severity) * 100
    print("Percentage of Fatal Accidents: {:.2f}%".format(fatal_percentage))
    print("Percentage of Serious Accidents: {:.2f}%".format(serious_percentage))

    # 3c. Correlation: Convert Severity to a Numeric Value
    severity_mapping = {'Minor': 1, 'Serious': 2, 'Fatal': 3}
    df['Severity_Numeric'] = df['Severity'].map(severity_mapping)

    # Severity vs. Hour of Day
    if 'Hour' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Hour', y='Severity_Numeric', data=df, palette='Accent')
        plt.title("Accident Severity by Hour of the Day")
        plt.xlabel("Hour of the Day")
        plt.ylabel("Severity (Numeric)")
        plt.show()

    # Severity vs. Region
    if 'Region' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Region', y='Severity_Numeric', data=df, palette='Accent')
        plt.title("Accident Severity by Region")
        plt.xlabel("Region")
        plt.ylabel("Severity (Numeric)")
        plt.xticks(rotation=45)
        plt.show()

    # Additional Correlation Heatmap (e.g., Severity, Hour, Month)
    numeric_features = ['Severity_Numeric']
    if 'Hour' in df.columns:
        numeric_features.append('Hour')
    if 'Month' in df.columns:
        numeric_features.append('Month')
    if len(numeric_features) > 1:
        corr_matrix = df[numeric_features].corr()
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of Severity and Time Factors")
        plt.show()

# =============================================================================
# 4. Demographic Insights
# =============================================================================

print("\n================== 4. Demographic Insights ==================")

# 4a. Age and Gender Distributions
if 'Age' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
    plt.title("Distribution of Age of Individuals Involved in Accidents")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

if 'Gender' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Gender', data=df, palette='pastel')
    plt.title("Gender Distribution of Individuals Involved in Accidents")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()

# 4b. Which Age Groups Exhibit Higher Involvement?
if 'Age' in df.columns:
    # Define age groups (example bins)
    bins = [0, 18, 30, 45, 60, 100]
    labels = ['0-18', '19-30', '31-45', '46-60', '60+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    plt.figure(figsize=(10, 6))
    age_group_counts = df['Age_Group'].value_counts().sort_index()
    sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='magma')
    plt.title("Accident Frequency by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Accidents")
    plt.show()

# 4c. Significant Difference in Accident Involvement Between Genders?
if ('Gender' in df.columns) and ('Age' in df.columns):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Gender', y='Age', data=df, palette='Set3')
    plt.title("Age Distribution by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Age")
    plt.show()

# =============================================================================
# 5. Environmental and Road Conditions
# =============================================================================

print("\n================== 5. Environmental and Road Conditions ==================")

# 5a. Weather Conditions vs. Accident Occurrences
if 'Weather' in df.columns:
    plt.figure(figsize=(12, 6))
    weather_counts = df['Weather'].value_counts()
    sns.barplot(x=weather_counts.index, y=weather_counts.values, palette='cool')
    plt.title("Accident Frequency by Weather Condition")
    plt.xlabel("Weather Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()

# 5b. Distribution of Accidents Across Different Road Types
if 'Road_Type' in df.columns:
    plt.figure(figsize=(12, 6))
    road_counts = df['Road_Type'].value_counts()
    sns.barplot(x=road_counts.index, y=road_counts.values, palette='autumn')
    plt.title("Accident Frequency by Road Type")
    plt.xlabel("Road Type")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()

# 5c. Impact of Lighting Conditions on Accident Frequencies
if 'Lighting' in df.columns:
    plt.figure(figsize=(12, 6))
    lighting_counts = df['Lighting'].value_counts()
    sns.barplot(x=lighting_counts.index, y=lighting_counts.values, palette='winter')
    plt.title("Accident Frequency by Lighting Condition")
    plt.xlabel("Lighting Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()

# =============================================================================
# 6. Vehicle and Driver Information
# =============================================================================

print("\n================== 6. Vehicle and Driver Information ==================")

# 6a. Types of Vehicles Most Frequently Involved
if 'Vehicle_Type' in df.columns:
    plt.figure(figsize=(12, 6))
    vehicle_counts = df['Vehicle_Type'].value_counts()
    sns.barplot(x=vehicle_counts.index, y=vehicle_counts.values, palette='Spectral')
    plt.title("Accident Frequency by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()

# 6b. Relationship Between Vehicle Type and Accident Severity
if ('Vehicle_Type' in df.columns) and ('Severity' in df.columns):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Vehicle_Type', hue='Severity', data=df, palette='Accent')
    plt.title("Accident Severity by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# 6c. Driver Experience/Behavior vs. Accident Occurrences
if 'Driver_Experience' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Driver_Experience'], bins=20, kde=True, color='olive')
    plt.title("Distribution of Driver Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Frequency")
    plt.show()

if 'Speeding' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Speeding', data=df, palette='coolwarm')
    plt.title("Frequency of Accidents Involving Speeding")
    plt.xlabel("Speeding (Yes/No)")
    plt.ylabel("Number of Accidents")
    plt.show()

if 'Seatbelt_Usage' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Seatbelt_Usage', data=df, palette='coolwarm')
    plt.title("Frequency of Accidents by Seatbelt Usage")
    plt.xlabel("Seatbelt Usage (Yes/No)")
    plt.ylabel("Number of Accidents")
    plt.show()

# =============================================================================
# 7. Temporal Patterns
# =============================================================================

print("\n================== 7. Temporal Patterns ==================")

# 7a. Peak Times During the Day / Specific Days of the Week
if 'Hour' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Hour', data=df, palette='cubehelix')
    plt.title("Accident Frequency by Hour of the Day")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Accidents")
    plt.show()

if 'DayOfWeek' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='DayOfWeek', data=df, order=days_order, palette='cubehelix')
    plt.title("Accident Frequency by Day of the Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Number of Accidents")
    plt.show()

# 7b. Weekdays vs. Weekends
if 'DayOfWeek' in df.columns:
    df['Is_Weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Is_Weekend', data=df, palette='pastel')
    plt.title("Accident Frequency: Weekdays vs. Weekends")
    plt.xlabel("Is Weekend (True/False)")
    plt.ylabel("Number of Accidents")
    plt.show()

# 7c. Seasonal Variation in Accident Occurrences
if 'Month' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Month', data=df, palette='rainbow')
    plt.title("Accident Frequency by Month (Seasonal Variation)")
    plt.xlabel("Month")
    plt.ylabel("Number of Accidents")
    plt.show()

# =============================================================================
# 8. Contributing Factors
# =============================================================================

print("\n================== 8. Contributing Factors ==================")

if 'Contributing_Factors' in df.columns:
    # 8a. Most Common Contributing Factors
    plt.figure(figsize=(12, 6))
    factors_counts = df['Contributing_Factors'].value_counts().head(10)
    sns.barplot(x=factors_counts.index, y=factors_counts.values, palette='mako')
    plt.title("Top 10 Contributing Factors to Accidents")
    plt.xlabel("Contributing Factor")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    # 8b. Factors by Accident Severity
    if 'Severity' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Contributing_Factors', hue='Severity', data=df, palette='viridis')
        plt.title("Contributing Factors by Accident Severity")
        plt.xlabel("Contributing Factor")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # 8c. Factors in Specific Locations or Times (example: by Region)
    if 'Region' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Contributing_Factors', hue='Region', data=df, palette='Spectral')
        plt.title("Contributing Factors by Region")
        plt.xlabel("Contributing Factor")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

# =============================================================================
# 9. Injury and Fatality Analysis
# =============================================================================

print("\n================== 9. Injury and Fatality Analysis ==================")

# 9a. Distribution of Injuries and Fatalities Among Different Road Users
if ('Injury_Count' in df.columns) or ('Fatality_Count' in df.columns):
    # Assuming a column "Road_User" exists (e.g., Driver, Passenger, Pedestrian)
    if 'Road_User' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Road_User', data=df, palette='Set1')
        plt.title("Accident Frequency by Road User Type")
        plt.xlabel("Road User")
        plt.ylabel("Number of Accidents")
        plt.show()
    else:
        print("Column 'Road_User' not found; cannot analyze injury/fatality distribution by road user.")

# 9b. Correlation of Injury Severity with Vehicle Type or Speed
if ('Injury_Count' in df.columns) and ('Vehicle_Type' in df.columns):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Vehicle_Type', y='Injury_Count', data=df, palette='cool')
    plt.title("Injury Count by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Injury Count")
    plt.xticks(rotation=45)
    plt.show()

# 9c. Scenarios Leading to Higher Fatality Rates
if ('Fatality_Count' in df.columns) and ('Speeding' in df.columns):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Speeding', y='Fatality_Count', data=df, palette='Reds')
    plt.title("Fatality Count vs. Speeding")
    plt.xlabel("Speeding (Yes/No)")
    plt.ylabel("Fatality Count")
    plt.show()

# =============================================================================
# 10. Comparative Analysis
# =============================================================================

print("\n================== 10. Comparative Analysis ==================")

# 10a. Compare Accident Statistics Between Different Regions/Time Periods
if 'Region' in df.columns and 'Year' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Region', hue='Year', data=df, palette='tab10')
    plt.title("Accident Frequency by Region and Year")
    plt.xlabel("Region")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()

# 10b. Urban vs. Rural Differences (assuming a column 'Area_Type' exists)
if 'Area_Type' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Area_Type', data=df, palette='Set2')
    plt.title("Accident Frequency: Urban vs. Rural")
    plt.xlabel("Area Type")
    plt.ylabel("Number of Accidents")
    plt.show()
else:
    print("Column 'Area_Type' not found. If available, compare urban vs. rural accident characteristics.")

# 10c. Compare with External Data or National Statistics
# This section would require external data sources. As an example, you could overlay your data with external statistics.
print("Comparative analysis with external data requires additional datasets. Please incorporate such datasets as needed.")

# =============================================================================
# End of Analysis
# =============================================================================

print("\nAnalysis complete. Please review the plots and outputs for insights.")
