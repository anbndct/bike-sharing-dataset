import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Function to load data
def load_data():
    day_df = pd.read_csv('day.csv')  # Adjust path as necessary
    hour_df = pd.read_csv('hour.csv')  # Adjust path as necessary
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    return day_df, hour_df

# Streamlit UI
st.title("Bike Sharing Usage Analysis")
day_df, hour_df = load_data()  # Assigning both dataframes

# Filter Date Range
st.sidebar.header("Date Range Filter")
start_date = st.sidebar.date_input("Start Date", day_df['dteday'].min())
end_date = st.sidebar.date_input("End Date", day_df['dteday'].max())
filtered_day_df = day_df[(day_df['dteday'] >= pd.to_datetime(start_date)) & (day_df['dteday'] <= pd.to_datetime(end_date))]

# Extract day of the week and map it to start from Sunday
filtered_day_df['day_of_week'] = filtered_day_df['dteday'].dt.dayofweek
filtered_day_df['day_of_week'] = (filtered_day_df['day_of_week'] + 1) % 7  # Start from Sunday

# Plotting
st.header("Analysis Results")

# Visualization: Daily Bike Usage Trend Over Years
st.header("1. Daily Bike Usage Trend Over Years (Filtered by Date)")
sns.set_style("whitegrid") 
fig, ax = plt.subplots()
sns.lineplot(data=filtered_day_df, x='day_of_week', y='cnt', hue='yr', marker="o", ax=ax, errorbar=None)
ax.set(xlabel='Day of the Week', ylabel='Daily Bike Count',
       title='Daily Bike Usage Trend Over Years')
plt.xticks(ticks=np.arange(7), labels=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], rotation=45)
plt.legend(title='Year')
st.pyplot(fig)

# Visualization: Hourly Bike Usage Trend for Weekdays vs Weekends
st.header("2. Hourly Bike Usage Trend Difference for Weekdays and Weekends")
hour_df['day_of_week'] = hour_df['dteday'].dt.dayofweek
hour_df['is_weekend'] = hour_df['day_of_week'].isin([5, 6])  # Saturday and Sunday are considered weekends
hourly_trend = hour_df.groupby(['hr', 'is_weekend']).agg({'cnt': 'mean'}).reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=hourly_trend, x='hr', y='cnt', hue='is_weekend', marker="o", ax=ax)
ax.set_title('Hourly Bike Usage Trend for Weekdays vs Weekends')
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Average Hourly Bike Count')
ax.set_xticks(np.arange(0, 24))
ax.legend(title='Weekend', labels=['Weekday', 'Weekend'])
st.pyplot(fig)

# Clustering Analysis based on 'windspeed'
st.header("3. Clustering Analysis based on Windspeed")

# Select number of clusters
number_of_clusters = st.slider("Select Number of Clusters", 2, 10, 4)

# Select features for clustering
features = ['windspeed']  # Using only windspeed for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(filtered_day_df[features])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
clusters = kmeans.fit_predict(scaled_features)
filtered_day_df['cluster'] = clusters

# Visualize clustering results
fig, ax = plt.subplots()
sns.scatterplot(x=filtered_day_df['windspeed'], y=filtered_day_df['cnt'], hue=filtered_day_df['cluster'], palette='viridis', ax=ax)
ax.set_xlabel('Windspeed')
ax.set_ylabel('Count of Bikes')
ax.set_title('Clusters based on Windspeed')
st.pyplot(fig)