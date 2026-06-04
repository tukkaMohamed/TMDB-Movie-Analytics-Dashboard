
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.plotting import scatter_matrix
import plotly.express as px

# ================= Page Config =================
st.set_page_config(
    page_title="Movie Ratings & Revenue Dashboard",
    layout="wide"
)

st.title("🎬 Movie Ratings & Revenue Analytics Dashboard")
st.markdown("Interactive dashboard for EDA, statistical insights, and PCA visualization.")

# ================= Load Data =================
df = pd.read_csv(r"tmdb_cleaned_movies.csv")

# ================= Feature Engineering =================
for col in ['budget', 'revenue']:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

df['profit'] = df['revenue'] - df['budget']
df['ROI'] = df['profit'] / df['budget']
df['ROI'].replace([np.inf, -np.inf], 0, inplace=True)
df['ROI'].fillna(0, inplace=True)

df['budget_log'] = np.log1p(df['budget'])
df['revenue_log'] = np.log1p(df['revenue'])

# ================= Sidebar =================
st.sidebar.header("Controls")

numeric_filter = st.sidebar.slider(
    "Minimum Vote Count",
    min_value=int(df['vote_count'].min()),
    max_value=int(df['vote_count'].max()),
    value=int(df['vote_count'].quantile(0.25))
)

df_filtered = df[df['vote_count'] >= numeric_filter]

# ================= Dataset Overview =================
st.header("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Movies", df_filtered.shape[0])
c2.metric("Average Rating", round(df_filtered['vote_average'].mean(), 2))
c3.metric("Average Revenue", f"${int(df_filtered['revenue'].mean()):,}")

st.dataframe(df_filtered.head())

# ================= Tabs =================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "EDA",
    "Correlation",
    "Genres & Revenue",
    "PCA",
    "Revenue Categories",
    "Revenue Treemap",
    "ROI by Genre"
])

# ================= TAB 1: EDA =================
with tab1:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df_filtered['vote_average'], bins=20, kde=True, ax=ax)
        ax.set_title("Distribution of Movie Ratings")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x='budget_log',
            y='revenue_log',
            data=df_filtered,
            alpha=0.6,
            ax=ax
        )
        ax.set_title("Budget vs Revenue (Log Scale)")
        st.pyplot(fig)

# ================= TAB 2: CORRELATION =================
with tab2:
    st.subheader("Correlation Heatmap")

    corr_features = [
        'budget_log', 'revenue_log', 'profit',
        'ROI', 'vote_average', 'vote_count',
        'popularity', 'runtime'
    ]

    corr_df = df_filtered[corr_features].select_dtypes(include=[np.number]).dropna()

    if corr_df.shape[0] > 0:
        corr = corr_df.corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.warning("Correlation could not be calculated because data became empty after cleaning.")

# ================= TAB 3: GENRES & REVENUE =================
with tab3:
    st.subheader("Genre & Revenue Insights")

    if 'genres' in df.columns:
        df['genres'] = (
            df['genres']
            .astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
            .str.replace("'", "", regex=False)
        )

        genre_counts = (
            df['genres']
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
            .head(15)
        )

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis", ax=ax)
            ax.set_title("Top 10 Genres (Count)")
            ax.set_xlabel("Movie Count")
            st.pyplot(fig)
    else:
        st.warning("No 'genres' column found.")

# ================= TAB 4: PCA =================
with tab4:
    st.subheader("PCA Visualization")

    pca_features = [
        'budget_log', 'revenue_log', 'profit',
        'ROI', 'vote_average', 'vote_count',
        'popularity', 'runtime'
    ]

    X = df_filtered[pca_features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

    fig, ax = plt.subplots()
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.6, ax=ax)
    ax.set_title("PCA Projection of Movies")
    st.pyplot(fig)

    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

# ================= TAB 5: Revenue Categories (Donut Chart) =================
with tab5:
    st.subheader("Revenue Category Distribution")

    df['revenue_cat'] = pd.cut(
        df['revenue'],
        bins=[0, 1e6, 5e7, 2e8, df['revenue'].max()],
        labels=["Low", "Medium", "High", "Blockbuster"]
    )

    rev_counts = df['revenue_cat'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(
        rev_counts,
        labels=rev_counts.index,
        wedgeprops=dict(width=0.4),
        autopct='%1.1f%%'
    )
    ax.set_title("Revenue Category Distribution")
    st.pyplot(fig)

# ================= TAB 6: Revenue Treemap =================
with tab6:
    st.subheader("Revenue Treemap")

    if 'genres' in df.columns:
        treemap_df = df[['genres', 'revenue']].copy()
        treemap_df['genres'] = treemap_df['genres'].astype(str).str.split(",").str[0]

        fig = px.treemap(
            treemap_df,
            path=['genres'],
            values='revenue',
            title="Revenue Contribution by Genre"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No 'genres' column found.")

# ================= TAB 7: ROI by Genre =================
with tab7:
    st.subheader("ROI Distribution by Genre")

    if 'genres' in df.columns:
        genre_roi = df[['ROI', 'genres']].copy()
        genre_roi['genres'] = genre_roi['genres'].astype(str).str.split(",").str[0]

        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(data=genre_roi, x='genres', y='ROI', ax=ax)
        ax.set_title("ROI Distribution Across Genres")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No 'genres' column found.")

# ================= Insights =================
st.header("💡 Key Insights")
st.markdown("""
- Movie ratings depend more on engagement (votes & popularity) than budget.
- Big budgets don't guarantee success — ROI varies heavily by genre.
- PCA clearly separates blockbusters vs normal-budget films.
""")
