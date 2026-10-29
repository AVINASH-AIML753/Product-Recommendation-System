
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Product Recommendation System")
st.write("Discover products similar to your interests.")

@st.cache_data
def load_data():
    df = pd.read_csv("products.csv")
    df["description"] = df["description"].fillna("")
    df["category"] = df["category"].fillna("")
    df["name"] = df["name"].fillna("")

    df["features"] = (
        df["name"] + " " +
        df["category"] + " " +
        df["description"]
    )
    return df

df = load_data()

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(df["features"])
similarity = cosine_similarity(matrix)

st.sidebar.header("Search & Filter")

search = st.sidebar.text_input("Search products")
categories = ["All"] + sorted(df["category"].unique().tolist())
category = st.sidebar.selectbox("Category", categories)

filtered = df.copy()

if search:
    mask = (
        filtered["name"] + " " +
        filtered["category"] + " " +
        filtered["description"]
    ).str.contains(search, case=False, regex=False)

    filtered = filtered[mask]

if category != "All":
    filtered = filtered[filtered["category"] == category]

st.subheader(f"Available Products ({len(filtered)})")

if filtered.empty:
    st.warning("No products found. Try another search.")
else:
    columns = st.columns(3)

    for i, (_, product) in enumerate(filtered.iterrows()):
        with columns[i % 3]:
            st.markdown(f"### {product['name']}")
            st.caption(product["category"])
            st.write(product["description"])
            st.write(f"**Price: ₹{product['price']:,.0f}**")

            if st.button(
                "Find Similar Products",
                key=f"recommend_{product['product_id']}"
            ):
                selected_index = product.name
                scores = list(enumerate(similarity[selected_index]))

                scores = sorted(
                    scores,
                    key=lambda x: x[1],
                    reverse=True
                )

                recommendations = [
                    (idx, score)
                    for idx, score in scores
                    if idx != selected_index
                ][:4]

                st.session_state["selected_product"] = product["name"]
                st.session_state["recommendations"] = recommendations

if "recommendations" in st.session_state:
    st.divider()
    st.subheader(
        f"Recommended Products for {st.session_state['selected_product']}"
    )

    recommendations = st.session_state["recommendations"]
    cols = st.columns(4)

    for i, (idx, score) in enumerate(recommendations):
        item = df.iloc[idx]

        with cols[i]:
            st.markdown(f"### {item['name']}")
            st.caption(item["category"])
            st.write(f"Price: ₹{item['price']:,.0f}")
            st.progress(float(score))
            st.caption(f"Similarity: {score * 100:.1f}%")

st.divider()
st.caption("Machine Learning: TF-IDF + Cosine Similarity")
