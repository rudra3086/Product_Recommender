import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
import random
import glob
import os
from pathlib import Path
from html import escape

from huggingface_hub import hf_hub_download
from rapidfuzz import process,fuzz


st.set_page_config(
    page_title="Amazon Office Products",
        page_icon="📦",
        layout="wide"
)


##########################################################

CSS = """

<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.10), transparent 22%),
        linear-gradient(180deg, #f7f8fb 0%, #eef2f7 100%);
    color: #0f172a;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

div[data-testid="stToolbar"], #MainMenu, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 2.25rem;
    padding-bottom: 2.5rem;
    max-width: 1220px;
}

.hero {
    padding: 2rem 2rem 1.5rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(248, 250, 252, 0.82));
    box-shadow: 0 22px 60px rgba(15, 23, 42, 0.08);
    margin-bottom: 1.2rem;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.04);
    color: #475569;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 0.8rem 0 0.5rem;
    font-size: clamp(2.1rem, 3.2vw, 3.4rem);
    line-height: 1.02;
    color: #0f172a;
    font-weight: 800;
}

.hero p {
    margin: 0;
    max-width: 760px;
    color: #475569;
    font-size: 1rem;
    line-height: 1.7;
}

.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1rem;
}

.pill {
    display: inline-flex;
    align-items: center;
    padding: 0.42rem 0.78rem;
    border-radius: 999px;
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.08);
    color: #334155;
    font-size: 0.88rem;
    font-weight: 600;
}

.section-label {
    margin: 0.25rem 0 0.55rem;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
}

.search-card {
    padding: 1rem 1rem 0.9rem;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.05);
    margin-bottom: 1.2rem;
}

.card, .rec-card, .buy-card, .reason-card, .empty-card {
    border: 1px solid rgba(15, 23, 42, 0.08);
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}

.card {
    padding: 1.1rem 1.1rem 1rem;
    border-radius: 22px;
}

.product-title {
    margin: 0 0 0.75rem;
    font-size: 1.2rem;
    line-height: 1.35;
    color: #0f172a;
    font-weight: 700;
}

.product-image img, .buy-image img {
    border-radius: 18px;
    object-fit: cover;
}

.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0.7rem 0 0;
}

.meta-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.42rem 0.7rem;
    border-radius: 999px;
    background: #f8fafc;
    color: #334155;
    font-size: 0.86rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
}

.muted {
    color: #64748b;
    font-size: 0.92rem;
}

.card-grid {
    display: grid;
    gap: 0.75rem;
}

.rec-card {
    padding: 0.95rem 1rem;
    border-radius: 18px;
    margin-bottom: 0.75rem;
}

.rec-card h4, .buy-card h4 {
    margin: 0 0 0.35rem;
    font-size: 0.98rem;
    line-height: 1.45;
    color: #0f172a;
}

.rec-topline, .buy-topline {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: center;
    margin-bottom: 0.55rem;
}

.score {
    font-size: 0.78rem;
    font-weight: 700;
    color: #0f766e;
    background: rgba(15, 118, 110, 0.10);
    border-radius: 999px;
    padding: 0.3rem 0.55rem;
    white-space: nowrap;
}

.progress-bar {
    height: 8px;
    border-radius: 999px;
    background: #e2e8f0;
    overflow: hidden;
    margin: 0.7rem 0 0.55rem;
}

.progress-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #2563eb 0%, #14b8a6 100%);
}

.buy-card, .reason-card, .empty-card {
    padding: 0.95rem 1rem;
    border-radius: 18px;
}

.empty-card {
    margin-top: 0.35rem;
    color: #475569;
}

.reason-card {
    margin-bottom: 0.7rem;
    color: #0f172a;
}

div[data-baseweb="input"], div[data-baseweb="select"] {
    border-radius: 16px !important;
}

div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.96);
    border-color: rgba(15, 23, 42, 0.10) !important;
}

div[data-testid="stTextInput"] input {
    padding: 0.8rem 0.95rem;
    color: #0f172a !important;
    caret-color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    background-color: rgba(255, 255, 255, 0.98) !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}

div[data-testid="stSelectbox"] {
    margin-top: 0.25rem;
}

hr {
    margin: 1.3rem 0;
    border-color: rgba(15, 23, 42, 0.08);
}

</style>

"""

st.markdown(CSS,unsafe_allow_html=True)


##########################################################


@st.cache_resource
def load_artifacts():


    repo_id="24CS075/ProductRecommender"
    local_dir=Path("artifacts")


    def resolve_artifact(filename):
        local_path=local_dir / filename
        if local_path.exists():
            return str(local_path)
        return hf_hub_download(repo_id=repo_id, filename=filename)


    meta_filtered=pd.read_pickle(

        resolve_artifact("meta_filtered.pkl")

    )



    item_embeddings=np.load(

        resolve_artifact("item_embeddings.npy")

    )



    index=faiss.read_index(

        resolve_artifact("faiss.index")

    )




    with open(

        resolve_artifact("item2idx.pkl"),

        'rb'

    ) as f:


        item2idx=pickle.load(f)



    with open(

        resolve_artifact("idx2item.pkl"),

        'rb'

    ) as f:


        idx2item=pickle.load(f)




    with open(

        resolve_artifact("cf_top100.pkl"),

        'rb'

    ) as f:


        cf_top100=pickle.load(f)



    return(

        meta_filtered,

        item_embeddings,

        index,

        item2idx,

        idx2item,

        cf_top100

    )



(
meta_filtered,
item_embeddings,
index,
item2idx,
idx2item,
cf_top100
)=load_artifacts()


titles=meta_filtered.title.fillna('').tolist()



##########################################################


def search_products(query,topk=10):



    if not query:


        return []



    res=process.extract(

        query,

        titles,

        scorer=fuzz.token_set_ratio,

        limit=topk

    )



    out=[]



    for title,score,idx in res:



        out.append({


            'idx':idx,


            'title':title,


            'score':score


        })



    return out



##########################################################


def get_similar_products(


        query_idx,

        topk=10

):



    D,I=index.search(

        item_embeddings[query_idx:query_idx+1],

        k=topk+1

    )



    results=[]



    for score,idx in zip(

            D[0][1:],

            I[0][1:]):



        row=meta_filtered.iloc[idx]



        results.append({


            'idx':idx,


            'title':row['title'],


            'store':row['store'],


            'score':float(score)

        })




    return results



##########################################################


def get_customers_also_bought(


        query_idx,

        topk=5

):


    query_asin=idx2item[query_idx]



    results=[]



    if query_asin not in cf_top100:


        return results




    for asin,score in cf_top100[query_asin][:topk]:



        row=meta_filtered[

            meta_filtered.parent_asin==asin

        ]



        if len(row)==0:


            continue



        row=row.iloc[0]



        results.append({


            'title':row['title'],


            'store':row['store'],


            'score':score

        })



    return results



##########################################################


def explain_recommendation(

        query_idx,

        rec_idx

):



    q=meta_filtered.iloc[query_idx]

    r=meta_filtered.iloc[rec_idx]



    reasons=[]



    if str(q['store'])==str(r['store']):


        reasons.append("✓ Same Brand")




    qcat=set(

        q['categories']

    ) if isinstance(

        q['categories'],list

    ) else set()



    rcat=set(

        r['categories']

    ) if isinstance(

        r['categories'],list

    ) else set()



    if len(qcat&rcat):


        reasons.append(

            "✓ Same Category"

        )



    return reasons



##########################################################


images=glob.glob(

        "pictures/*.jpg"

)



def random_image():



    if len(images)==0:



        return "pictures/default.jpg"



    return random.choice(images)

##############################################################

def format_price(value):
    try:
        return f"${float(value):.2f}"
    except Exception:
        return str(value)


def render_card_title(title):
    return escape(str(title))


st.markdown(
    """
    <div class='hero'>
        <div class='eyebrow'>Product recommendations</div>
        <h1> Office Products Recommender</h1>
        <p>Search a product to compare the closest matches, browse customer-bought alternatives, and review the main signals behind each recommendation in a cleaner, decision-ready layout.</p>
        <div class='hero-meta'>
            <span class='pill'>SBERT embeddings</span>
            <span class='pill'>FAISS similarity search</span>
            <span class='pill'>Collaborative filtering</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='search-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Search</div>", unsafe_allow_html=True)


query = st.text_input(
        "Product search",
        placeholder="Search for items like OCR Pen, desk organizer, sticky notes",
        label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)


selected_idx=None


if query:


    results=search_products(query)


    if len(results):


        options={

            r['title']:r['idx']

            for r in results

        }



        selected=st.selectbox(
            "Select a matching product",
            list(options.keys()),
            label_visibility="collapsed"

        )


        selected_idx=options[selected]


if query and not len(results):

    st.markdown(
        "<div class='empty-card'>No products matched your search. Try a shorter or broader query.</div>",
        unsafe_allow_html=True,
    )


if selected_idx is not None:

    row=meta_filtered.iloc[selected_idx]

    similar=get_similar_products(

                selected_idx

    )

    bought=get_customers_also_bought(

                selected_idx

    )

    st.markdown("---")

    col1, col2 = st.columns([0.95, 1.25], gap="large")

    with col1:
        st.markdown("<div class='section-label'>Selected product</div>", unsafe_allow_html=True)
        st.image(random_image(), use_container_width=True)
        st.markdown(
            f"""
            <div class='card'>
                <h3 class='product-title'>{render_card_title(row['title'])}</h3>
                <div class='meta-row'>
                    <span class='meta-chip'>Brand: {escape(str(row['store']))}</span>
                    <span class='meta-chip'>Rating: {escape(str(row['average_rating']))}</span>
                    <span class='meta-chip'>Price: {escape(format_price(row['price']))}</span>
                </div>
                <p class='muted' style='margin-top:0.85rem;margin-bottom:0;'>This item is used as the anchor for similar-item and bought-together results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("<div class='section-label'>Similar products</div>", unsafe_allow_html=True)

        if similar:
            for rec in similar[:5]:
                score = max(0.0, min(float(rec['score']), 1.0))
                st.markdown(
                    f"""
                    <div class='rec-card'>
                        <div class='rec-topline'>
                            <h4>{render_card_title(rec['title'])}</h4>
                            <span class='score'>{score:.3f}</span>
                        </div>
                        <div class='muted'>Store: {escape(str(rec['store']))}</div>
                        <div class='progress-bar'><div class='progress-fill' style='width:{score * 100:.1f}%;'></div></div>
                        <div class='muted'>Similarity score</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='empty-card'>No close matches were returned for this product.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("<div class='section-label'>Customers also bought</div>", unsafe_allow_html=True)

        if bought:
            for item in bought:
                st.markdown(
                    f"""
                    <div class='buy-card'>
                        <div class='buy-topline'>
                            <div>
                                <h4>{render_card_title(item['title'])}</h4>
                                <div class='muted'>{escape(str(item['store']))}</div>
                            </div>
                            <span class='score'>{float(item['score']):.3f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='empty-card'>No customer-bought alternatives were available for this selection.</div>",
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown("<div class='section-label'>Why this was recommended</div>", unsafe_allow_html=True)

        if len(similar):
            reasons=explain_recommendation(
                        selected_idx,
                        similar[0]['idx']
            )

            if reasons:
                for r in reasons:
                    st.markdown(f"<div class='reason-card'>{escape(r)}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='empty-card'>The top match is similar by embedding distance, but no shared store or category metadata was strong enough to explain it further.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='empty-card'>Select a product to see the explanation panel.</div>",
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        "<div class='empty-card'>Start with a search above to load a selected product and its recommendations.</div>",
        unsafe_allow_html=True,
    )



##############################################################


st.markdown("---")


st.caption(

"Office Products Recommender | SBERT + FAISS + Collaborative Filtering"

)
