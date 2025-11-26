
import streamlit as st
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Health Diagnosis", layout="centered")

@st.cache_resource
def load_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_pickle_model(path="model/transformer_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_embeddings(path="model/embeddings.npy"):
    return np.load(path)

transformer_model = load_transformer_model()
model_data = load_pickle_model()
embeddings = load_embeddings()

disease_names = model_data['disease_names']
source_urls = model_data['source_urls']
sentences = model_data['sentences']

def predict_disease(user_input):
    user_embedding = transformer_model.encode([user_input])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    top_idx = int(np.argmax(similarities))
    top_score = float(similarities[top_idx])
    return disease_names[top_idx], source_urls[top_idx], sentences[top_idx], top_score

st.title("AI Health Diagnosis Assistant")
st.markdown("Describe your symptoms and get a likely match from the knowledge base.")

with st.form("symptom_form"):
    name = st.text_input("Full Name", "")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Gender", ["", "Male", "Female", "Prefer not to say"])
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=70.0)
    symptoms = st.text_area("Describe your symptoms", placeholder="e.g., sore throat, mild fever, headache...", height=140)
    submitted = st.form_submit_button("Analyze Symptoms")

if submitted:
    if not symptoms.strip():
        st.error("Please enter your symptoms.")
    else:
        with st.spinner("Analyzing..."):
            disease, url, matched_sentence, score = predict_disease(symptoms)
        st.success("Diagnosis Result")
        st.markdown(f"**Patient:** {name if name else 'Anonymous'}  •  **Age:** {int(age)}  •  **Gender:** {gender if gender else 'N/A'}  •  **Weight:** {weight} kg")
        st.markdown("---")
        st.markdown(f"**Predicted Disease:**  `{disease}`")
        st.markdown(f"**Matched Sentence:**  {matched_sentence}")
        st.markdown(f"**Confidence Score:**  {score:.3f}")
        if url:
            st.markdown(f"**Source:**  [{url}]({url})")
        st.info("This is an AI-assisted suggestion based on textual matching. Not a substitute for professional medical diagnosis.")
