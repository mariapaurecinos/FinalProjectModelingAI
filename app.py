import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F 

st.set_page_config(
    page_title="Fake News Detection App",
    layout="wide",
)

st.title("Fake News Detection App")
st.subheader("Write a news and get a prediction to verify if it is fake or real.")

# Input box
user_input = st.text_area("Text to predict:", "")

# Cargar modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained("modelo_guardado/")
model = AutoModelForSequenceClassification.from_pretrained("modelo_guardado/")

# Usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dummy output (esto lo reemplazarás luego con modelos reales)
if st.button("Predict"):
    # Tokenización
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(device)

    # Predicción
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)  # Convertir a probabilidades
        predicted_class_id = torch.argmax(logits, dim=1).item()
        confidence_score = probs[0][predicted_class_id].item()

    # Mapeo de clases
    label_map = {0: "Fake News", 1: "Real News"}
    predicted_label = label_map.get(predicted_class_id, "Desconocido")

    st.write(f"Predicción: {predicted_label} ({confidence_score*100:.2f}%)")

#hf_env\Scripts\activate  
#.\hf_env\Scripts\python.exe -m streamlit run app.py
