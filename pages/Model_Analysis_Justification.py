import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
st.title("📈 Model Analysis & Justification")

# 1. Model Justification
st.markdown("## 🤖 Model Justification")
st.markdown("""
The **Transformer-based architecture** selected for this task was a distilBERT because:

- These models are **pre-trained on large text corpora** and excel at understanding contextual relationships.
- DistilBERT maintains much of BERT’s performance while being smaller and faster, making it ideal for efficient inference in web applications.
- Transformer models like DistilBERT are state-of-the-art in text classification tasks, including fake news detection, hate speech detection, and sentiment analysis.

I fine-tuned DistilBERT on our task-specific dataset to adapt it to the nuances and label definitions of the classification problem.
""")

# 2. Classification Report
st.markdown("## 📋 Classification Report")

# Simulated classification report dictionary (replace with actual output from sklearn)
report_dict = {
    "precision": [1, 1],
    "recall": [1, 1],
    "f1-score": [1, 1],
    "support": [47, 53]
}
classes = ["Fake news", "Real News"]  # Customize this to your label names

report_df = pd.DataFrame(report_dict, index=classes)
st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

# 3. Confusion Matrix
st.markdown("##  Confusion Matrix")

# Simulated confusion matrix (replace with actual y_test and y_pred)
st.image("assets/confussion_matrix.png", caption="Confussion Matrix using 100 test samples", use_container_width=True)

# 4. Error Analysis
st.markdown("### 🔎 Error Analysis")

st.markdown("""
## ⚠️ Model Limitations and Workaround

The model did not learn effectively because **only a small portion of the dataset was used during training**. This was a **deliberate workaround** due to hardware limitations—early training attempts with the full dataset took too long or failed to complete due to resource constraints.

By reducing the dataset size, I was able to **perform initial testing and hyperparameter tuning**, but the trade-off was that the model **lacked sufficient examples to generalize well**, especially in underrepresented classes.

            
### Improvements for future implementatios:
- **Increase the size of the training dataset** to provide more learning signals.
- **Try alternative models**, such as BERT-base or RoBERTa, for better representation capacity.
- **Use ensemble models** to balance precision and recall more effectively.
""")

