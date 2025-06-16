import streamlit as st

st.title("📊 Dataset Visualization")
st.markdown("### Exploratory Data Analysis")
st.write("""
The dataset presents two classes: **Fake News** and **Real News**.

- **Fake News**: Approximately 23,500 instances
- **Real News**: Approximately 21,500 instances

**Key Insight**:  
> The dataset is **slightly imbalanced**, with more fake news than real news. Although the imbalance is not severe, it's something to keep in mind when training classifiers, as it might slightly bias predictions towards the majority class.
""")
st.image("assets/class_distribution.png", caption="Class distribution", use_container_width=True)

#HISTOGRAM
st.write("""
This histogram shows the distribution of token lengths across the dataset.

- Most texts are relatively short, with the majority containing between **100 and 1000 tokens**.
- There's a steep drop-off beyond 1000 tokens, and very few samples exceed 2000 tokens.
- A small number of outliers have token counts above **5000**, indicating very long texts.

**Key Insight**:  
> The distribution is **right-skewed**, which is typical for natural language datasets.  
To ensure efficient training and reduce padding:
- It's recommended to consider truncating or setting a maximum sequence length around **600–800 tokens**, which covers most of the data.
- Also to use padding to align shorter sequences to this length during training.
        
**Reason behind Right Skewed Data**:
         
Zipf’s Law in language,
    This statistical pattern in natural language implies that a small number of elements (very long documents) appear much less frequently than short, common ones.
""")

st.image("assets/token_lengths.png", caption="Token lengths", use_container_width=True)

# WORD CLOUD
st.write("""
The word cloud provides a quick visual representation of the most frequent words across the dataset. Larger words appear more frequently in the text.

**Top frequent terms:**
- **said, say, according, called:** These are common in journalistic writing, indicating **reporting verbs** used to attribute statements or claims.
- **Donald Trump, Trump, United States, White House:** These reflect the dataset’s political nature and suggest a **strong focus on U.S. politics**, particularly around Donald Trump.
- **vote, people, country, government:** Terms related to **democracy, governance, and civic participation**.
- **will, make, want, don’t:** Action-oriented words that may reflect opinions, predictions, or quoted intentions.

**Key Insight:**
> The dataset likely consists of **political news articles**, especially from the Trump administration era, with a heavy emphasis on **quotations and statements** made by public figures.

""")

st.image("assets/word_cloud.png", caption="Word Cloud", use_container_width=True)

#NOISY TEXTS
st.markdown("### Examples of Noisy or Ambiguous Texts:")

st.markdown("### ❗️Example 1: Loaded Language & Bias")
st.write("""
> **“Stein referred to the judgement by the 9th Circuit Court as a *Coup d'état* against the executive branch…”**  
This example shows **highly charged emotional language** (e.g., *coup d'état*, *political puppets*), which may signal bias or propaganda. It’s also a mix of **opinion and fact**, making it hard to verify objectively.
""")

st.markdown("### 🌀 Example 2: Overloaded with Information & Attribution")
st.write("""
> **“Trump removed Steve Bannon from the National Security Council... Vice President Mike Pence said... Bannon said in a statement...”**  
This excerpt contains a **dense sequence of events and statements** from multiple figures, which can cause ambiguity in tracking **who said what** and **what's opinion vs. action**.
""")