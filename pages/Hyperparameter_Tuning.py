import streamlit as st
import pandas as pd

st.title("⚙️ Hyperparameter Tuning")
st.markdown("### 1. Tuning setup")
st.markdown("""
The optimization of the model was using the **Hugging Face Trainer + Optuna** integration. The objective was to maximize the model's **accuracy** while keeping training time manageable.

To reduce computational time during hyperparameter tuning, I used subsets of the original dataset—specifically, 100 samples for training and 25 for validation.
            
The tuned the hyperparameters were:
""")

# Section 2: Table of Tuned Parameters
st.markdown("### 🔍 Tuned Hyperparameters")
tuning_params = pd.DataFrame({
    "Hyperparameter": ["Learning Rate", "Batch Size", "Epochs"],
    "Search Range / Values": ["5e-5 to 5e-4 (log scale)", "[8, 12]", "1 to 2"],
    "Reason for Tuning": [
        "Controls model update step size. Critical for convergence.",
        "Smaller sizes help avoid OOM errors and overfitting.",
        "Fewer epochs reduce overfitting and speed up training."
    ]
})
st.table(tuning_params)

# Section 3: Summary of Optuna Use
st.markdown("### 🧠 Why Optuna?")
st.write("""
Optuna is a powerful hyperparameter optimization framework. It uses **Bayesian optimization (TPE)** to focus the search on promising configurations, making it more efficient than traditional grid search.

I used `trainer.hyperparameter_search()` from Hugging Face with Optuna under the hood.
- **Number of trials**: 5
- **Objective**: Maximize validation accuracy
- **Subset of dataset**: Used to reduce computation time
""")

st.markdown("### 2. Visualization and interpretation")

# Section 5: Best Trial Summary
st.markdown("### 🏆 Best Trial Summary")
st.success("""
- **Best Accuracy**: 100%
- **Best Parameters**:
    - Learning Rate: 4.18e-4  
    - Batch Size: 12  
    - Epochs: 2
""")
st.markdown("""Although the best run shows a 100% accuracy, this result is misleading. In reality, the model didn’t truly learn to generalize — the high accuracy is likely due to the small size of the dataset used during hyperparameter tuning, which led to overfitting on that limited subset.

The reason for using such a small dataset was due to computational limitations. In the initial trials, the full dataset caused my machine to timeout or crash, making it impossible to complete the training process. As a workaround, I reduced the training and evaluation datasets to a manageable size (100 training samples and 25 validation samples), just to allow the tuning process to run end-to-end.

While this approach helped me reach the best hyperparameters (learning rate, batch size, and epochs), the results cannot yet be trusted for real-world deployment — the model must be retrained and evaluated on the full dataset to confirm its actual performance.

""")

st.markdown("### Hyperparameter Search")
st.image("assets/search.png", caption="Optuna search trials and best accuracy", use_container_width=True)
