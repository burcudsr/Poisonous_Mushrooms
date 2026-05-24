# 🍄 Mushroom Classification

This project focuses on the binary classification of mushrooms as edible or poisonous, developed during the **Kaggle Playground Series - Season 4, Episode 8**. The model leverages physical and environmental characteristics to accurately categorize mushroom species.

### 🚀 Live Demo
Explore the interactive prediction model here: https://huggingface.co/spaces/bdaser/Mushroom

### 📊 Dataset & Preprocessing
The dataset provides a rich set of categorical features representing mushroom attributes. Key preprocessing steps include:
* **Categorical Encoding**: To convert string-based categorical features into a numerical format, `OrdinalEncoder` was utilized with `handle_unknown='use_encoded_value'` and `unknown_value=-1` to ensure robust handling of unseen data during inference.
* **Missing Value Imputation**: All missing values were identified and systematically addressed to ensure the model maintains high predictive integrity.
* **Data Integrity**: The pipeline was designed to handle high-cardinality categorical features effectively, ensuring the model remains accurate even with complex input variations.

### 🧠 Model Architecture & Methodology
The model utilizes a custom Keras architecture optimized for tabular data:
* **Entity Embeddings**: Categorical features are mapped into dense vectors using `Embedding` layers, allowing the model to capture semantic relationships between mushroom attributes.
* **Hybrid Processing**: Embedding outputs and numerical features are concatenated and processed through multiple **Dense layers (256, 128, 64 units)** with ReLU activation.
* **Regularization**: `Dropout(0.3)` is implemented to prevent overfitting.
