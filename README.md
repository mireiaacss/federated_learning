# Federated Learning — Bank Loan Prediction

This project implements a privacy-preserving Machine Learning system to predict loan eligibility across three independent banking nodes. The system is designed to train a robust global model without ever sharing raw customer data between entities, ensuring data sovereignty and regulatory compliance.


## Core Features

- **Distributed Training (FedAvg):** Implements the Federated Averaging algorithm where banks only share model weights, never customer records.
- **Handling Imbalance:** Uses **SMOTE** (Synthetic Minority Oversampling Technique) locally at each node to handle the 88/12 class imbalance typical in financial datasets.
- **Threshold Optimization:** Automatically calculates the optimal decision threshold to maximize the **F1 Score**, balancing precision and recall for high-risk predictions.
- **High-Performance Ranking:** Achieves a **ROC-AUC of ~0.90**, demonstrating superior ability to distinguish between eligible and non-eligible customers regardless of the decentralized nature of the data.


## Project Structure

- `federated_learning.py`: Main training script including data processing, FedAvg implementation, and baseline comparison.
- `dashboard.py`: Streamlit-based MLOps dashboard to visualize model performance and data analysis.
- `eda_analysis.csv`: Statistical summaries of the dataset generated during training.
- `round_log.csv`: Convergence metrics for each communication round used for tracking training progress.
- `results_summary.csv`: Final comparative evaluation between the Federated and Centralized models.
- `confusion_matrices.csv`: Stores counts of True Positives, True Negatives, False Positives, and False Negatives for both models
- `bank_metrics.csv`: F1, AUC, Precision, and Recall for each individual bank, showing how the global model performs


## Setup and Usage

**1. Install dependencies:**

```bash
pip install numpy pandas scikit-learn imbalanced-learn ucimlrepo streamlit plotly
```
**2. Run the training pipeline:**

```bash
python federated_learning.py
```
This script fetches the UCI Bank Marketing dataset, performs the federated training, and generates the necessary telemetry files for the dashboard.

**3. Launch the dashboard:**

```bash
streamlit run dashboard.py
```
## Metric choice
In financial risk modeling, Accuracy is often a misleading metric because 88% of customers are typically "not eligible." A model could achieve 88% accuracy by simply predicting "No" for everyone.

ROC-AUC (~0.90): This is the primary indicator of the model's quality. It confirms that the model's internal probability ranking is highly accurate, effectively separating "good" from "bad" risks.

F1 Score: Used to select the Optimal Threshold (e.g., 0.65). By adjusting the threshold, we fine-tune the model to catch as many eligible customers as possible (Recall) while maintaining a reliable rate of true positives (Precision).

## How it Works
1. Local Preprocessing: Each bank scales its data and applies SMOTE to balance its training set locally.

2. Weight Aggregation: Banks perform local gradient descent and send only the resulting model parameters to a central server.

3. Federated Averaging: The server computes a weighted average of the parameters and redistributes the updated global model back to the nodes.

4. Privacy First: Raw data never leaves the local environment. The central server only ever processes arrays of floating-point numbers, never individual customer records.