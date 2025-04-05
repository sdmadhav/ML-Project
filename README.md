# ML-Project
Project -  Helping HR for employee Retentions Using Logistic Regression 

Dataset is downloaded from Kaggle. Link: https://www.kaggle.com/giripujar/hr-analytics

# Here's a breakdown of the libraries used in american sign language recognition code, along with their purposes:

**Data Handling and Visualization:**

*   **pandas:** Used for data manipulation and analysis, particularly for creating and working with DataFrames.
*   **numpy:** Fundamental package for numerical computation in Python.  Provides support for arrays, matrices, and mathematical functions.
*   **matplotlib.pyplot:**  A plotting library for creating static, interactive, and animated visualizations.
*   **plotly:** Used for creating interactive and visually appealing plots, charts, and dashboards. Specifically `plotly.express` and `plotly.io` are used.
*   **seaborn:**  (Although not explicitly in the code, based on the plotting style used, it's likely present).  Seaborn builds upon matplotlib to create more statistically informative and visually appealing plots.

**Machine Learning:**

*   **scikit-learn:** Used for various machine learning tasks, including `classification_report`, `confusion_matrix`, `precision_recall_curve`, `roc_curve`, `train_test_split`, and `compute_class_weight` suggesting that the code performs model evaluation and preprocessing.
*   **torch:** The core PyTorch library for deep learning, providing tensor computations, automatic differentiation, and neural network building blocks.
*   **torch.nn:**  Provides neural network modules and building blocks for constructing and training neural networks in PyTorch.
*   **torch.nn.functional:** Contains functions that don't have any parameters (like activation functions).
*   **torchmetrics:**  A library to compute various evaluation metrics for machine learning models.
*   **torchvision:**  Provides datasets, model architectures, and image transformations for computer vision tasks. `torchvision.datasets`, `torchvision.transforms`, and `torchvision.models` are used explicitly in the code.
* **mlflow:** Used for managing the machine learning lifecycle, including experiment tracking, model versioning, and deployment.
* **dagshub:** Used for experiment logging and version control.
*  **kagglehub:**  Used to interact with the Kaggle Datasets library.

**Other Utilities:**

*   **os:**  Provides functions for interacting with the operating system, such as file path manipulation and directory creation.
*   **pickle:**  Used for serializing and deserializing Python objects, allowing saving and loading of model states.
*   **typing:** Provides type hints for improved code readability and static analysis.
*   **urllib.parse:** For parsing URLs.
*   **icecream:**  For debugging, provides a convenient way to print variables with context.
*   **rich.progress:** Provides a nice progress bar.
*   **dataclasses:** For creating simple data classes.



**Specific Uses**

* `torch.utils.data` used for creating datasets and dataloaders

This list covers the major libraries.  Minor or specialized utilities might also be used within these libraries' functionality.
