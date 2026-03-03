# CAP 5610 — Machine Learning Practice Project

This repository contains a collection of machine learning models implemented as part of my practice and experimentation while studying Advanced Machine Learning (CAP 5610).

The project includes:

- Logistic Regression (classification)
- Neural Network Regression
- Neural Network Classification (CIFAR-10)

Each task explores model design, evaluation, regularization strategies, and architectural tradeoffs across structured and image data.

---

# 📊 Task 1 — Logistic Regression (Adult Census Income)

In this task, I implemented a Logistic Regression classifier on the Adult Census dataset.

Main steps:

- Replaced missing values with median imputation
- One-hot encoded categorical variables
- Compared balanced vs unbalanced class weighting
- Interpreted beta coefficients and odds ratios
- Evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC

This task highlights how class imbalance affects decision boundaries and recall performance.

---

# 🏠 Task 2 — Neural Network Regressor (King County Housing)

I built a fully connected neural network to predict house prices.

Key components:

- Log transformation of the target variable to stabilize variance
- Feature standardization using StandardScaler
- Comparison between Sigmoid and ReLU activations
- EarlyStopping to prevent overfitting

This task demonstrates how activation functions and target transformations impact regression performance on tabular datasets.

---

# 🖼 Task 3 — Neural Network Classifier (CIFAR-10)

This task implements a flexible Multilayer Perceptron (MLP) classifier for CIFAR-10.

Experiments include:

- Architecture search (shallow-wide vs deeper configurations)
- Dropout and L2 regularization comparison
- Analysis of training/validation gaps
- Confusion matrix evaluation

Images are normalized to [0,1] and flattened inside the model forward pass into 3072-dimensional vectors.

Results illustrate how architectural constraints limit performance when spatial structure is not preserved.

---

# ⚙️ Environment Setup (VS Code Recommended)

It is recommended to create a virtual environment before running the notebooks.

### 1️⃣ Create virtual environment

python -m venv venv

### 2️⃣ Activate environment

Mac/Linux:
source venv/bin/activate

Windows:
venv\Scripts\activate

### 3️⃣ Install dependencies

pip install -r requirements.txt

Then open the project in VS Code and select the virtual environment as the Python interpreter.

---

# ☁️ Running Task 3 (Recommended: Google Colab)

Task 3 (CIFAR-10 MLP) involves heavier computation.

For best performance:

1. Open the notebook in Google Colab
2. Go to Runtime → Change runtime type
3. Set Hardware accelerator → GPU (T4 if available)

Using a GPU significantly reduces training time compared to CPU.

---

# 📂 Data

All datasets used in this project are included inside the `/data` directory.

CIFAR-10 is automatically downloaded via `torchvision.datasets.CIFAR10`.

---

# 🧰 Technologies Used

- Python
- PyTorch
- scikit-learn
- NumPy
- pandas
- Matplotlib
- Seaborn

---

# 📌 Notes

This repository reflects experimental exploration of model capacity, regularization, and architectural limitations across both structured and image data.
