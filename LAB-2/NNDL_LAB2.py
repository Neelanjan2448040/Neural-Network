import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

# ------------------ Activation Functions ------------------
def binary_step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def binary_sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except OverflowError:
        return 0 if x < 0 else 1

def bipolar_sigmoid(x):
    try:
        return (2 / (1 + np.exp(-x))) - 1
    except OverflowError:
        return -1 if x < 0 else 1

def ramp(x):
    if x < -1:
        return 0
    elif x > 1:
        return 1
    else:
        return (x + 1) / 2

activation_functions = {
    'binary_step': binary_step,
    'bipolar_step': bipolar_step,
    'binary_sigmoid': binary_sigmoid,
    'bipolar_sigmoid': bipolar_sigmoid,
    'ramp': ramp
}

# ------------------ Base Neural Network Class ------------------
class BaseModel:
    def __init__(self, input_dim, lr=0.01, epochs=100, activation='binary_step'):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros(input_dim)
        self.bias = 0
        self.activation_name = activation
        self.activation = activation_functions[activation]

    def net_input(self, x):
        return np.dot(x, self.weights) + self.bias

    def predict(self, X):
        outputs = [self.activation(self.net_input(x)) for x in X]
        if self.activation_name in ['binary_sigmoid', 'bipolar_sigmoid', 'ramp']:
            return np.array([1 if out >= 0.5 else 0 for out in outputs])
        elif self.activation_name == 'bipolar_step':
            return np.array([1 if out == 1 else 0 for out in outputs])
        else:
            return np.array(outputs)

# ------------------ Perceptron ------------------
class Perceptron(BaseModel):
    def fit(self, X, y):
        logs = []
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                z = self.net_input(xi)
                y_pred = self.activation(z)
                error = target - y_pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
            preds = self.predict(X)
            acc = np.mean(preds == y)
            logs.append(f"Epoch {epoch + 1}: Accuracy = {acc:.4f}")
        return logs

# ------------------ ADALINE ------------------
class ADALINE(BaseModel):
    def fit(self, X, y):
        logs = []
        prev_mse = float('inf')
        for epoch in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = target - output
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_error += error ** 2
            mse = total_error / len(X)
            preds = self.predict(X)
            acc = np.mean(preds == y)
            logs.append(f"Epoch {epoch + 1}: MSE = {mse:.4f}, Accuracy = {acc:.4f}")
            if abs(prev_mse - mse) < 0.0001:
                logs.append(f"Stopped early at epoch {epoch + 1}: MSE stabilized")
                break
            prev_mse = mse
        return logs

# ------------------ MADALINE ------------------
class MADALINE:
    def __init__(self, input_dim, hidden_units=2, lr=0.01, epochs=100, activation='binary_step'):
        self.lr = lr
        self.epochs = epochs
        self.activation = activation_functions[activation]
        self.weights = np.random.randn(hidden_units, input_dim)
        self.bias = np.random.randn(hidden_units)
        self.out_weights = np.random.randn(hidden_units)
        self.out_bias = np.random.randn(1)

    def net_input(self, x):
        return np.dot(self.weights, x) + self.bias

    def predict(self, X):
        predictions = []
        for x in X:
            hidden_out = np.array([self.activation(net) for net in self.net_input(x)])
            output = np.dot(self.out_weights, hidden_out) + self.out_bias
            predictions.append(1 if output >= 0 else 0)
        return np.array(predictions)

    def fit(self, X, y):
        logs = []
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                hidden_out = np.array([self.activation(net) for net in self.net_input(xi)])
                output = np.dot(self.out_weights, hidden_out) + self.out_bias
                error = target - output
                self.out_weights += self.lr * error * hidden_out
                self.out_bias += self.lr * error
                for i in range(len(hidden_out)):
                    self.weights[i] += self.lr * error * self.out_weights[i] * xi
                    self.bias[i] += self.lr * error * self.out_weights[i]
            preds = self.predict(X)
            acc = np.mean(preds == y)
            logs.append(f"Epoch {epoch + 1}: Accuracy = {acc:.4f}")
        return logs

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Neural Network Simulator", layout="wide")
st.title("Neural Network Simulator (Perceptron, ADALINE, MADALINE)")

model_type = st.selectbox("Choose Model", ["Perceptron", "ADALINE", "MADALINE"])
activation = st.selectbox("Select Activation Function", list(activation_functions.keys()),
                          help="Choose from 5 supported activation functions")
lr = st.slider("Learning Rate", 0.001, 1.0, 0.01, 0.001)
epochs = st.slider("Number of Epochs", 10, 500, 100, 10)

upload = st.file_uploader("Upload Your Dataset (CSV)", type=["csv"])

if upload is not None:
    df = pd.read_csv(upload)
    st.write("Preview of Uploaded Data:", df.head())

    target_col = st.selectbox("Select Target Column", df.columns)
    X_raw = df.drop(columns=[target_col])
    y_raw = df[target_col]

    num_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    X = preprocessor.fit_transform(X_raw)
    y = y_raw.values
    if y.dtype == object:
        y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

else:
    data = load_breast_cancer()
    X = data.data[:, :3]
    y = np.where(data.target == 0, 0, 1)
    X = SimpleImputer(strategy='mean').fit_transform(X)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if st.button("Train and Evaluate"):
    if model_type.lower() == 'perceptron':
        model = Perceptron(input_dim=X_train.shape[1], lr=lr, epochs=epochs, activation=activation)
    elif model_type.lower() == 'adaline':
        model = ADALINE(input_dim=X_train.shape[1], lr=lr, epochs=epochs, activation=activation)
    else:
        model = MADALINE(input_dim=X_train.shape[1], lr=lr, epochs=epochs, activation=activation)

    logs = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    st.success(f"Test Accuracy ({model_type.upper()}): {acc:.4f}")

    with st.expander("Evaluation Metrics"):
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(3,2),dpi=120)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,cbar=False, linewidths=0.5, linecolor='gray', square=True )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix", fontsize=15)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

    with st.expander("Predictions"):
        result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.dataframe(result_df.head(10))

    with st.expander("Training Log"):
        st.code("\n".join(logs))