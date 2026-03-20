import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score
import time

seed = 42
test_share = 0.2

data = pd.read_csv("../data/banana_quality.csv")

X = data.iloc[:, 0:7]
y = data.iloc[:, 7]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_share,
    random_state=seed,
    shuffle=True
)

model = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    activation="relu",
    learning_rate_init=0.04,
    batch_size=32,
    max_iter=100,
    shuffle=True,
    random_state=42,
    early_stopping=False,
    tol=0,
    n_iter_no_change=10000,
    verbose=True
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

loss = log_loss(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)

print("Cross Entropy Loss:", loss)
print("Accuracy:", acc)
