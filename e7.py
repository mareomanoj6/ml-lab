import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("20newsgroups.csv")

df = df[["text", "target_label"]].dropna()

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["text"])

words = vectorizer.get_feature_names_out()

y = df["target_label"].values

classes = np.unique(y)

print("\nMLE")

mle = {}

for c in classes:

    X_class = X[y == c]

    word_counts = np.asarray(
        X_class.sum(axis=0)
    ).flatten()

    probabilities = (
        word_counts / word_counts.sum()
    )

    mle[c] = probabilities

print("MLE estimation completed.")


def map_estimation(X_class, alpha):

    word_counts = np.asarray(
        X_class.sum(axis=0)
    ).flatten()

    probabilities = word_counts + alpha

    probabilities = (
        probabilities / probabilities.sum()
    )

    return probabilities


print("\nMAP - Alpha = 1")

map_1 = {}

for c in classes:

    X_class = X[y == c]

    map_1[c] = map_estimation(
        X_class,
        1
    )


print("\nMAP - Alpha = 0.1")

map_01 = {}

for c in classes:

    X_class = X[y == c]

    map_01[c] = map_estimation(
        X_class,
        0.1
    )


print("\nMAP - Alpha = 10")

map_10 = {}

for c in classes:

    X_class = X[y == c]

    map_10[c] = map_estimation(
        X_class,
        10
    )


def get_top_3(probs, words):
    top_indices = np.argsort(probs)[-3:][::-1]
    return [(words[i], probs[i]) for i in top_indices]

first_class = classes[0]
print(f"\nTop 3 words for class {first_class}:")

results = {
    "MLE": mle[first_class],
    "MAP (Alpha=1)": map_1[first_class],
    "MAP (Alpha=0.1)": map_01[first_class],
    "MAP (Alpha=10)": map_10[first_class],
}

for name, probs in results.items():
    print(f"\n{name}:")
    for word, prob in get_top_3(probs, words):
        print(f"{word}: {prob:.6f}")

