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


print("\nMAP - Alpha = 0.5")

map_05 = {}

for c in classes:

    X_class = X[y == c]

    map_05[c] = map_estimation(
        X_class,
        0.5
    )


print("\nMAP - Alpha = 2")

map_2 = {}

for c in classes:

    X_class = X[y == c]

    map_2[c] = map_estimation(
        X_class,
        2
    )


first_class = classes[0]

print("\nFirst 10 words:")
print(words[:10])

print("\nMLE probabilities:")
print(mle[first_class][:10])

print("\nMAP probabilities - Alpha = 1:")
print(map_1[first_class][:10])

print("\nMAP probabilities - Alpha = 0.5:")
print(map_05[first_class][:10])

print("\nMAP probabilities - Alpha = 2:")
print(map_2[first_class][:10])

