# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Load dataset
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# 2. Add labels
fake["label"] = 0
real["label"] = 1

# 3. Combine datasets
data = pd.concat([fake, real], axis=0)

# 4. Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# 5. Select text and labels
X = data["text"]
y = data["label"]

# 6. Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X)

# 7. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# 9. Train Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# 10. Predictions
lr_pred = lr_model.predict(X_test)
nb_pred = nb_model.predict(X_test)

# 11. Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))