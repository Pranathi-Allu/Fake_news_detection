import pandas as pd
import re
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report

df_fake = pd.read_csv(r"C:\ML Projects\Fake News Detection\Fake.csv")
df_true = pd.read_csv(r"C:\ML Projects\Fake News Detection\True.csv")

df_fake["target"] = 0
df_true["target"] = 1

df_merged = pd.concat([df_fake,df_true],ignore_index=True)

#merging the title and text coloumn to a single coloumn "content"
df_merged["content"] = df_merged["title"] + " " + df_merged["text"] 

# suffling the rows so that not all fake and real are at one time
shuffled_df = df_merged.sample(frac=1).reset_index(drop=True)

# data prepocessing (cleaning) to input coloumn (content)
# 1. converting to lower case
shuffled_df["content"] = shuffled_df["content"].str.lower()

# 2. remove punctuation, digits and special characters
shuffled_df["content"] = shuffled_df["content"].apply(lambda x:re.sub(r'[^a-zA-Z\s]'," ",str(x)))

# 3. remove stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return " ".join(text.split())

shuffled_df["content"] = shuffled_df["content"].apply(clean_text)

# feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(shuffled_df["content"])

y = shuffled_df["target"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LogisticRegression(C=1.0, max_iter=500,class_weight="balanced")
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nconfusion matrix:")
print(confusion_matrix(y_test,y_pred))
print("\nclassification report:")
print(classification_report(y_test,y_pred))

# Save the model and vectorizer for deployment
joblib.dump(lr, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(shuffled_df["target"].value_counts())