import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# Load or train model
def load_model():
    try:
        return joblib.load("sentiment_model.joblib")
    except:
        # Train here or raise error
        print("Train model first.")
        return None

def train_model(train_data):
    X, y = zip(*train_data)
    X = [preprocess(x) for x in X]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", MultinomialNB()),
    ])
    model.fit(X, y)
    joblib.dump(model, "sentiment_model.joblib")
    return model
