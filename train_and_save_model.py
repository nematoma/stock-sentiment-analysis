import nltk
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

train_data = [
    ("Stock prices surged after the earnings report.", "pos"),
    ("Shares dipped due to disappointing forecasts.", "neg"),
    ("Tech stocks rallied despite market uncertainty.", "pos"),
    ("The CEO resigned amid accounting scandals.", "neg"),
    ("Revenue beat expectations, driving the stock up.", "pos"),
    ("Layoffs announced due to declining sales.", "neg"),
    ("Investors show renewed confidence in the company.", "pos"),
    ("Poor economic data spooked investors.", "neg"),
    ("Company expands operations overseas.", "pos"),
    ("Regulatory hurdles stall product launch.", "neg"),
    ("Dividend increased for third straight quarter.", "pos"),
    ("New tax law causes financial sector pullback.", "neg"),
    ("Strong demand drives growth in mobile sales.", "pos"),
    ("Production halted due to safety concerns.", "neg"),
    ("Company announces strategic partnership.", "pos"),
    ("Lawsuit filed over data breach.", "neg"),
    ("Improved margins fuel optimistic outlook.", "pos"),
    ("Higher interest rates weigh on bank profits.", "neg"),
    ("Mergers and acquisitions activity heats up.", "pos"),
    ("Supply chain disruptions impact delivery timelines.", "neg"),
    ("New product praised by early adopters.", "pos"),
    ("Executives under investigation for fraud.", "neg"),
    ("Customer base expanded through subscriptions.", "pos"),
    ("Fines imposed for environmental violations.", "neg"),
    ("New CEO promises culture shift.", "pos"),
    ("Sluggish consumer spending affects revenue.", "neg"),
    ("Tech giant hits new all-time high.", "pos"),
    ("Missed earnings target leads to sell-off.", "neg"),
    ("Company sees strong demand for services.", "pos"),
    ("Hack causes shutdown of main operations.", "neg"),
    ("Financials remain strong despite inflation.", "pos"),
    ("Quarterly loss stuns shareholders.", "neg"),
    ("Investors applaud new growth strategy.", "pos"),
    ("Management issues profit warning.", "neg"),
    ("Major contract secured with government.", "pos"),
    ("Analysts downgrade stock on weak guidance.", "neg"),
    ("Productivity increases across business units.", "pos"),
    ("Foreign exchange losses hit profits.", "neg"),
    ("Stock jumps on acquisition rumors.", "pos"),
    ("Operations suspended due to regulatory probe.", "neg"),
    ("Company hits key development milestone.", "pos"),
    ("Earnings decline for second consecutive quarter.", "neg"),
    ("Sales rebound after initial slump.", "pos"),
    ("Debt burden raises solvency concerns.", "neg"),
    ("Board approves share buyback program.", "pos"),
    ("Stock tumbles amid leadership turmoil.", "neg"),
    ("Retailer posts better-than-expected results.", "pos"),
    ("Cyberattack affects customer confidence.", "neg"),
    ("Outlook lifted on improving fundamentals.", "pos"),
    ("Asset write-down leads to quarterly loss.", "neg"),
    ("Cloud business sees accelerating growth.", "pos"),
]
train_data.extend([
    ("Company reports record sales but shares fall on weak outlook.", "neg"),
    ("New CFO joins amid restructuring efforts.", "neg"),
    ("Company sees growth, but stock slumps.", "neg"),
    ("Investors unsure despite solid fundamentals.", "neg"),
    ("Stock jumps briefly, then retreats on inflation fears.", "neg"),
    ("Company expands, but faces rising debt.", "neg"),
    ("New product delayed due to supply issues.", "neg"),
    ("Partnership announced, but market reaction muted.", "neg"),
    ("Lawsuit filed even as profits rise.", "neg"),
])

X_train_raw, y_train = zip(*train_data)
X_train = [preprocess(x) for x in X_train_raw]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", MultinomialNB()),
])
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "fear_greed_model.joblib")
print("✅ Model saved to fear_greed_model.joblib")