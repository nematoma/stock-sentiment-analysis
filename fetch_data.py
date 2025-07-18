import os
import praw
import tweepy
from dotenv import load_dotenv
from newsapi import NewsApiClient

load_dotenv()

# Twitter
twitter_client = tweepy.Client(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))

def fetch_twitter_data(query, max_results=10):
    tweets = twitter_client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["lang"])
    return [tweet.text for tweet in tweets.data if tweet.lang == "en"] if tweets.data else []

# Reddit
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def fetch_reddit_data(subreddits, query, limit=10):
    results = []
    for sub in subreddits:
        for post in reddit.subreddit(sub).search(query, limit=limit):
            results.append(post.title + " " + post.selftext)
    return results

# NewsAPI
newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))

def fetch_news_data(query, category="business", n_news=10):
    articles = newsapi.get_top_headlines(q=query, language="en", page_size=n_news, category=category)
    return [a["title"] + " " + a["description"] for a in articles["articles"] if a["description"]]
