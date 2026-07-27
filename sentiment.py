"""
sentiment.py
------------
Lightweight news sentiment scoring using NLTK's VADER lexicon. VADER is
tuned for short, informal text (like headlines/social posts) and needs
no model download beyond a small lexicon file, which makes it a good
fit for a dashboard that should start up quickly.

For higher accuracy on financial text specifically, swap this out for a
model fine-tuned on financial news (e.g. FinBERT) later -- the
`analyze_headlines` interface below would stay the same.
"""

import nltk


def _ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def analyze_sentiment(text: str):
    """Return (label, compound_score) for a piece of text."""
    _ensure_vader()
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text or "")
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, compound


def analyze_headlines(headlines):
    """Given a list of headline strings, return per-headline sentiment."""
    results = []
    for headline in headlines:
        label, score = analyze_sentiment(headline)
        results.append({"headline": headline, "label": label, "score": score})
    return results
