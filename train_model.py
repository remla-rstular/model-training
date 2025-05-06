import argparse
import sys

import joblib
import pandas as pd
from lib_ml.preprocess import TextPreprocessor
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DEFAULT_DATA_PATH = "data/a1_RestaurantReviews_HistoricDump.tsv"

# Parse the arguments
# We accept a single positional argument: the path to the TSV file
parser = argparse.ArgumentParser(
    description="Train the model from a TSV file containing restaurant reviews."
)
parser.add_argument(
    "data_path",
    type=str,
    default=DEFAULT_DATA_PATH,
    help=f"Path to the TSV file (default: {DEFAULT_DATA_PATH})",
    nargs="?",
)
parser.add_argument(
    "--output-model", "-o", type=str, default="sentiment_model.pkl", help="Path to save the trained model."
)
parser.add_argument(
    "-t",
    "--test-split",
    type=float,
    default=0.2,
    help="Proportion of the dataset to include in the test split.",
)
parser.add_argument(
    "-r",
    "--random-state",
    type=int,
    default=42,
    help="Random seed for shuffling the data before applying the split.",
)
args = parser.parse_args()

data_path = args.data_path
logger.debug(f"Using data path: {data_path}")

# Load the TSV
try:
    df = pd.read_csv(data_path, sep="\t")
except FileNotFoundError:
    logger.error(f"File not found: {data_path}. Please check the path.")
    sys.exit(1)
except pd.errors.ParserError:
    logger.exception(f"Error parsing the file: {data_path}. Please check the file format.")
    sys.exit(1)
except Exception as e:
    logger.exception(f"An unexpected error occurred: {e}")
    sys.exit(1)

logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from the TSV file.")

reviews = df["Review"]
ratings = df["Liked"]

if args.test_split == 0.0:
    X_train, X_test, y_train, y_test = reviews, [], ratings, []
else:
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, ratings, test_size=args.test_split, random_state=args.random_state
    )
logger.info(f"Split the data into {len(X_train)} training and {len(X_test)} testing samples.")

model = Pipeline(
    [
        ("text_preprocessor", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000, random_state=args.random_state)),
    ]
)
model.fit(X_train, y_train)
logger.info("Model training completed.")

if len(X_test) > 0:
    y_pred = model.predict(X_test)
    logger.info(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")

joblib.dump(model, args.output_model)
logger.info(f"Model saved to {args.output_model}.")
