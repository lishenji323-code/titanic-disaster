"""Titanic survival prediction with logistic regression.

Loads the Kaggle Titanic data from data/, explores and cleans it, trains a
logistic regression model, and predicts survival for the test set. Every
step prints what it did, so the full process is visible in the terminal.
"""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path('data')
RANDOM_STATE = 42


def section(title):
    """Print a section header so the output is easy to follow."""
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def load_csv(name):
    """Load a CSV from the data folder, with a clear error if missing."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Download the Kaggle Titanic data into '
            'src/data/ and mount it with -v (see README).')
    df = pd.read_csv(path)
    print(f'Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns')
    return df


# ----------------------------------------------------------------------
# Step 13: load packages and data
# ----------------------------------------------------------------------
section('STEP 13: Load the training data')
print(f'pandas version: {pd.__version__}')
train = load_csv('train.csv')
print('\nFirst 5 rows:')
print(train.head().to_string())

# ----------------------------------------------------------------------
# Step 14 (part 1): explore the data
# ----------------------------------------------------------------------
section('STEP 14a: Explore the training data')

print('Column types:')
print(train.dtypes.to_string())

print('\nMissing values per column:')
missing = train.isna().sum()
missing_pct = (missing / len(train) * 100).round(1)
print(pd.DataFrame({'missing': missing, 'percent': missing_pct})
      [missing > 0].to_string())

print(f'\nOverall survival rate: {train["Survived"].mean():.1%}')
for col in ['Sex', 'Pclass', 'Embarked']:
    rates = train.groupby(col)['Survived'].agg(['mean', 'count'])
    rates['mean'] = (rates['mean'] * 100).round(1)
    rates.columns = ['survival_%', 'passengers']
    print(f'\nSurvival rate by {col}:')
    print(rates.to_string())

# ----------------------------------------------------------------------
# Step 14 (part 2): clean the data and add features
# ----------------------------------------------------------------------
section('STEP 14b: Clean the data and add features')


def get_title(names):
    """Extract the title (Mr, Mrs, Miss, ...) from passenger names.

    Names look like 'Braund, Mr. Owen Harris', so the title is the word
    between the comma and the period. Rare titles are grouped together.
    """
    titles = names.str.extract(r',\s*([^.]+)\.', expand=False).str.strip()
    titles = titles.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    common = ['Mr', 'Mrs', 'Miss', 'Master']
    return titles.where(titles.isin(common), 'Rare')


def learn_fill_values(df):
    """Learn the values used to fill missing data, from training data only.

    Using only the training data avoids leaking information from the
    test set into the model.
    """
    titles = get_title(df['Name'])
    return {
        'age_by_title': df.groupby(titles)['Age'].median(),
        'age_overall': df['Age'].median(),
        'embarked': df['Embarked'].mode()[0],
        'fare': df['Fare'].median(),
    }


def prepare(df, fill, verbose=True):
    """Clean a raw Titanic DataFrame and add model features.

    Args:
        df: Raw train or test data.
        fill: Fill values from learn_fill_values(), computed on train.
        verbose: If True, print every change that is made.

    Returns:
        A new DataFrame with cleaned columns and added features.
    """
    df = df.copy()
    say = print if verbose else (lambda *args, **kwargs: None)

    # Title: a strong signal for age and sex/social status
    df['Title'] = get_title(df['Name'])
    say('Added Title (from Name):',
        df['Title'].value_counts().to_dict())

    # Age: fill missing values with the median age for that title,
    # e.g. a missing "Master" (young boy) gets a child's age
    n_age = df['Age'].isna().sum()
    df['Age'] = df['Age'].fillna(df['Title'].map(fill['age_by_title']))
    df['Age'] = df['Age'].fillna(fill['age_overall'])
    say(f'Filled {n_age} missing Age values with the median age '
        'for each title')

    # Embarked: only a few missing, so use the most common port
    n_emb = df['Embarked'].isna().sum()
    df['Embarked'] = df['Embarked'].fillna(fill['embarked'])
    say(f"Filled {n_emb} missing Embarked values with "
        f"'{fill['embarked']}' (most common port)")

    # Fare: fill any missing values with the median fare
    n_fare = df['Fare'].isna().sum()
    df['Fare'] = df['Fare'].fillna(fill['fare'])
    say(f'Filled {n_fare} missing Fare values with the median fare '
        f"({fill['fare']:.2f})")

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    say('Added FamilySize (SibSp + Parch + 1) and IsAlone '
        '(1 if traveling alone)')

    # Sex: convert text to a number
    df['IsFemale'] = (df['Sex'] == 'female').astype(int)
    say('Converted Sex to IsFemale (1 = female, 0 = male)')

    # Drop columns that are not used as features
    dropped = ['Name', 'Sex', 'Ticket', 'Cabin']
    df = df.drop(columns=dropped)
    say(f'Dropped columns: {dropped} (Cabin is 77% missing, Ticket is '
        'an ID-like code, and Name and Sex are replaced by new features)')

    # One-hot encode the categorical columns
    df = pd.get_dummies(df, columns=['Embarked', 'Title'], dtype=int)
    say('One-hot encoded Embarked and Title into 0/1 columns')
    return df


fill_values = learn_fill_values(train)
train_clean = prepare(train, fill_values)

print(f'\nCleaned training data: {train_clean.shape[0]} rows, '
      f'{train_clean.shape[1]} columns')
print(f'Missing values remaining: {train_clean.isna().sum().sum()}')
print('\nFirst 5 rows after cleaning:')
print(train_clean.head().to_string())

# ----------------------------------------------------------------------
# Step 15: build a logistic regression model
# ----------------------------------------------------------------------
section('STEP 15: Build a logistic regression model')

# Use every cleaned column except the ID and the answer
FEATURES = [c for c in train_clean.columns
            if c not in ('PassengerId', 'Survived')]
X = train_clean[FEATURES]
y = train_clean['Survived']
print(f'Using {len(FEATURES)} features: {FEATURES}')

# Scale features to a common range, then fit logistic regression.
# Scaling helps the model converge and makes coefficients comparable.
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000))

# Hold out 20% of the training data to check the model on passengers it
# has not seen, since test.csv has no Survived column to check against.
X_fit, X_val, y_fit, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
model.fit(X_fit, y_fit)
val_acc = accuracy_score(y_val, model.predict(X_val))
print(f'\nHeld-out check: trained on {len(X_fit)} passengers, '
      f'tested on {len(X_val)} unseen passengers')
print(f'Held-out (validation) accuracy: {val_acc:.1%}')

# Refit on all 891 training passengers for the final model
model.fit(X, y)
print('\nRefit the final model on all training passengers')

coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
print('\nModel coefficients (positive = higher chance of survival):')
print(coefs.round(3).to_string())

# ----------------------------------------------------------------------
# Step 16: accuracy on the training set
# ----------------------------------------------------------------------
section('STEP 16: Accuracy on the training set')
train_acc = accuracy_score(y, model.predict(X))
print(f'Training accuracy: {train_acc:.1%}')
print(f'Baseline (predict everyone died): {1 - y.mean():.1%}')

# ----------------------------------------------------------------------
# Step 17: load test.csv and predict
# ----------------------------------------------------------------------
section('STEP 17: Load test.csv and predict survival')
test = load_csv('test.csv')

# Clean the test data with the SAME steps and the fill values learned
# from the training data
print('\nApplying the same cleaning steps to the test data:')
test_clean = prepare(test, fill_values)

# Make sure the test data has exactly the training feature columns, in
# the same order. A category missing from the test data (e.g. no
# passenger with a given title) gets a column of 0s.
X_test = test_clean.reindex(columns=FEATURES, fill_value=0)
print(f'\nTest features: {X_test.shape[0]} rows, {X_test.shape[1]} columns')

test_pred = model.predict(X_test)
predictions = pd.DataFrame({'PassengerId': test['PassengerId'],
                            'Survived': test_pred})
print(f'Predicted survival rate on test set: {test_pred.mean():.1%}')
print('\nFirst 10 predictions:')
print(predictions.head(10).to_string(index=False))

out_path = DATA_DIR / 'predictions_python.csv'
predictions.to_csv(out_path, index=False)
print(f'\nSaved all {len(predictions)} predictions to src/{out_path}')

# ----------------------------------------------------------------------
# Step 18: accuracy on the test set
# ----------------------------------------------------------------------
section('STEP 18: Accuracy on the test set')
if 'Survived' in test.columns:
    test_acc = accuracy_score(test['Survived'], test_pred)
    print(f'Test accuracy: {test_acc:.1%}')
else:
    print('test.csv has no Survived column: Kaggle keeps the true answers '
          'hidden, so test accuracy cannot be computed locally.')
    print(f'Best local estimate (held-out accuracy from step 15): '
          f'{val_acc:.1%}')
    print('To get the true test accuracy, submit predictions_python.csv '
          'at kaggle.com/competitions/titanic.')

section('DONE')