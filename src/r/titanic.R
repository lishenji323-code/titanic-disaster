# Titanic survival prediction with logistic regression (R version).
#
# Loads the Kaggle Titanic data from data/, explores and cleans it, trains
# a logistic regression model, and predicts survival for the test set.
# Every step prints what it did, so the full process is visible in the
# terminal.

suppressPackageStartupMessages(library(dplyr))

DATA_DIR <- "data"
SEED <- 42

# Print wide tables on one line instead of wrapping them
options(width = 200)

# Print a section header so the output is easy to follow
section <- function(title) {
  line <- strrep("=", 70)
  cat("\n", line, "\n", title, "\n", line, "\n", sep = "")
}

# Load a CSV from the data folder, with a clear error if it is missing.
# Empty cells are read as NA (missing), the same as pandas does.
load_csv <- function(name) {
  path <- file.path(DATA_DIR, name)
  if (!file.exists(path)) {
    stop(paste(path, "not found. Download the Kaggle Titanic data into",
               "src/data/ and mount it with -v (see README)."))
  }
  df <- read.csv(path, na.strings = c("", "NA"))
  cat(sprintf("Loaded %s: %d rows, %d columns\n", name, nrow(df), ncol(df)))
  df
}

# ----------------------------------------------------------------------
# Step 23: load packages and data
# ----------------------------------------------------------------------
section("STEP 23: Load the training data")
cat(sprintf("%s, dplyr %s\n", R.version.string,
            as.character(packageVersion("dplyr"))))
train <- load_csv("train.csv")
cat("\nFirst 5 rows:\n")
print(head(train, 5))

# ----------------------------------------------------------------------
# Step 24 (repeating step 14, part 1): explore the data
# ----------------------------------------------------------------------
section("STEP 24 / 14a: Explore the training data")

cat("Column types:\n")
print(data.frame(type = sapply(train, class)))

cat("\nMissing values per column:\n")
missing <- colSums(is.na(train))
missing <- missing[missing > 0]
print(data.frame(missing = missing,
                 percent = round(missing / nrow(train) * 100, 1)))

cat(sprintf("\nOverall survival rate: %.1f%%\n",
            mean(train$Survived) * 100))

for (col in c("Sex", "Pclass", "Embarked")) {
  rates <- train %>%
    group_by(.data[[col]]) %>%
    summarise(survival_pct = round(mean(Survived) * 100, 1),
              passengers = n(),
              .groups = "drop")
  cat(sprintf("\nSurvival rate by %s:\n", col))
  print(as.data.frame(rates), row.names = FALSE)
}
# ----------------------------------------------------------------------
# Step 24 (repeating step 14, part 2): clean the data and add features
# ----------------------------------------------------------------------
section("STEP 24 / 14b: Clean the data and add features")

# Extract the title (Mr, Mrs, Miss, ...) from passenger names.
# Names look like "Braund, Mr. Owen Harris", so the title is the text
# between the comma and the period. Rare titles are grouped together.
get_title <- function(names) {
  titles <- trimws(sub("^[^,]*,\\s*([^.]+)\\..*$", "\\1", names))
  titles <- recode(titles, Mlle = "Miss", Ms = "Miss", Mme = "Mrs")
  ifelse(titles %in% c("Mr", "Mrs", "Miss", "Master"), titles, "Rare")
}

# Most common value of a vector (R has no built-in mode function)
most_common <- function(x) {
  names(sort(table(x), decreasing = TRUE))[1]
}

# Learn the values used to fill missing data, from training data only.
# Using only the training data avoids leaking information from the test
# set into the model.
learn_fill_values <- function(df) {
  titles <- get_title(df$Name)
  list(
    age_by_title = tapply(df$Age, titles, median, na.rm = TRUE),
    age_overall = median(df$Age, na.rm = TRUE),
    embarked = most_common(df$Embarked),
    fare = median(df$Fare, na.rm = TRUE)
  )
}

# Clean a raw Titanic data frame and add model features.
# fill: fill values from learn_fill_values(), computed on train.
# verbose: if TRUE, print every change that is made.
prepare <- function(df, fill, verbose = TRUE) {
  say <- function(...) if (verbose) cat(sprintf(...), "\n", sep = "")

  # Title: a strong signal for age and sex/social status
  df$Title <- get_title(df$Name)
  counts <- sort(table(df$Title), decreasing = TRUE)
  say("Added Title (from Name): %s",
      paste(names(counts), counts, sep = ": ", collapse = ", "))

  # Age: fill missing values with the median age for that title,
  # e.g. a missing "Master" (young boy) gets a child's age
  n_age <- sum(is.na(df$Age))
  df$Age <- coalesce(df$Age, unname(fill$age_by_title[df$Title]),
                     fill$age_overall)
  say("Filled %d missing Age values with the median age for each title",
      n_age)

  # Embarked: only a few missing, so use the most common port
  n_emb <- sum(is.na(df$Embarked))
  df$Embarked <- coalesce(df$Embarked, fill$embarked)
  say("Filled %d missing Embarked values with '%s' (most common port)",
      n_emb, fill$embarked)

  # Fare: fill any missing values with the median fare
  n_fare <- sum(is.na(df$Fare))
  df$Fare <- coalesce(df$Fare, fill$fare)
  say("Filled %d missing Fare values with the median fare (%.2f)",
      n_fare, fill$fare)

  # Family features
  df <- df %>%
    mutate(FamilySize = SibSp + Parch + 1L,
           IsAlone = as.integer(FamilySize == 1))
  say("Added FamilySize (SibSp + Parch + 1) and IsAlone (1 if traveling alone)")

  # Sex: convert text to a number
  df$IsFemale <- as.integer(df$Sex == "female")
  say("Converted Sex to IsFemale (1 = female, 0 = male)")

  # Drop columns that are not used as features
  dropped <- c("Name", "Sex", "Ticket", "Cabin")
  df <- df[, setdiff(names(df), dropped)]
  say(paste("Dropped columns: %s (Cabin is 77%% missing, Ticket is an",
            "ID-like code, and Name and Sex are replaced by new features)"),
      paste(dropped, collapse = ", "))

  # Categorical columns become factors with fixed levels, so train and
  # test always have the same categories. glm() turns factors into 0/1
  # columns automatically (the R equivalent of one-hot encoding).
  df$Embarked <- factor(df$Embarked, levels = c("C", "Q", "S"))
  df$Title <- factor(df$Title,
                     levels = c("Master", "Miss", "Mr", "Mrs", "Rare"))
  say("Converted Embarked and Title to factors (categories)")
  df
}

fill_values <- learn_fill_values(train)
train_clean <- prepare(train, fill_values)

cat(sprintf("\nCleaned training data: %d rows, %d columns\n",
            nrow(train_clean), ncol(train_clean)))
cat(sprintf("Missing values remaining: %d\n", sum(is.na(train_clean))))
cat("\nFirst 5 rows after cleaning:\n")
print(head(train_clean, 5))

# ----------------------------------------------------------------------
# Step 24 (repeating step 15): build a logistic regression model
# ----------------------------------------------------------------------
section("STEP 24 / 15: Build a logistic regression model")

# Two features from the cleaning step are left out of the R model:
# - FamilySize equals SibSp + Parch + 1 exactly, so glm() cannot
#   estimate it separately from SibSp and Parch.
# - IsFemale is almost fully determined by Title (Mr and Master are male,
#   Miss and Mrs are female), so including both makes the estimates
#   unstable. Title is kept because it also separates boys (Master)
#   from men.
# Python's LogisticRegression can keep both because it applies
# regularization by default; R's glm() does not.
FEATURES <- c("Pclass", "Age", "SibSp", "Parch", "Fare", "IsAlone",
              "Embarked", "Title")
model_formula <- reformulate(FEATURES, response = "Survived")
cat("Model formula:", deparse(model_formula, width.cutoff = 200), "\n")
cat("Left out: FamilySize (= SibSp + Parch + 1) and IsFemale",
    "(already captured by Title)\n")

# Predict 1 (survived) when the predicted probability is above 0.5
predict_class <- function(model, newdata) {
  as.integer(predict(model, newdata = newdata, type = "response") > 0.5)
}

accuracy <- function(actual, predicted) mean(actual == predicted)

# Hold out 20% of the training data to check the model on passengers it
# has not seen, since test.csv has no Survived column to check against.
# Sampling within each class keeps the survival rate the same in both
# parts (a stratified split).
set.seed(SEED)
val_idx <- unlist(lapply(split(seq_len(nrow(train_clean)),
                               train_clean$Survived),
                         function(idx) sample(idx, round(0.2 * length(idx)))))
fit_part <- train_clean[-val_idx, ]
val_part <- train_clean[val_idx, ]

check_model <- glm(model_formula, data = fit_part, family = binomial)
val_acc <- accuracy(val_part$Survived, predict_class(check_model, val_part))
cat(sprintf("\nHeld-out check: trained on %d passengers, tested on %d unseen passengers\n",
            nrow(fit_part), nrow(val_part)))
cat(sprintf("Held-out (validation) accuracy: %.1f%%\n", val_acc * 100))

# Refit on all 891 training passengers for the final model
model <- glm(model_formula, data = train_clean, family = binomial)
cat("\nRefit the final model on all 891 training passengers\n")

cat("\nModel coefficients (positive = higher chance of survival):\n")
coefs <- summary(model)$coefficients
print(round(coefs[order(coefs[, "Estimate"]), c("Estimate", "Pr(>|z|)")], 3))

# ----------------------------------------------------------------------
# Step 24 (repeating step 16): accuracy on the training set
# ----------------------------------------------------------------------
section("STEP 24 / 16: Accuracy on the training set")
train_acc <- accuracy(train_clean$Survived, predict_class(model, train_clean))
cat(sprintf("Training accuracy: %.1f%%\n", train_acc * 100))
cat(sprintf("Baseline (predict everyone died): %.1f%%\n",
            (1 - mean(train_clean$Survived)) * 100))


# ----------------------------------------------------------------------
# Step 24 (repeating step 17): load test.csv and predict
# ----------------------------------------------------------------------
section("STEP 24 / 17: Load test.csv and predict survival")
test <- load_csv("test.csv")

# Clean the test data with the SAME steps and the fill values learned
# from the training data. The fixed factor levels in prepare() make sure
# the test categories are encoded exactly like the training ones.
cat("\nApplying the same cleaning steps to the test data:\n")
test_clean <- prepare(test, fill_values)

test_pred <- predict_class(model, test_clean)
predictions <- data.frame(PassengerId = test$PassengerId,
                          Survived = test_pred)
cat(sprintf("\nPredicted survival rate on test set: %.1f%%\n",
            mean(test_pred) * 100))
cat("\nFirst 10 predictions:\n")
print(head(predictions, 10), row.names = FALSE)

out_path <- file.path(DATA_DIR, "predictions_r.csv")
write.csv(predictions, out_path, row.names = FALSE)
cat(sprintf("\nSaved all %d predictions to src/%s\n",
            nrow(predictions), out_path))

# ----------------------------------------------------------------------
# Step 24 (repeating step 18): accuracy on the test set
# ----------------------------------------------------------------------
section("STEP 24 / 18: Accuracy on the test set")
if ("Survived" %in% names(test)) {
  test_acc <- accuracy(test$Survived, test_pred)
  cat(sprintf("Test accuracy: %.1f%%\n", test_acc * 100))
} else {
  cat("test.csv has no Survived column: Kaggle keeps the true answers",
      "hidden, so test accuracy cannot be computed locally.\n")
  cat(sprintf("Best local estimate (held-out accuracy from step 15): %.1f%%\n",
              val_acc * 100))
  cat("To get the true test accuracy, submit predictions_r.csv at",
      "kaggle.com/competitions/titanic.\n")
}

section("DONE")