# Placeholder script: checks that Docker, packages and the data mount work.
suppressPackageStartupMessages(library(dplyr))

cat("Container is running!\n")
cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("dplyr version: %s\n", as.character(packageVersion("dplyr"))))

train_path <- file.path("data", "train.csv")
if (!file.exists(train_path)) {
  stop(paste(train_path, "not found. Put train.csv in src/data/ and",
             "mount it with -v (see README)."))
}

train <- read.csv(train_path)
cat(sprintf("Loaded train.csv: %d rows, %d columns\n",
            nrow(train), ncol(train)))