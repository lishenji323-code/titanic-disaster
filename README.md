# Titanic Disaster: Survival Prediction with Docker (Python and R)

This repository predicts which passengers survived the Titanic disaster
(April 15, 1912) using logistic regression. The same pipeline is
implemented twice, in **Python** and in **R**, and each runs in its own
Docker container.

Each script loads the data, explores it, cleans it and adds features,
trains a logistic regression model, and predicts survival for the test
set, printing every step to the terminal.

---

## Repository structure

```
titanic-disaster/
├── .gitignore
├── README.md
└── src/
    ├── data/                  # put train.csv and test.csv here (not in the repo)
    │   └── .gitkeep
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt   # pinned Python package versions
    │   └── titanic.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R # pinned R package versions
        └── titanic.R
```

The dataset is **not** included in this repository. You download it
from Kaggle (step 2 below), and it is mounted into the containers when
they run.

---

## Prerequisites

1. **Git**, to clone the repository.
2. **Docker Desktop**, installed and **running** before you use any
   `docker` command. Download it from
   [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).
   To check it works, run `docker --version` in a terminal.
3. **A free Kaggle account**, to download the data.

You do **not** need Python or R installed on your computer: both run
inside Docker.

---

## Step 1: Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/lishenji323-code/titanic-disaster.git
cd titanic-disaster
```

**All commands below are run from this top-level `titanic-disaster`
folder.**

---

## Step 2: Download the data

1. Go to [kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)
   and sign in.
2. If you see a **Join Competition** or **Late Submission** button,
   click it and accept the rules. Kaggle does not allow downloads until
   you do.
3. Click **Download All** and unzip the downloaded file.
4. Move **`train.csv`** and **`test.csv`** into the **`src/data/`**
   folder of this repository.

Your `src/data/` folder should now contain:

```
src/data/
├── .gitkeep
├── test.csv
└── train.csv
```

Do not rename the files: the scripts look for exactly `train.csv` and
`test.csv`.

---

## Step 3: Run the Python version

Make sure Docker Desktop is running, then:

**Build the image:**

```bash
docker build -t titanic-python src/python
```

**Run the container:**

macOS / Linux:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```

Windows (Command Prompt):

```cmd
docker run --rm -v "%cd%/src/data:/app/data" titanic-python
```

The first build takes a few minutes while Docker downloads Python and
installs the packages.

---

## Step 4: Run the R version

**Build the image:**

```bash
docker build -t titanic-r src/r
```

**Run the container:**

macOS / Linux:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-r
```

Windows (Command Prompt):

```cmd
docker run --rm -v "%cd%/src/data:/app/data" titanic-r
```

The first R build can take **5 to 15 minutes**, especially on Apple
Silicon Macs, because R packages are compiled from source. Later builds
reuse the cached packages and take only seconds.

---

## What the commands do

| Part of the command | Meaning |
|---|---|
| `docker build -t NAME FOLDER` | Builds an image called `NAME` from the Dockerfile in `FOLDER` |
| `docker run NAME` | Starts a container from the image, which runs the script |
| `--rm` | Deletes the container when the script finishes |
| `-v "<local path>:/app/data"` | Mounts your local `src/data` folder into the container at `/app/data`, where the scripts read the data and save predictions |

---

## Expected output

Each container prints its steps with numbered section headers that match
the assignment. A shortened example from the Python version:

```
======================================================================
STEP 13: Load the training data
======================================================================
Loaded train.csv: 891 rows, 12 columns
...
======================================================================
STEP 14b: Clean the data and add features
======================================================================
Added Title (from Name): {'Mr': 517, 'Miss': 185, 'Mrs': 126, 'Master': 40, 'Rare': 23}
Filled 177 missing Age values with the median age for each title
...
======================================================================
STEP 16: Accuracy on the training set
======================================================================
Training accuracy: 83.6%
...
======================================================================
DONE
======================================================================
```

Each run also saves its predictions in Kaggle's submission format:

- `src/data/predictions_python.csv`
- `src/data/predictions_r.csv`

---

## Results

| | Python | R |
|---|---|---|
| Held-out (validation) accuracy | 85.5% | 83.1% |
| Training accuracy | 83.6% | 82.9% |
| Baseline (predict everyone died) | 61.6% | 61.6% |

The two models make the same prediction for all 418 test passengers.

---

## Method

Both scripts follow the same steps:

1. **Explore** the data: column types, missing values, and survival
   rates by sex, class, and port of embarkation.
2. **Clean** the data and **add features**:
   - extract each passenger's **title** (Mr, Mrs, Miss, Master, or Rare)
     from their name
   - fill missing **Age** with the median age for that title
   - fill missing **Embarked** with the most common port, and missing
     **Fare** with the median fare
   - add **FamilySize** and **IsAlone**, and convert **Sex** to a number
   - drop **Cabin** (77% missing), **Ticket**, and **Name**

   All fill values are learned from the training data only, then applied
   to the test data, so no information leaks from the test set.
3. **Train** a logistic regression model, first on 80% of the training
   data to check accuracy on the 20% it has not seen, then on all of it.
4. **Predict** survival for the test set and save the predictions.

**Difference between the two models:** the R model leaves out
`FamilySize` (it equals `SibSp + Parch + 1`, so `glm()` cannot estimate
it separately) and `IsFemale` (it is almost fully determined by `Title`,
which made the estimates unstable). Python's `LogisticRegression` keeps
both because it applies regularization by default, while R's `glm()`
does not.

---

## About test accuracy

Kaggle's `test.csv` **does not include the `Survived` column**: Kaggle
keeps the true answers hidden to score competition submissions. So test
accuracy cannot be computed locally. Instead, each script:

- reports the **held-out accuracy** (on 20% of the training data the
  model never saw) as the best local estimate, and
- saves its test predictions in Kaggle's format, so the true test
  accuracy can be obtained by uploading the file at
  [kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)
  (**Submit Prediction**).

---

## Reproducibility

- **Python:** Python 3.12 (`python:3.12-slim`), with exact versions
  pinned in `src/python/requirements.txt`.
- **R:** R 4.4.1 (`rocker/r-ver:4.4.1`), with packages installed from a
  dated CRAN snapshot in `src/r/install_packages.R`, so versions never
  change.
- Random splits use a fixed seed, so every run gives the same results.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop is not running. Open it and wait until it shows **Engine running**. |
| `command not found: docker` | Docker Desktop is not installed, or the terminal was opened before installing it. Install Docker Desktop and open a new terminal. |
| `data/train.csv not found` | Check that `train.csv` and `test.csv` are in `src/data/`, and that you run the command from the top-level `titanic-disaster` folder. |
| `path "src/python" not found` | You are not in the top-level folder. Run `cd titanic-disaster` first. |
| The output does not reflect a code change | Rebuild the image with `docker build` before running it again. |