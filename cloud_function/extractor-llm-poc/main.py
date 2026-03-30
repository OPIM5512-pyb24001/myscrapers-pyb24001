 A08: Midterm Project Week — Craigslist LLM + Modeling Challenge
Dr. Dave Wanik - Operations and Infromation Management - University of Connecticut

This is your chance to push an end-to-end data science pipeline further and show what you can build. You will work with Craigslist data only, extending either the LLM-enhanced car-price pipeline we’ve been developing or applying the same ideas to a different Craigslist category.

🧭 Choose ONE Project Path
Option A — Improve the Craigslist Car Price Model
You will start from the provided /myscrapers repository and extend the existing pipeline.

Current pipeline
Scrapes Craigslist car listings
Uses RegEx + LLMs (Gemini) for ETL
Extracts structured fields (e.g., price, mileage, transmission)
Trains a baseline model to predict car prices
Syncs predictions back to GitHub using GitHub Actions
Your task
Push this pipeline further by improving both the LLM-based ETL and the modeling.

Requirements
Keep the scraping logic FIXED (everyone uses the same dataset)
Extend the LLM ETL to extract additional features, such as:
color
city
state
zip_code
Update the materialize-llm Cloud Function to support your new fields
Train an improved model using hyperparameter tuning:
Grid search, Optuna, or AutoML are all acceptable
Use past data to predict today’s listings
Send the following artifacts to GitHub (via Actions):
Predictions
Permutation importance (all features)
Partial Dependence Plots (PDPs) for your top 3 features
Create a well-documented Jupyter notebook that:
Tracks model performance as the dataset grows
Includes a dashboard showing:
Model accuracy (MAE, MAPE, RMSE, Bias)
Feature importance (permutation importance)
How the model is using the data — and whether patterns change over time
Option B — New Craigslist Scraper + Model
Instead of cars, you may choose a different Craigslist category (must be Craigslist).

Requirements
Build a scraper for your chosen Craigslist category
Use RegEx + LLMs to perform ETL and create structured features
Define a meaningful prediction target (price, time-to-sell, etc.)
Train and tune at least one model
Sync predictions and model artifacts back to GitHub
Create a notebook documenting:
Model performance
Feature importance
How patterns evolve as more data is collected
Important: Your dataset must come from Craigslist.

📦 Deliverables
1. Code & Repository — 80%
On your Discussion Group (M3.4), you should post:

A link to your public GitHub repo (we will check out everyone's network graph and commit history)
A link to your 'model trending' notebook
Your GitHub repo should include:

Updated ETL logic (LLM + RegEx)
Updated cloud functions (if applicable)
Tuned model(s)
Synced predictions, permutation importance, and PDP outputs
A clean, reproducible pipeline
Your 'model trending' notebook:

Everything should be self-contained in the notebook and easily opened from Google Drive or Colab. It should 1) clone your repo, 2) read the results and 3) trends how the error and interpretability (importance or PDP of top features) of your model changes over time. It should be very easy and quick to run.
Companies want to know that you can productionalize and trend models... and this is one of your first chances to do just that!

Of course, we will also be looking at your network graph to see when and what you committed.

2. Short Video Overview — 20%
2 minutes MAX First, at the beginning of the video, hold up your UConn Student ID card and state your name and netID so we know who you are (for course authentication purposes.)

Then, briefly explain:

What you built
What you improved or discovered
One thing that surprised you
You should briefly show us your repo and your 'model trending' notebook.

Post the video (directly embedded) to your Discussion Group and engage with your teammates.

10-50% points off if you go more than a few seconds over 2 min limit.

🎯 Evaluation Emphasis
Thoughtful feature engineering with LLMs
Sound modeling and validation choices
Clear evidence that the model improves (or evolves) over time
Clean, professional documentation
This is a capstone-style project — treat it like something you’d be proud to show in a portfolio.

