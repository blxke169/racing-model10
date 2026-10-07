# AU Racing Edge

A deployable Australian thoroughbred racing model built around free/legitimate data sources.

## Data stack
1. **Betfair Australia public historical SP/market CSV files** — winner labels, BSP and market prices for model training/backtesting.
2. **FormFav free API** — race fields, runners, barrier, weight, jockey, trainer, form and career information. FormFav currently documents a free allowance of up to 1,000 requests/day.
3. Extension points are reserved for official state racing data/sectionals. Proprietary Punting Form data is not reproduced.

## What the model outputs
For each race: Top 4 runners, model win probability, fair odds, entered market odds, edge %, confidence and BET/NO BET.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Get a free FormFav key
Create a key at https://formfav.com/get-api-key and paste it into the Race Analyser. Do not commit the key to GitHub.

For Streamlit Community Cloud, add:
```toml
FORMFAV_API_KEY="fk_..."
```
to **App settings → Secrets**.

## Train on real Betfair history
Use **Data Builder → Download + train**, or:
```bash
python bootstrap.py --days 90
```
The downloader never substitutes generated rows if a source date fails.

## Model notes
The trained market model is a HistGradientBoosting classifier using implied probability/no-vig probability, price, price movement, field size and market rank. Race-day form is converted into a transparent form probability and blended with current market probability when odds are entered.

This is deliberately conservative: a fake historical dataset would make the backtest meaningless. A proper next iteration should add time-aware joins to official historical form/sectional data, then use date-based train/validation/test splits and probability calibration.

## Responsible use
Predictions are uncertain and betting can lose money. Treat the output as analysis, not a guarantee. Use small stakes and validate out of sample before relying on it.
