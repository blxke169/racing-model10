from datetime import date,timedelta
from data_sources import fetch_betfair_range, standardize_betfair
from model import RacingModel
import argparse, os
p=argparse.ArgumentParser(); p.add_argument("--days",type=int,default=90); a=p.parse_args()
end=date.today()-timedelta(days=1); start=end-timedelta(days=a.days-1)
raw,errors=fetch_betfair_range(start,end)
if raw.empty: raise SystemExit("No files downloaded.\n"+"\n".join(errors[:10]))
clean=standardize_betfair(raw); os.makedirs("data",exist_ok=True); clean.to_csv("data/betfair_history.csv",index=False)
m=RacingModel(); print(m.fit(clean)); m.save("models/market_model.joblib")
print(f"saved {len(clean)} rows; {len(errors)} unavailable dates")
