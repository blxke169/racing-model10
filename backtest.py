from __future__ import annotations
import numpy as np, pandas as pd
from features import add_market_features

def backtest(model, df: pd.DataFrame, min_edge=.05, stake=1.0):
    x=add_market_features(df).copy()
    p=model.predict_market(x)
    x["pred_prob"]=p
    x["fair_odds"]=1/p.clip(lower=.001)
    x["edge"]=x["market_price"]*p-1
    x["bet"]=(x["edge"]>=min_edge)&(x["market_price"]>1)
    x["profit"]=np.where(x["bet"],np.where(x["winner"]==1,(x["market_price"]-1)*stake,-stake),0.0)
    bets=x[x.bet]
    summary={"bets":int(len(bets)),"profit":float(bets.profit.sum()),"roi_pct":float(100*bets.profit.sum()/(len(bets)*stake)) if len(bets) else 0.0,"strike_rate_pct":float(100*bets.winner.mean()) if len(bets) else 0.0}
    return summary,x
