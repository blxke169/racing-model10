from __future__ import annotations
import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from features import add_market_features, heuristic_form_probability

FEATURES=["implied_prob","no_vig_prob","log_price","bsp_vs_ppwap","field_size","market_rank"]

class RacingModel:
    def __init__(self):
        self.model=HistGradientBoostingClassifier(max_depth=4,learning_rate=.055,max_iter=220,l2_regularization=1.0,random_state=42)
        self.fitted=False

    def fit(self, df: pd.DataFrame):
        x=add_market_features(df)
        train=x.dropna(subset=["winner","market_price"]).copy()
        if train["winner"].nunique()<2 or len(train)<50:
            raise ValueError("Need at least 50 labelled runners containing both winners and losers.")
        self.model.fit(train[FEATURES].fillna(0), train["winner"].astype(int))
        self.fitted=True
        p=self.model.predict_proba(train[FEATURES].fillna(0))[:,1]
        return {"rows":len(train),"log_loss":float(log_loss(train.winner,p)),"brier":float(brier_score_loss(train.winner,p)),"auc":float(roc_auc_score(train.winner,p))}

    def predict_market(self, df: pd.DataFrame)->pd.Series:
        x=add_market_features(df)
        if not self.fitted:
            # sensible pre-training fallback: market no-vig probability
            return x["no_vig_prob"].fillna(1/len(x))
        raw=self.model.predict_proba(x[FEATURES].fillna(0))[:,1]
        s=pd.Series(raw,index=x.index)
        grp="market_id" if "market_id" in x and x["market_id"].notna().any() else None
        if grp:
            denom=s.groupby(x[grp]).transform("sum")
            return s/denom.replace(0,np.nan)
        return s/s.sum()

    def rank_race(self, runners: pd.DataFrame, market_prices: pd.Series|None=None, blend_market: float=.55):
        x=runners.copy().reset_index(drop=True)
        form_p=heuristic_form_probability(x)
        if market_prices is not None:
            price=pd.to_numeric(market_prices,errors="coerce").reset_index(drop=True)
            imp=np.where(price>1,1/price,np.nan)
            imp=pd.Series(imp).fillna(0)
            market_p=imp/imp.sum() if imp.sum()>0 else pd.Series(1/len(x),index=x.index)
            p=blend_market*market_p+(1-blend_market)*form_p
        else:
            market_p=pd.Series(np.nan,index=x.index)
            p=form_p
        p=p/p.sum()
        x["model_prob"]=p
        x["fair_odds"]=1/x["model_prob"].clip(lower=.001)
        x["market_odds"]=market_prices.values if market_prices is not None else np.nan
        x["edge_pct"]=np.where(pd.to_numeric(x["market_odds"],errors="coerce")>1,(x["market_odds"]*x["model_prob"]-1)*100,np.nan)
        x["confidence"]=pd.cut(x["model_prob"],[-1,.10,.18,.28,1],labels=["Low","Medium","High","Very High"])
        x["decision"]=np.where((x["edge_pct"]>=8)&(x["model_prob"]>=.10),"BET","NO BET")
        return x.sort_values("model_prob",ascending=False).reset_index(drop=True)

    def save(self,path): os.makedirs(os.path.dirname(path),exist_ok=True); joblib.dump(self,path)
    @staticmethod
    def load(path): return joblib.load(path)
