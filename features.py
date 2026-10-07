from __future__ import annotations
import re, numpy as np, pandas as pd


def _num(v):
    if v is None: return np.nan
    if isinstance(v, (int,float,np.number)): return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",",""))
    return float(m.group()) if m else np.nan


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ["bsp","ppwap","morning_wap"]:
        if c not in x: x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    price = x["ppwap"].fillna(x["bsp"]).fillna(x["morning_wap"])
    x["market_price"] = price
    x["implied_prob"] = np.where(price > 1, 1.0/price, np.nan)
    x["log_price"] = np.where(price > 1, np.log(price), np.nan)
    x["bsp_vs_ppwap"] = np.where((x["bsp"]>0)&(x["ppwap"]>0), x["bsp"]/x["ppwap"]-1, 0.0)
    grp = None
    for k in ["market_id","event_id","event_name"]:
        if k in x and x[k].notna().any(): grp = k; break
    if grp:
        x["field_size"] = x.groupby(grp)["runner"].transform("count")
        x["market_rank"] = x.groupby(grp)["market_price"].rank(method="min")
        overround = x.groupby(grp)["implied_prob"].transform("sum")
        x["no_vig_prob"] = x["implied_prob"] / overround.replace(0,np.nan)
    else:
        x["field_size"] = len(x)
        x["market_rank"] = x["market_price"].rank(method="min")
        x["no_vig_prob"] = x["implied_prob"] / x["implied_prob"].sum()
    return x


def add_formfav_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    def first_existing(names):
        for n in names:
            if n in x.columns: return x[n]
        return pd.Series(np.nan,index=x.index)
    x["barrier_n"] = first_existing(["barrier"]).map(_num)
    x["weight_n"] = first_existing(["weight"]).map(_num)
    x["age_n"] = first_existing(["age"]).map(_num)
    x["career_starts"] = first_existing(["careerStats_starts","stats_starts","starts"]).map(_num)
    x["career_wins"] = first_existing(["careerStats_wins","stats_wins","wins"]).map(_num)
    x["career_places"] = first_existing(["careerStats_places","stats_places","places"]).map(_num)
    x["career_win_rate"] = np.where(x["career_starts"]>0, x["career_wins"]/x["career_starts"], np.nan)
    x["career_place_rate"] = np.where(x["career_starts"]>0, x["career_places"]/x["career_starts"], np.nan)
    # Parse recent form string: smaller finish positions score higher; 0/x/f ignored.
    form_s = first_existing(["form"]).astype(str)
    def form_score(s):
        nums = [int(v) for v in re.findall(r"\d", s)[-5:] if v != "0"]
        if not nums: return np.nan
        return float(np.mean([1.0/max(1,n) for n in nums]))
    x["recent_form_score"] = form_s.map(form_score)
    return x


def heuristic_form_probability(df: pd.DataFrame) -> pd.Series:
    x = add_formfav_features(df)
    score = pd.Series(0.0,index=x.index,dtype=float)
    for col, weight in [("career_win_rate",2.2),("career_place_rate",0.8),("recent_form_score",1.5)]:
        v=x[col].astype(float)
        med=v.median() if v.notna().any() else 0.0
        score += weight*v.fillna(med)
    if x["barrier_n"].notna().any():
        score += -0.025*(x["barrier_n"].fillna(x["barrier_n"].median())-1)
    if x["weight_n"].notna().any():
        score += -0.015*(x["weight_n"].fillna(x["weight_n"].median())-x["weight_n"].median())
    score = score - score.max()
    exp=np.exp(score.clip(-20,20))
    return exp/exp.sum() if exp.sum()>0 else pd.Series(1/len(x),index=x.index)
