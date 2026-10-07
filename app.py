from __future__ import annotations
import os, sqlite3, pandas as pd, streamlit as st
from datetime import date, timedelta
from data_sources import fetch_betfair_range, standardize_betfair, fetch_formfav_meetings, fetch_formfav_form, flatten_formfav_race
from model import RacingModel
from backtest import backtest

st.set_page_config(page_title="AU Racing Edge",page_icon="🏇",layout="wide")
st.title("🏇 AU Racing Edge")
st.caption("Betfair + FormFav + free Australian racing-data model • Top 4 • fair odds • value • NO BET")

MODEL_PATH="models/market_model.joblib"
DATA_PATH="data/betfair_history.csv"

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH): return RacingModel.load(MODEL_PATH)
    return RacingModel()

model=get_model()

tab1,tab2,tab3,tab4=st.tabs(["Race Analyser","Data Builder","Backtest","About"])

with tab1:
    st.subheader("Race analyser")
    key=st.text_input("FormFav API key",type="password",value=st.secrets.get("FORMFAV_API_KEY","") if hasattr(st,"secrets") else "")
    c1,c2=st.columns([1,1])
    with c1: race_date=st.date_input("Race date",value=date.today())
    with c2: st.caption("Free FormFav accounts allow up to 1,000 requests/day according to its current docs.")
    if key:
        try:
            meetings=fetch_formfav_meetings(key,race_date.isoformat())
            opts=[]
            for m in meetings.get("meetings",[]):
                for r in m.get("races",[]): opts.append((f"{m.get('track')} R{r.get('raceNumber')} — {r.get('raceName','')}",m.get('slug',m.get('track')),r.get('raceNumber')))
            if opts:
                labels=[o[0] for o in opts]; sel=st.selectbox("Race",labels); item=opts[labels.index(sel)]
                payload=fetch_formfav_form(key,race_date.isoformat(),item[1],item[2])
                runners=flatten_formfav_race(payload)
                if "scratched" in runners: runners=runners[~runners["scratched"].fillna(False)].reset_index(drop=True)
                st.markdown("#### Enter current market odds")
                odds={}
                cols=st.columns(2)
                for i,row in runners.iterrows():
                    with cols[i%2]: odds[i]=st.number_input(str(row.get("name",f"Runner {i+1}")),min_value=1.01,value=5.00,step=.10,key=f"odd_{i}")
                ranked=model.rank_race(runners,pd.Series(odds))
                show=[c for c in ["number","name","jockey","trainer","barrier","weight","form","model_prob","fair_odds","market_odds","edge_pct","confidence","decision"] if c in ranked]
                out=ranked[show].head(4).copy(); out["model_prob"]=(out["model_prob"]*100).round(1).astype(str)+"%"; out["fair_odds"]=out["fair_odds"].round(2); out["edge_pct"]=out["edge_pct"].round(1)
                st.dataframe(out,use_container_width=True,hide_index=True)
                if (ranked["decision"]=="BET").any(): st.success("Model found at least one runner above the default 8% value threshold.")
                else: st.info("NO BET — no runner clears the default value threshold.")
            else: st.warning("No gallops meetings returned for that date.")
        except Exception as e: st.error(f"FormFav request failed: {e}")
    else:
        st.info("Add your free FormFav API key to load Australian race fields and form automatically.")

with tab2:
    st.subheader("Build real Betfair training data")
    st.write("Downloads public Australian Betfair SP CSV files directly from Betfair. Nothing synthetic is added.")
    c1,c2=st.columns(2)
    with c1: start=st.date_input("Start",value=date.today()-timedelta(days=30),key="start")
    with c2: end=st.date_input("End",value=date.today()-timedelta(days=1),key="end")
    if st.button("Download + train",type="primary"):
        with st.spinner("Downloading Betfair public files..."):
            raw,errors=fetch_betfair_range(start,end)
        if raw.empty:
            st.error("No Betfair CSVs could be downloaded. The public file host may have changed its filename pattern or blocked this environment.")
            if errors: st.code("\n".join(errors[:8]))
        else:
            clean=standardize_betfair(raw); os.makedirs("data",exist_ok=True); clean.to_csv(DATA_PATH,index=False)
            st.success(f"Saved {len(clean):,} runner rows.")
            try:
                metrics=model.fit(clean); model.save(MODEL_PATH); st.cache_resource.clear()
                st.success(f"Model trained: {metrics}")
            except Exception as e: st.warning(f"Data saved, but model training needs more usable labelled rows: {e}")
            if errors: st.warning(f"{len(errors)} dates were unavailable; the rest were kept.")

with tab3:
    st.subheader("Backtest")
    if os.path.exists(DATA_PATH) and model.fitted:
        hist=pd.read_csv(DATA_PATH)
        edge=st.slider("Minimum edge",0,30,8,1)/100
        summary,detail=backtest(model,hist,min_edge=edge)
        a,b,c,d=st.columns(4); a.metric("Bets",summary["bets"]); b.metric("Profit (1u)",f"{summary['profit']:.1f}u"); c.metric("ROI",f"{summary['roi_pct']:.1f}%"); d.metric("Strike",f"{summary['strike_rate_pct']:.1f}%")
        st.caption("Training-set backtest is diagnostic only. For genuine performance, use a date-based holdout/out-of-sample period before staking real money.")
    else: st.info("Build and train historical data in the Data Builder tab first.")

with tab4:
    st.markdown("""
### What this version does
- Pulls free **FormFav** Australian race form into the race analyser.
- Pulls public **Betfair Australia** starting-price history for training and backtesting.
- Produces **Top 4**, model win probability, fair odds, market edge, confidence and **BET / NO BET**.
- Uses a market ML model when trained and a transparent form-rating layer for race-day enrichment.

### Important limitation
Punting Form's proprietary sectional benchmarks, speed maps and point-in-time modelling data are not copied or fabricated. The project has clean extension points for official sectional data once a lawful machine-readable feed is available.
""")
