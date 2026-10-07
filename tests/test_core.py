import pandas as pd
from features import add_market_features, heuristic_form_probability
from model import RacingModel

def test_probs_sum_one():
    d=pd.DataFrame({"name":["A","B","C"],"form":["112","543","999"],"barrier":[1,5,10],"weight":[56,55,54],"careerStats_starts":[10,10,10],"careerStats_wins":[4,2,1]})
    p=heuristic_form_probability(d)
    assert abs(p.sum()-1)<1e-8

def test_rank_race():
    d=pd.DataFrame({"name":["A","B","C"],"form":["112","543","999"]})
    o=pd.Series([3.0,4.0,8.0])
    r=RacingModel().rank_race(d,o)
    assert {"model_prob","fair_odds","decision"}.issubset(r.columns)
