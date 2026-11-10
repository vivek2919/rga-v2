# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Stage 0: profile the data and write artifacts/data_card.json (fields and levels shown to the models)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rga import config as C
from rga.common import *
df = load_data()
print(f"records {len(df):,}  policyholders {df['insured_id'].nunique():,}  years {sorted(df['year'].unique().tolist())}  variables {df.shape[1]}")
print("duplicates", int(df.duplicated().sum()), " negatives:", {c: int((df[c] < 0).sum()) for c in ["total_claims", "total_exposure", "total_incurred", "total_premium"]})
rec = (df.groupby("insured_id")["year"].nunique() > 1).mean(); print(f"recurrence (>1 year): {rec:.1%}")
print("new-business share by year:", df.groupby("year")["business_type"].apply(lambda s: (s == "NB").mean()).round(3).to_dict())
card = {"columns": {c: ("numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical") for c in df.columns},
        "segment_fields": {f: levels(df, f) for f in segment_fields(df)}, "years": sorted(int(y) for y in df["year"].unique())}
os.makedirs(os.path.join(C.ROOT, "artifacts"), exist_ok=True)
json.dump(card, open(os.path.join(C.ROOT, "artifacts", "data_card.json"), "w"), indent=1)
print("brands:", df["vehicle_brand"].nunique() if "vehicle_brand" in df else "n/a", "| wrote artifacts/data_card.json")
