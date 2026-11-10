# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""SMOKE-TEST ONLY. Builds a synthetic file with the same schema as the real dataset so the code can be
tested end to end. Never use its output as a result."""
import sys, numpy as np, pandas as pd
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
rng = np.random.default_rng(1)
brands = [f"BRAND{i:02d}" for i in range(68)]
bw = rng.dirichlet(np.ones(68) * 0.3)
ids = rng.integers(0, int(n * 0.55), n)
df = pd.DataFrame(dict(insured_id=ids, year=rng.choice([2022, 2023, 2024], n),
    policy_type=rng.choice(["TPL", "COMP", "TPL+"], n), policy_status=rng.choice(["ACTIVE", "CANCELLED"], n, p=[.8, .2]),
    business_type=rng.choice(["NB", "RN"], n, p=[.3, .7]), payment_frequency=rng.choice(["ANNUAL", "SEMI", "MONTHLY"], n),
    bonus_score=rng.integers(1, 10, n), fuel_type=rng.choice(["GAS", "DIESEL", "HYBRID", "ELECTRIC"], n, p=[.45, .45, .08, .02]),
    vehicle_brand=rng.choice(brands, n, p=bw), municipality_type=rng.choice(["URBAN", "RURAL", "MIXED"], n),
    circulation_area=rng.choice(["NORTH", "SOUTH", "EAST", "WEST", "CENTRE"], n)))
df["total_exposure"] = rng.uniform(0.05, 1.0, n).round(3)
rate = 0.08 * (1 + 0.5 * (df.fuel_type == "DIESEL")) * (1 + 0.4 * (df.municipality_type == "URBAN"))
df["total_claims"] = rng.poisson(rate * df.total_exposure)
df["total_incurred"] = (df.total_claims * rng.gamma(2, 900, n)).round(2)
df["total_premium"] = (df.total_exposure * rng.uniform(250, 700, n)).round(2)
df.to_csv(sys.argv[1], sep=";", index=False); print("wrote", sys.argv[1], df.shape)
