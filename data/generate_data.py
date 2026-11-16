"""
generate_data.py
Generates a synthetic shelter intake/outcome dataset by zip code and saves it
to data/shelter_data.csv.  Run once before launching the app if the CSV is
missing, or just let app.py call generate_if_missing().
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_ZIPS = 80

ZIP_BASE = 10001  # fictional zip codes 10001–10080


def generate(path: str = "data/shelter_data.csv") -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    zip_codes = [str(ZIP_BASE + i) for i in range(N_ZIPS)]

    # Raw features that drive intake pressure
    stray_intake = rng.integers(50, 800, size=N_ZIPS)
    owner_surrender = rng.integers(20, 400, size=N_ZIPS)
    adoption_rate = rng.uniform(0.30, 0.85, size=N_ZIPS)        # fraction adopted
    return_to_owner = rng.uniform(0.05, 0.25, size=N_ZIPS)
    median_household_income = rng.integers(28_000, 120_000, size=N_ZIPS)
    population_density = rng.integers(500, 12_000, size=N_ZIPS)  # people/sq-mi
    prior_spay_neuter_rate = rng.uniform(0.10, 0.75, size=N_ZIPS)
    vet_clinics_per_10k = rng.uniform(0.5, 6.0, size=N_ZIPS)
    avg_litter_size = rng.uniform(3.5, 7.5, size=N_ZIPS)
    repeat_intakes = rng.integers(5, 120, size=N_ZIPS)

    total_intake = stray_intake + owner_surrender

    # Synthetic "ground-truth" overpopulation risk score (0–100)
    # Higher intake, lower s/n rate, lower income → higher risk
    risk_raw = (
        0.30 * (stray_intake / stray_intake.max())
        + 0.20 * (owner_surrender / owner_surrender.max())
        + 0.25 * (1 - prior_spay_neuter_rate)
        + 0.10 * (1 - adoption_rate)
        + 0.08 * (1 - median_household_income / median_household_income.max())
        + 0.05 * (avg_litter_size / avg_litter_size.max())
        + 0.02 * (repeat_intakes / repeat_intakes.max())
    )
    # Add a little noise and scale to 0–100
    noise = rng.normal(0, 0.03, size=N_ZIPS)
    risk_score = np.clip((risk_raw + noise) * 100, 0, 100).round(1)

    df = pd.DataFrame(
        {
            "zip_code": zip_codes,
            "stray_intake": stray_intake,
            "owner_surrender": owner_surrender,
            "total_intake": total_intake,
            "adoption_rate": adoption_rate.round(3),
            "return_to_owner_rate": return_to_owner.round(3),
            "median_household_income": median_household_income,
            "population_density": population_density,
            "prior_spay_neuter_rate": prior_spay_neuter_rate.round(3),
            "vet_clinics_per_10k": vet_clinics_per_10k.round(2),
            "avg_litter_size": avg_litter_size.round(2),
            "repeat_intakes": repeat_intakes,
            "risk_score": risk_score,
        }
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df


def generate_if_missing(path: str = "data/shelter_data.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return generate(path)
    return pd.read_csv(path)


if __name__ == "__main__":
    df = generate()
    print(f"Generated {len(df)} rows → data/shelter_data.csv")
    print(df.head())
