
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BrakeFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.rotor_median_ = X["rotor_condition_score"].median()
        return self

    def transform(self, X):
        X_out = X.copy()

        # Brake pad wear rate per 1000 km
        X_out["wear_per_1k_km"] = (
            (12.0 - X_out["pad_thickness_mm"]) /
            (X_out["km_on_current_pads"] / 1000.0 + 1.0)
        )

        # Average distance driven per year
        X_out["km_per_year"] = (
            X_out["current_odometer_km"] /
            (X_out["vehicle_age_years"] + 0.5)
        )

        # Service frequency per year
        X_out["service_freq_per_year"] = (
            X_out["number_of_service_visits_total"] /
            (X_out["vehicle_age_years"] + 0.5)
        )

        # Pad and rotor risk score
        rotor_filled = X_out["rotor_condition_score"].fillna(
            self.rotor_median_
        )

        X_out["pad_rotor_risk_score"] = (
            X_out["pad_thickness_mm"] * rotor_filled
        )

        return X_out
