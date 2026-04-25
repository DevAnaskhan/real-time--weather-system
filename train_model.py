from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"



def generate_synthetic_dataset(n_samples: int = 8000) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate (temp_c, humidity, pressure) and a plausible next-hour temperature.
    Next hour is correlated with current temp with modest adjustments from humidity/pressure.
    """
    temp_c = RNG.uniform(-15.0, 42.0, size=n_samples)
    humidity = RNG.uniform(15.0, 100.0, size=n_samples)
    pressure = RNG.normal(1013.25, 25.0, size=n_samples)

    # Stylized next-hour change: drier / rising pressure tends to warm slightly, etc.
    delta = (
        0.12 * (pressure - 1013.25) / 10.0
        - 0.04 * (humidity - 55.0) / 10.0
        + RNG.normal(0.0, 0.35, size=n_samples)
    )
    next_hour_temp = temp_c + delta

    X = pd.DataFrame(
        {
            "temp_c": temp_c,
            "humidity": humidity,
            "pressure_hpa": pressure,
        }
    )
    return X, next_hour_temp


def main() -> None:
    X, y = generate_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "reg",
                GradientBoostingRegressor(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test R-squared: {score:.4f}")
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
