"""
Stochastic Cost Model - Predicts execution costs using machine learning.

Uses LightGBM to learn tool execution time and energy costs from historical data.
Replaces the hard-coded stub implementation.

Extracted from tool_selector.py to reduce module size.
"""

import logging
import pickle  # SECURITY: Internal data only, never deserialize untrusted data
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning(
        "LightGBM not available. StochasticCostModel will use a simple average."
    )


class StochasticCostModel:
    """
    Predicts execution costs (time, energy) using machine learning models.
    This replaces the hard-coded stub implementation.

    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.models = {}  # {tool_name: {'time': model, 'energy': model}}
        # CRITICAL FIX: Changed from defaultdict(lambda: defaultdict(list)) to defaultdict(list)
        self.data_buffer = defaultdict(list)
        self.retrain_threshold = config.get("retrain_threshold", 100)
        self.lock = threading.RLock()
        self.default_costs = {
            "symbolic": {"time": 2000, "energy": 200},
            "probabilistic": {"time": 800, "energy": 80},
            "causal": {"time": 3000, "energy": 300},
            "analogical": {"time": 600, "energy": 60},
            "multimodal": {"time": 8000, "energy": 800},  # Increased from 5000 to allow more processing time
            "philosophical": {"time": 1500, "energy": 150},  # FIX #2: Add philosophical tool costs
            "mathematical": {"time": 1200, "energy": 120},   # FIX #2: Add mathematical tool costs
        }

    def predict_cost(self, tool_name: str, features: np.ndarray) -> Dict[str, Any]:
        """Predict execution costs using a trained LightGBM model."""
        with self.lock:
            base = self.default_costs.get(tool_name, {"time": 1000, "energy": 100})
            if not LGBM_AVAILABLE or tool_name not in self.models:
                return {
                    "time": {"mean": base["time"], "std": base["time"] * 0.3},
                    "energy": {"mean": base["energy"], "std": base["energy"] * 0.3},
                }

            try:
                time_model = self.models[tool_name].get("time")
                energy_model = self.models[tool_name].get("energy")

                time_pred = (
                    time_model.predict(features.reshape(1, -1))[0]
                    if time_model
                    else base["time"]
                )
                energy_pred = (
                    energy_model.predict(features.reshape(1, -1))[0]
                    if energy_model
                    else base["energy"]
                )

                # Use historical variance as uncertainty estimate
                time_values = [
                    d["value"] for d in self.data_buffer.get(f"{tool_name}_time", [])
                ]
                energy_values = [
                    d["value"] for d in self.data_buffer.get(f"{tool_name}_energy", [])
                ]

                time_std = np.std(time_values) if time_values else base["time"] * 0.3
                energy_std = (
                    np.std(energy_values) if energy_values else base["energy"] * 0.3
                )

                return {
                    "time": {"mean": float(time_pred), "std": float(time_std)},
                    "energy": {"mean": float(energy_pred), "std": float(energy_std)},
                }
            except Exception as e:
                logger.error(f"Cost prediction failed for {tool_name}: {e}")
                return {
                    "time": {"mean": base["time"], "std": base["time"] * 0.3},
                    "energy": {"mean": base["energy"], "std": base["energy"] * 0.3},
                }

    def update(
        self,
        tool_name: str,
        component: str,
        value: float,
        features: Optional[np.ndarray] = None,
    ):
        """Add new data point and trigger retraining if buffer is full."""
        if features is None:
            return

        with self.lock:
            key = f"{tool_name}_{component}"
            self.data_buffer[key].append({"features": features, "value": value})

            if len(self.data_buffer[key]) >= self.retrain_threshold:
                self._train_model(tool_name, component)

    def _train_model(self, tool_name: str, component: str):
        """Train a new cost model for a specific tool and component."""
        if not LGBM_AVAILABLE:
            return

        key = f"{tool_name}_{component}"
        data = self.data_buffer.pop(key, [])
        if len(data) < 20:  # Need enough data to train
            self.data_buffer[key] = data  # Put it back
            return

        logger.info(f"Retraining cost model for {tool_name} -> {component}")
        X = np.vstack([d["features"] for d in data])
        y = np.array([d["value"] for d in data])

        try:
            model = lgb.LGBMRegressor(
                objective="regression_l1", n_estimators=50, random_state=42
            )
            model.fit(X, y)

            if tool_name not in self.models:
                self.models[tool_name] = {}
            self.models[tool_name][component] = model
        except Exception as e:
            logger.error(f"Failed to train cost model for {key}: {e}")
            self.data_buffer[key] = data  # Put data back on failure

    def save_model(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path, encoding="utf-8") / "cost_model.pkl", "wb") as f:
            pickle.dump(self.models, f)

    def load_model(self, path: str):
        model_path = Path(path) / "cost_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self.models = pickle.load(f)  # nosec B301 - Internal data structure
