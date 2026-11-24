# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : model_suite.py
# Purpose : Every learned model of the project, fitted on 2023 and tuned on
#           2024 ONLY (hyperparameters, early stopping). 2025 is only ever
#           predicted, never looked at here. Predictions are cached so that
#           Parts 1-3 share exactly the same fitted models.
# ==============================================================================

import hashlib
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, Lasso, LogisticRegression, QuantileRegressor, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config
from feature_factory import ALL_FEATURE_COLUMNS, FEATURE_GROUPS, PRICE_FEATURE_GROUPS

PRICE_FEATURE_SET = {c for g in PRICE_FEATURE_GROUPS for c in FEATURE_GROUPS[g]}
ONE_HOT_FEATURES = ("hour_ending", "day_of_week", "month")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ---------------------------------------------------------------------- losses used for tuning
def huber_loss(true_values, predicted_values, delta=config.HUBER_DELTA_DOLLARS):
    residual = np.abs(np.asarray(true_values) - np.asarray(predicted_values))
    return float(np.mean(np.where(residual <= delta, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))))


LOSS_SCORERS = {"squared": mean_squared_error, "absolute": mean_absolute_error, "huber": huber_loss}


def make_linear_preprocessor(feature_columns, price_transform="asinh"):
    # So, linear models need three things trees do not: one-hot calendar codes, a squashing of the
    # heavy-tailed $ price features (asinh is linear near 0 and logarithmic in the tails, so one
    # $3,900 day cannot dominate a coefficient), and standardization.
    price_columns = [c for c in feature_columns if c in PRICE_FEATURE_SET]
    one_hot_columns = [c for c in feature_columns if c in ONE_HOT_FEATURES]
    other_columns = [c for c in feature_columns if c not in PRICE_FEATURE_SET and c not in ONE_HOT_FEATURES]
    squash_step = FunctionTransformer(np.arcsinh, feature_names_out="one-to-one") if price_transform == "asinh" else "passthrough"
    column_steps = []
    if price_columns:
        column_steps.append(("price", Pipeline([("squash", squash_step), ("scale", StandardScaler())]), price_columns))
    if one_hot_columns:
        column_steps.append(("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), one_hot_columns))
    if other_columns:
        column_steps.append(("scale", StandardScaler(), other_columns))
    return ColumnTransformer(column_steps)


def lightgbm_base_params(objective_name, extra_params=None):
    # In my opinion deterministic + fixed seed is non-negotiable: the grader must get our numbers.
    base_params = {"objective": objective_name, "learning_rate": 0.03, "n_estimators": 3000,
                   "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                   "random_state": config.GLOBAL_RANDOM_SEED, "deterministic": True, "force_row_wise": True,
                   "n_jobs": 2, "verbose": -1}
    base_params.update(extra_params or {})
    return base_params


LIGHTGBM_GRID = [{"num_leaves": 15, "min_child_samples": 50}, {"num_leaves": 15, "min_child_samples": 200},
                 {"num_leaves": 31, "min_child_samples": 50}, {"num_leaves": 31, "min_child_samples": 200}]


class FittedModel:
    """A fitted estimator plus everything needed to explain and reuse it."""

    def __init__(self, model_name, estimator, feature_columns, chosen_params, tuning_table, task_kind):
        self._model_name = model_name
        self._estimator = estimator
        self._feature_columns = list(feature_columns)
        self._chosen_params = dict(chosen_params)
        self._tuning_table = tuning_table
        self._task_kind = task_kind

    def get_name(self):
        return self._model_name

    def get_params(self):
        return dict(self._chosen_params)

    def get_tuning_table(self):
        return self._tuning_table.assign(model=self._model_name)

    def get_estimator(self):
        return self._estimator

    def get_feature_columns(self):
        return list(self._feature_columns)

    def predict(self, frame):
        feature_matrix = frame[self._feature_columns]
        if self._task_kind == "binary":
            return self._estimator.predict_proba(feature_matrix)[:, 1]
        if self._task_kind == "multiclass":
            return self._estimator.predict_proba(feature_matrix)
        return self._estimator.predict(feature_matrix)


# ---------------------------------------------------------------------- classifiers
def fit_logistic(train_frame, validation_frame, feature_columns, label_fn, weight_fn=None, model_name="logistic",
                 c_grid=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2, 0.1, 1.0), transform_grid=("none", "asinh"), fixed_params=None):
    # So, grid over the L2 strength C and the price transform; pick the lowest VALIDATION log loss
    # (weighted the same way as training when the model is profit-weighted).
    train_labels, validation_labels = label_fn(train_frame), label_fn(validation_frame)
    train_weights = weight_fn(train_frame) if weight_fn else None
    validation_weights = weight_fn(validation_frame) if weight_fn else None
    is_multiclass = len(np.unique(train_labels)) > 2
    candidate_grid = ([fixed_params] if fixed_params else
                      [{"C": c, "price_transform": t} for t in transform_grid for c in c_grid])
    tuning_rows, best_score, best_pipeline, best_params = [], np.inf, None, None
    for candidate in candidate_grid:
        candidate_pipeline = Pipeline([
            ("prep", make_linear_preprocessor(feature_columns, candidate["price_transform"])),
            ("model", LogisticRegression(C=candidate["C"], max_iter=5000))])
        candidate_pipeline.fit(train_frame[feature_columns], train_labels, model__sample_weight=train_weights)
        validation_probability = candidate_pipeline.predict_proba(validation_frame[feature_columns])
        if not is_multiclass:
            validation_probability = validation_probability[:, 1]
        validation_score = log_loss(validation_labels, np.clip(validation_probability, 1e-6, 1 - 1e-6),
                                    sample_weight=validation_weights, labels=np.unique(train_labels))
        tuning_rows.append({**candidate, "validation_log_loss": validation_score})
        if validation_score < best_score:
            best_score, best_pipeline, best_params = validation_score, candidate_pipeline, candidate
    return FittedModel(model_name, best_pipeline, feature_columns, best_params, pd.DataFrame(tuning_rows),
                       "multiclass" if is_multiclass else "binary")


def fit_lightgbm_classifier(train_frame, validation_frame, feature_columns, model_name="lightgbm_classifier",
                            fixed_params=None):
    tuning_rows, best_score, best_model, best_params = [], np.inf, None, None
    for candidate in ([fixed_params] if fixed_params else LIGHTGBM_GRID):
        tree_params = {k: v for k, v in candidate.items() if k != "n_estimators"}
        if fixed_params and "n_estimators" in fixed_params:
            tree_params["n_estimators"] = fixed_params["n_estimators"]
        candidate_model = lgb.LGBMClassifier(**lightgbm_base_params("binary", tree_params))
        fit_kwargs = {}
        if not (fixed_params and "n_estimators" in fixed_params):
            fit_kwargs = {"eval_X": (validation_frame[feature_columns],), "eval_y": (validation_frame["y"],),
                          "eval_metric": "binary_logloss", "callbacks": [lgb.early_stopping(100, verbose=False)]}
        candidate_model.fit(train_frame[feature_columns], train_frame["y"], **fit_kwargs)
        validation_probability = candidate_model.predict_proba(validation_frame[feature_columns])[:, 1]
        validation_score = log_loss(validation_frame["y"], np.clip(validation_probability, 1e-6, 1 - 1e-6))
        used_trees = candidate_model.best_iteration_ or candidate_model.n_estimators
        tuning_rows.append({**candidate, "n_estimators": used_trees, "validation_log_loss": validation_score})
        if validation_score < best_score:
            best_score, best_model = validation_score, candidate_model
            best_params = {**candidate, "n_estimators": used_trees}
    return FittedModel(model_name, best_model, feature_columns, best_params, pd.DataFrame(tuning_rows), "binary")


# ---------------------------------------------------------------------- regressors
def fit_linear_regressor(train_frame, validation_frame, feature_columns, target_fn, loss_name, model_name,
                         fixed_params=None):
    # Let's do now one function for the four linear families: Ridge / Lasso (squared), Huber, LAD (absolute).
    family_grids = {"ridge": [{"alpha": a} for a in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)],
                    "lasso": [{"alpha": a} for a in (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)],
                    "huber_linear": [{"alpha": a} for a in (1e-2, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)],
                    "lad_linear": [{"alpha": a} for a in (1e-3, 1e-2, 0.1, 1.0)]}
    family_name = model_name.split("__")[-1]
    train_target, validation_target = target_fn(train_frame), target_fn(validation_frame)
    tuning_rows, best_score, best_pipeline, best_params = [], np.inf, None, None
    for candidate in ([fixed_params] if fixed_params else family_grids[family_name]):
        if family_name == "ridge":
            estimator = Ridge(alpha=candidate["alpha"])
        elif family_name == "lasso":
            estimator = Lasso(alpha=candidate["alpha"], max_iter=20000)
        elif family_name == "huber_linear":
            estimator = HuberRegressor(alpha=candidate["alpha"], epsilon=1.35, max_iter=2000)
        else:
            estimator = QuantileRegressor(quantile=0.5, alpha=candidate["alpha"], solver="highs")
        candidate_pipeline = Pipeline([("prep", make_linear_preprocessor(feature_columns, "asinh")), ("model", estimator)])
        candidate_pipeline.fit(train_frame[feature_columns], train_target)
        validation_score = LOSS_SCORERS[loss_name](validation_target, candidate_pipeline.predict(validation_frame[feature_columns]))
        tuning_rows.append({**candidate, f"validation_{loss_name}_loss": validation_score})
        if validation_score < best_score:
            best_score, best_pipeline, best_params = validation_score, candidate_pipeline, candidate
    return FittedModel(model_name, best_pipeline, feature_columns, best_params, pd.DataFrame(tuning_rows), "regression")


def fit_lightgbm_regressor(train_frame, validation_frame, feature_columns, target_fn, loss_name, model_name,
                           quantile_level=None, fixed_params=None):
    objective_map = {"squared": ("regression", "l2", {}), "absolute": ("regression_l1", "l1", {}),
                     "huber": ("huber", "huber", {"alpha": config.HUBER_DELTA_DOLLARS}),
                     "quantile": ("quantile", "quantile", {"alpha": quantile_level})}
    objective_name, metric_name, objective_extra = objective_map[loss_name]
    train_target, validation_target = target_fn(train_frame), target_fn(validation_frame)
    tuning_rows, best_score, best_model, best_params = [], np.inf, None, None
    for candidate in ([fixed_params] if fixed_params else LIGHTGBM_GRID):
        tree_params = {**objective_extra, **{k: v for k, v in candidate.items() if k != "n_estimators"}}
        use_fixed_trees = bool(fixed_params and "n_estimators" in fixed_params)
        if use_fixed_trees:
            tree_params["n_estimators"] = fixed_params["n_estimators"]
        candidate_model = lgb.LGBMRegressor(**lightgbm_base_params(objective_name, tree_params))
        fit_kwargs = {} if use_fixed_trees else {
            "eval_X": (validation_frame[feature_columns],), "eval_y": (validation_target,), "eval_metric": metric_name,
            "callbacks": [lgb.early_stopping(100, verbose=False)]}
        candidate_model.fit(train_frame[feature_columns], train_target, **fit_kwargs)
        validation_prediction = candidate_model.predict(validation_frame[feature_columns])
        if loss_name == "quantile":
            residual = validation_target - validation_prediction
            validation_score = float(np.mean(np.maximum(quantile_level * residual, (quantile_level - 1) * residual)))
        else:
            validation_score = LOSS_SCORERS[loss_name](validation_target, validation_prediction)
        used_trees = candidate_model.best_iteration_ or candidate_model.n_estimators
        tuning_rows.append({**candidate, "n_estimators": used_trees, f"validation_{loss_name}_loss": validation_score})
        if validation_score < best_score:
            best_score, best_model = validation_score, candidate_model
            best_params = {**candidate, "n_estimators": used_trees}
    return FittedModel(model_name, best_model, feature_columns, best_params, pd.DataFrame(tuning_rows), "regression")


# ---------------------------------------------------------------------- labels, weights, targets
def direction_label(frame):
    return frame["y"].to_numpy()


def three_class_label(frame):
    # So, 0 = large negative (S < -$10), 1 = small (|S| <= $10), 2 = large positive (S > $10).
    large_move = config.THREE_CLASS_LARGE_SPREAD
    return np.where(frame["spread"] > large_move, 2, np.where(frame["spread"] < -large_move, 0, 1))


def make_profit_weight_fn(train_frame):
    # In my opinion a trader cares about $ at stake, so the hour weight is |S|, capped at the training
    # 99th percentile (so a $3,900 hour does not become 1,000 normal hours), then normalised to mean 1.
    abs_cap = train_frame["spread"].abs().quantile(0.99)
    train_mean_weight = np.minimum(train_frame["spread"].abs(), abs_cap).mean()
    return lambda frame: (np.minimum(frame["spread"].abs(), abs_cap) / train_mean_weight).to_numpy()


def spread_target(frame):
    return frame["spread"].to_numpy()


def abs_spread_target(frame):
    return frame["spread"].abs().to_numpy()


# ---------------------------------------------------------------------- the shared prediction store
REGRESSION_SPECS = [
    # (prediction column, view, family, loss)
    ("muB__ridge", "B", "linear", "squared"), ("muB__lasso", "B", "linear", "squared"),
    ("muB__huber_linear", "B", "linear", "huber"), ("muB__lad_linear", "B", "linear", "absolute"),
    ("muB__lgbm_squared", "B", "tree", "squared"), ("muB__lgbm_absolute", "B", "tree", "absolute"),
    ("muB__lgbm_huber", "B", "tree", "huber"),
    ("absA__ridge", "A", "linear", "squared"), ("absA__lad_linear", "A", "linear", "absolute"),
    ("absA__lgbm_squared", "A", "tree", "squared"), ("absA__lgbm_absolute", "A", "tree", "absolute"),
    ("absA__lgbm_huber", "A", "tree", "huber"),
]


def fit_regression_spec(prediction_column, view_name, family_name, loss_name, train_frame, validation_frame,
                        feature_columns, fixed_params=None):
    target_fn = spread_target if view_name == "B" else abs_spread_target
    if family_name == "linear":
        return fit_linear_regressor(train_frame, validation_frame, feature_columns, target_fn, loss_name,
                                    prediction_column, fixed_params)
    return fit_lightgbm_regressor(train_frame, validation_frame, feature_columns, target_fn, loss_name,
                                  prediction_column, fixed_params=fixed_params)


class PredictionStore:
    """Fits every model once (train = 2023, tune = 2024) and caches predictions for all three years."""

    def __init__(self, model_frame, feature_columns=ALL_FEATURE_COLUMNS, use_cache=True, be_verbose=True):
        self._model_frame = model_frame
        self._feature_columns = list(feature_columns)
        self._use_cache = use_cache
        self._be_verbose = be_verbose
        self._train_frame = model_frame[model_frame["split"] == "train"]
        self._validation_frame = model_frame[model_frame["split"] == "validation"]
        self._fitted_models = {}
        signature_text = json.dumps({"features": self._feature_columns, "rows": len(model_frame),
                                     "spread_sum": round(float(model_frame["spread"].sum()), 4), "version": 3})
        self._signature = hashlib.md5(signature_text.encode()).hexdigest()[:10]
        config.ensure_output_folders()

    def _say(self, message):
        if self._be_verbose:
            print(message)

    def _cache_path(self, kind_name):
        return config.OUTPUT_CACHE_DIR / f"{kind_name}_{self._signature}.pkl"

    def get_fitted_models(self):
        return dict(self._fitted_models)

    def _fit_all_classifiers(self):
        train_frame, validation_frame, features = self._train_frame, self._validation_frame, self._feature_columns
        self._say("  fitting logistic regression (C x price transform, chosen by 2024 log loss)")
        logistic_model = fit_logistic(train_frame, validation_frame, features, direction_label, model_name="logistic")
        self._say("  fitting LightGBM classifier (4 tree settings, early stopping on 2024)")
        lightgbm_model = fit_lightgbm_classifier(train_frame, validation_frame, features)
        self._say("  fitting profit-weighted logistic (weight = capped |S|)")
        weighted_model = fit_logistic(train_frame, validation_frame, features, direction_label,
                                      weight_fn=make_profit_weight_fn(train_frame), model_name="weighted_logistic")
        self._say(f"  fitting threshold-label logistic (only hours with |S| > ${config.THRESHOLD_LABEL_MIN_ABS_SPREAD:.0f})")
        is_big_train = train_frame["spread"].abs() > config.THRESHOLD_LABEL_MIN_ABS_SPREAD
        is_big_validation = validation_frame["spread"].abs() > config.THRESHOLD_LABEL_MIN_ABS_SPREAD
        threshold_model = fit_logistic(train_frame[is_big_train], validation_frame[is_big_validation], features,
                                       direction_label, model_name="threshold_label_logistic")
        self._say(f"  fitting three-class logistic (|S| > ${config.THREE_CLASS_LARGE_SPREAD:.0f} is large)")
        three_class_model = fit_logistic(train_frame, validation_frame, features, three_class_label,
                                         model_name="three_class_logistic", transform_grid=("asinh",))
        for fitted in (logistic_model, lightgbm_model, weighted_model, threshold_model, three_class_model):
            self._fitted_models[fitted.get_name()] = fitted
        prediction_frame = pd.DataFrame(index=self._model_frame.index)
        prediction_frame["p__logistic"] = logistic_model.predict(self._model_frame)
        prediction_frame["p__lightgbm"] = lightgbm_model.predict(self._model_frame)
        prediction_frame["p__weighted_logistic"] = weighted_model.predict(self._model_frame)
        prediction_frame["p__threshold_logistic"] = threshold_model.predict(self._model_frame)
        three_class_probability = three_class_model.predict(self._model_frame)
        prediction_frame["p3__large_negative"] = three_class_probability[:, 0]
        prediction_frame["p3__small"] = three_class_probability[:, 1]
        prediction_frame["p3__large_positive"] = three_class_probability[:, 2]
        return prediction_frame

    def _fit_all_regressors(self):
        prediction_frame = pd.DataFrame(index=self._model_frame.index)
        for prediction_column, view_name, family_name, loss_name in REGRESSION_SPECS:
            self._say(f"  fitting {prediction_column:22s} (view {view_name}, {family_name}, {loss_name} loss)")
            fitted = fit_regression_spec(prediction_column, view_name, family_name, loss_name, self._train_frame,
                                         self._validation_frame, self._feature_columns)
            self._fitted_models[prediction_column] = fitted
            prediction_frame[prediction_column] = fitted.predict(self._model_frame)
        for quantile_level in config.QUANTILE_LEVELS:
            quantile_name = f"qB__lgbm_q{int(quantile_level * 100):02d}"
            self._say(f"  fitting {quantile_name:22s} (view B, tree, quantile {quantile_level})")
            fitted = fit_lightgbm_regressor(self._train_frame, self._validation_frame, self._feature_columns,
                                            spread_target, "quantile", quantile_name, quantile_level=quantile_level)
            self._fitted_models[quantile_name] = fitted
            prediction_frame[quantile_name] = fitted.predict(self._model_frame)
        return prediction_frame

    def _cached_or_fit(self, kind_name, fit_method):
        cache_path = self._cache_path(kind_name)
        if self._use_cache and cache_path.exists():
            self._say(f"  using cache {cache_path.name}")
            cached_bundle = pd.read_pickle(cache_path)
            self._fitted_models.update(cached_bundle["models"])
            return cached_bundle["predictions"]
        names_before = set(self._fitted_models)
        prediction_frame = fit_method()
        kind_models = {name: self._fitted_models[name] for name in set(self._fitted_models) - names_before}
        pd.to_pickle({"predictions": prediction_frame, "models": kind_models}, cache_path)
        return prediction_frame

    def get_all_predictions(self):
        # Finally, one wide table: keys + outcomes + every model's prediction for every hour 2023-2025.
        classifier_predictions = self._cached_or_fit("classifier_predictions", self._fit_all_classifiers)
        regression_predictions = self._cached_or_fit("regression_predictions", self._fit_all_regressors)
        key_columns = ["delivery_date", "interval_start_utc", "interval_start_local", "hour_ending", "split",
                       "spread", "y", "spread_std28_d2"]
        combined_frame = pd.concat([self._model_frame[key_columns], classifier_predictions, regression_predictions], axis=1)
        combined_frame.to_csv(config.OUTPUT_PREDICTION_DIR / "all_model_predictions.csv", index=False)
        tuning_frames = [m.get_tuning_table() for m in self._fitted_models.values()]
        pd.concat(tuning_frames, ignore_index=True).to_csv(config.OUTPUT_TABLE_DIR / "model_tuning_grid.csv", index=False)
        chosen_rows = [{"model": name, "chosen_params": json.dumps(m.get_params(), default=str)}
                       for name, m in self._fitted_models.items()]
        pd.DataFrame(chosen_rows).to_csv(config.OUTPUT_TABLE_DIR / "model_chosen_params.csv", index=False)
        return combined_frame
