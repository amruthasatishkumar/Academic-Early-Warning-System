"""Exact underlying linear-score explanations using the training population as reference."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


OUTCOMES = ("excellent", "pass", "fail", "withdrawn")


def fitted_linear_pipeline(model):
    if isinstance(model, Pipeline):
        pipeline = model
    else:
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated is None or len(calibrated) != 1:
            raise ValueError("Expected the saved single four-class model, not a model ensemble.")
        member = calibrated[0]
        pipeline = getattr(member, "estimator", None)
        if pipeline is None:
            pipeline = getattr(member, "base_estimator", None)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("The saved model does not contain the expected fitted feature pipeline.")
    if not isinstance(pipeline.named_steps.get("classifier"), LogisticRegression):
        raise TypeError("Exact linear-score explanations require the saved logistic classifier.")
    if set(pipeline.classes_) != set(OUTCOMES):
        raise ValueError("The model must predict exactly the four agreed outcomes.")
    return pipeline


class OutcomeExplanations:
    def __init__(self, model, training):
        self.model = model
        self.pipeline = fitted_linear_pipeline(model)
        self.features = list(self.pipeline.feature_names_in_)
        if training.empty or not set(self.features) <= set(training.columns):
            raise ValueError("The reference must contain the actual training rows and model features.")
        self.classes = list(self.pipeline.classes_)
        self.coefficients = self.pipeline.named_steps["classifier"].coef_
        self.intercepts = self.pipeline.named_steps["classifier"].intercept_
        self.transformed_names = list(self.pipeline.named_steps["imputer"].get_feature_names_out(self.features))
        self.groups = [self.features.index(name.removeprefix("missingindicator_")) for name in self.transformed_names]
        self.training_z = self.transform(training)
        self.reference = self.training_z.mean(axis=0)
        self.baseline_scores = self.reference @ self.coefficients.T + self.intercepts

    def transform(self, frame):
        values = frame.loc[:, self.features].astype(float)
        if np.isinf(values.to_numpy()).any():
            raise ValueError("Infinite model feature encountered.")
        result = np.asarray(self.pipeline[:-1].transform(values), dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("The saved feature pipeline produced nonfinite values.")
        return result

    def contributions(self, frame, outcome):
        if outcome not in self.classes:
            raise ValueError(f"Unknown outcome: {outcome}")
        transformed = self.transform(frame)
        index = self.classes.index(outcome)
        terms = (transformed - self.reference) * self.coefficients[index]
        grouped = np.zeros((len(frame), len(self.features)))
        for position, group in enumerate(self.groups):
            grouped[:, group] += terms[:, position]
        expected = self.pipeline.decision_function(frame.loc[:, self.features].astype(float))[:, index]
        np.testing.assert_allclose(self.baseline_scores[index] + grouped.sum(axis=1), expected, atol=1e-9)
        return pd.DataFrame(grouped, index=frame.index, columns=self.features)

    def training_importance(self, training):
        rows = []
        for outcome in OUTCOMES:
            values = self.contributions(training, outcome).abs().mean()
            for feature, magnitude in values.items():
                rows.append({"outcome": outcome, "feature": feature, "mean_absolute_score_contribution": float(magnitude)})
        return pd.DataFrame(rows)

    def result_drivers(self, current, predictions):
        if len(current) != len(predictions):
            raise ValueError("Current features and predictions must be aligned one-to-one.")
        expected = self.model.predict_proba(current.loc[:, self.features].astype(float))
        model_classes = list(self.model.classes_)
        for outcome in OUTCOMES:
            np.testing.assert_allclose(expected[:, model_classes.index(outcome)],
                                       predictions["p_" + outcome].to_numpy(), rtol=1e-5, atol=1e-7)
        predicted = np.asarray(model_classes)[expected.argmax(axis=1)]
        if not np.array_equal(predicted, predictions.predicted_class.to_numpy()):
            raise ValueError("The saved model does not reproduce the published outcome labels.")
        drivers = [None] * len(current)
        for outcome in OUTCOMES:
            indices = np.flatnonzero(predicted == outcome)
            if not len(indices):
                continue
            terms = self.contributions(current.iloc[indices], outcome).to_numpy()
            for position, row_index in enumerate(indices):
                top = np.argsort(np.abs(terms[position]))[::-1][:3]
                drivers[row_index] = [
                    {"feature": self.features[i], "score_contribution": float(terms[position, i]),
                     "direction": "raises" if terms[position, i] > 0 else "lowers" if terms[position, i] < 0 else "neutral"}
                    for i in top
                ]
        return drivers
