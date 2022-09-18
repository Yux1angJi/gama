from river import base, linear_model, naive_bayes, tree
from river.ensemble import VotingClassifier
from typing import List
import typing


class VotingPipeline(VotingClassifier):
    def __init__(self, models: List[base.Classifier]):
        super().__init__(models)

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        num_models = len(self.models)
        proba = {}
        for model in self.models:
            proba_i = model.predict_proba_one(x)
            for k, v in proba_i.items():
                proba[k] = v + proba.get(k, 0.0)
        for k, v in proba.items():
            proba[k] = v / num_models
        return proba

