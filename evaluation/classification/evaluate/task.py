class RuGPTEvaluationTask:
    def __init__(self, config):
        self.config = config

    def verbalize_samples(self, lang, dataset, prompt):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def calculate_scores(self, data, model):
        raise NotImplementedError

    def predict(self, model):
        y_true = {}
        y_pred = {}
        return y_true, y_pred
