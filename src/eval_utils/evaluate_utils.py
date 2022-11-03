import evaluate


class HFMetric:
    def __init__(self, metric_name, post_process = lambda x : x, **kwargs):
        self.metric = evaluate.load(metric_name)
        self.post_process = post_process
        self.kwargs = kwargs

    def __call__(self, predictions, references):
        return self.post_process(self.metric.compute(predictions=predictions, references=references, **self.kwargs))

class MultiHFMetric:
    def __init__(self, **kwargs):
       self.metrics = kwargs

    def __call__(self, predictions, references):
        return {k: v(predictions=predictions, references=references) for k, v in self.metrics.items()}