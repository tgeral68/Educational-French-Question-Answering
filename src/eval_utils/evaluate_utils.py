import evaluate
import os

class HFMetric:
    def __init__(self, metric_name, post_process = lambda x : x, **kwargs):
        if "HF_HOME" in os.environ:
            metric_name = os.path.join(os.path.expandvars("$HF_HOME"),"evaluate/downloads", metric_name + ".py")
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