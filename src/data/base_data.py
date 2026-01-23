class BaseDataTask:
    def __init__(self, config, device="cuda"):
        self.n_train = config.n_train
        self.n_test = config.n_test
        self.device = device

    def sample(self, num_samples, num_tokens):
        raise NotImplementedError("Subclasses must implement the `sample` method.")