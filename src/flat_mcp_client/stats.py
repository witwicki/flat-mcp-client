from types import SimpleNamespace
import numpy as np
#import polars as pl
from flat_mcp_client import debug_pp


class ModelStats:
    """Tracking statistics about context length"""

    def __init__(self, model_metadata: dict[str, object]):
        self.model_metadata: dict[str, object] = model_metadata

        # vectors of inference metrics, one element per inference call
        self.inference: dict[str, list[float]] = {
            "time_to_first_token": [],
            "time_to_first_nonthinking_token": [],
            "prompt_parsing_time": [],
            "generation_time": [],
            "response_time": [],
            "num_input_tokens": [],
            "num_output_tokens": [],
        }
        # number of inference calls
        self._n: int = 0
        # derived statistics (computed in compute method)
        self.stats: dict[str, float] = {}
        self.inference_np: dict[str, np.ndarray] = {}
        self.overall: SimpleNamespace = SimpleNamespace()

    def append(self, **kwargs: float) -> None:
        """Append measurements from one inference result to `inference` vector"""
        for key, value in kwargs.items():
            self.inference[key].append(value)
        self._n += 1

    def compute(self) -> dict[str, float]:
        """Compute and return simple stats across all inference calls"""
        debug_pp(self.inference)
        if self._n > 0:
            # numpy arrays for efficient operations
            self.inference_np = {
                key: np.array(value, dtype=float) for key, value in self.inference.items()
            }
            # simple python stats
            input_tokens: float = sum(self.inference["num_input_tokens"])
            parsing_time: float = sum(self.inference["prompt_parsing_time"])
            output_tokens: float = sum(self.inference["num_output_tokens"])
            generation_time: float = sum(self.inference["generation_time"])
            average_ttft = sum(self.inference["time_to_first_token"]) / self._n
            average_ttfnt = sum(self.inference["time_to_first_nonthinking_token"]) / self._n
            total_inference_time = sum(self.inference["response_time"])
            self.stats = {
                "average_ttft": average_ttft,
                "average_ttfnt": average_ttfnt,
                "input_tps": float(input_tokens / parsing_time),
                "output_tps": float(output_tokens / generation_time),
                "total_inference_time": total_inference_time,
            }
            # create object reference-able with dot notation (e.g., `overall.ttnt`)
            self.overall = SimpleNamespace(**self.stats)
        return self.stats
