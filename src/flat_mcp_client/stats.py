from types import SimpleNamespace
import numpy as np
#import polars as pl
from flat_mcp_client import debug_pp


class ModelStats:
    """Tracking statistics about context length"""

    def __init__(self, model_metadata: dict):
        self.model_metadata = model_metadata

        # vectors of inference metrics, one element per inference call
        self.inference = {
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
        # derived statistics
        self.stats: dict = {}

    def append(self, **kwargs) -> None:
        """Append measurements from one inference result to `inference` vector"""
        for key, value in kwargs.items():
            self.inference[key].append(value)
        self._n += 1

    def compute(self) -> dict:
        """Compute and return simple stats across all inference calls"""
        debug_pp(self.inference)
        if self._n > 0:
            # numpy arrays for efficient operations
            self.inference_np = {}
            for key, value in self.inference.items():
                self.inference_np[key] = np.array(value)
            # tokens / second generation
            self.stats = {
                "average_ttft": float(np.mean(self.inference_np["time_to_first_token"])),
                "average_ttfnt": float(np.mean(self.inference_np["time_to_first_nonthinking_token"])),
                "input_tps": float(np.sum(self.inference_np["num_input_tokens"]) / np.sum(self.inference_np["prompt_parsing_time"])),
                "output_tps": float(np.sum(self.inference_np["num_output_tokens"]) / np.sum(self.inference_np["generation_time"])),
                "total_inference_time": float(np.sum(self.inference_np["response_time"])),
            }
            # create object reference-able with dot notation (e.g., `overall.ttnt`)
            self.overall = SimpleNamespace(**self.stats)
        return self.stats
