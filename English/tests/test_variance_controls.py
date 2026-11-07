import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from model.modules import LengthRegulator, VarianceAdaptor


class DurationPredictorStub(nn.Module):
    def forward(self, x, mask):
        return torch.zeros(x.shape[:2], device=x.device)


class VarianceControlTests(unittest.TestCase):
    def test_energy_control_is_independent_of_pitch_control(self):
        adaptor = VarianceAdaptor.__new__(VarianceAdaptor)
        nn.Module.__init__(adaptor)
        adaptor.duration_predictor = DurationPredictorStub()
        adaptor.length_regulator = LengthRegulator()

        x = torch.zeros(1, 2, 4)
        prediction = torch.zeros(1, 2)
        embedding = torch.zeros_like(x)
        for feature_level in ("phoneme_level", "frame_level"):
            adaptor.pitch_feature_level = feature_level
            adaptor.energy_feature_level = feature_level
            with patch.object(
                adaptor,
                "get_pitch_embedding",
                return_value=(prediction, embedding),
            ) as pitch_embedding:
                with patch.object(
                    adaptor,
                    "get_energy_embedding",
                    return_value=(prediction, embedding),
                ) as energy_embedding:
                    adaptor(
                        x,
                        torch.zeros(1, 2, dtype=torch.bool),
                        duration_target=torch.ones(1, 2, dtype=torch.long),
                        p_control=0.5,
                        e_control=2.0,
                    )

            self.assertEqual(pitch_embedding.call_args.args[3], 0.5)
            self.assertEqual(energy_embedding.call_args.args[3], 2.0)


if __name__ == "__main__":
    unittest.main()
