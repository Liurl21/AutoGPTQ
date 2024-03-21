import tempfile
import unittest

from parameterized import parameterized
from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.modeling.format import FORMAT

class TestQuantization(unittest.TestCase):
    @parameterized.expand([(False,), (True,)])
    def test_quantize(self, use_marlin: bool):
        pretrained_model_dir = "saibo/llama-1B"

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
        examples = [
            tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]
        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            checkpoint_format= FORMAT.MARLIN if use_marlin else FORMAT.GPTQ,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            pretrained_model_dir,
            quantize_config=quantize_config,
            use_flash_attention_2=False,
        )

        model.quantize(examples)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

            model = AutoGPTQForCausalLM.from_quantized(tmpdirname, device="cuda:0", use_marlin=use_marlin)
