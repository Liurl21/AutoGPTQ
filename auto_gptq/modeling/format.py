from enum import Enum

# checkpoint formats
class FORMAT(Enum):
    GPTQ = "gptq"
    GPTQ_V2 = "gptq_v2"
    MARLIN = "marlin"
    AWQ_GEMM = "awq_gemm"