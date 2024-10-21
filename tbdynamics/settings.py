from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"
VN_PATH = BASE_PATH / "tbdynamics/vietnam"
CM_PATH = BASE_PATH / "tbdynamics/camau"
INPUT_PATH = DATA_PATH / "inputs"
OUT_PATH = DATA_PATH / 'outputs'
DOCS_PATH = BASE_PATH / 'docs'