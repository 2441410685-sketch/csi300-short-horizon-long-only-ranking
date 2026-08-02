"""项目路径与固定研究口径。"""

import os
from pathlib import Path


ROOT = Path(
    os.environ.get("CSI300_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"

RAW_PRICE_PATH = RAW_DIR / "hs300_daily_raw.csv"
MEMBERSHIP_PATH = RAW_DIR / "hs300_membership_daily.csv"
CLEAN_PRICE_PATH = PROCESSED_DIR / "prices_daily_clean.csv"
SAMPLE_PATH = PROCESSED_DIR / "samples_with_label.csv"
FEATURE_PATH = ARTIFACT_DIR / "features_float32.parquet"
MODEL_PATH = MODEL_DIR / "final_model.txt"
MODEL_MANIFEST_PATH = MODEL_DIR / "model_manifest.json"
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.csv"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
LATEST_SELECTION_PATH = OUTPUT_DIR / "latest_selection.csv"

DATE = "date"
STOCK = "stock_id"
LABEL = "label_ret_t1_t5"

SEED = 2026
TOP_K = 5
HORIZON_DAYS = 5
GAP_DAYS = 5
RELEVANCE_LEVELS = 10
INNER_VALID_DAYS = 120
TRAIN_START_DATE = "2022-08-01"
DOWNLOAD_START_DATE = "2022-05-01"
MAX_BOOST_ROUNDS = 400
MIN_SELECT_ROUND = 20
SMOOTH_ROUNDS = 5


def ensure_directories() -> None:
    """创建运行时需要、但 Git 不保存内容的目录。"""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        ARTIFACT_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
