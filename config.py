"""
Configuration file for PDF Image-Text Separation Pipeline
"""
import os

# Paths
OUTPUT_DIR = "output"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
RENDER_DIR = os.path.join(OUTPUT_DIR, "renders")

# Layout Service Configuration (HTTP client)
# Layout detection is done via HTTP call to layout-service running in Linux/Docker
# Default uses Docker host gateway for separately run containers.
LAYOUT_SERVICE_URL = os.getenv('LAYOUT_SERVICE_URL', 'http://host.docker.internal:8001')

# Shared Volume Configuration
# Path contract: All cross-service file paths use shared volume absolute paths
SHARED_VOLUME_ROOT = os.getenv('SHARED_VOLUME_ROOT', '/app/shared_data')

# Layout Service Timeout Configuration
LAYOUT_CONNECT_TIMEOUT = float(os.getenv('LAYOUT_CONNECT_TIMEOUT', '5'))  # seconds
LAYOUT_READ_TIMEOUT = float(os.getenv('LAYOUT_READ_TIMEOUT', '30'))  # seconds
LAYOUT_MAX_RETRIES = int(os.getenv('LAYOUT_MAX_RETRIES', '2'))  # number of retries

# codex update: PaddleOCR service configuration (HTTP client)
PADDLE_SERVICE_URL = os.getenv('PADDLE_SERVICE_URL', 'http://host.docker.internal:8002')
PADDLE_CONNECT_TIMEOUT = float(os.getenv('PADDLE_CONNECT_TIMEOUT', '5'))
PADDLE_READ_TIMEOUT = float(os.getenv('PADDLE_READ_TIMEOUT', '60'))
PADDLE_MAX_RETRIES = int(os.getenv('PADDLE_MAX_RETRIES', '2'))
PADDLE_CROP_DIR = os.getenv('PADDLE_CROP_DIR', os.path.join(OUTPUT_DIR, 'paddle_crops'))
PADDLE_KEEP_CROPS = os.getenv('PADDLE_KEEP_CROPS', 'false').lower() == 'true'

# codex update: inference service configuration (HTTP client)
INFERENCE_SERVICE_URL = os.getenv('INFERENCE_SERVICE_URL', 'http://host.docker.internal:8003')
INFERENCE_CONNECT_TIMEOUT = float(os.getenv('INFERENCE_CONNECT_TIMEOUT', '5'))
INFERENCE_READ_TIMEOUT = float(os.getenv('INFERENCE_READ_TIMEOUT', '120'))
INFERENCE_MAX_RETRIES = int(os.getenv('INFERENCE_MAX_RETRIES', '1'))

# OCR config
OCR_LANG = "eng+chi_sim"  # English + Simplified Chinese

# Region Refiner thresholds (configurable hyperparameters)
# These are used to distinguish text blocks from image regions
MIN_TEXT_CHARS = 50  # Minimum total characters to consider as text region
MIN_TEXT_LINES = 3  # Minimum lines (after clustering) to consider as text region
TEXT_ALIGNMENT_THRESHOLD = 0.7  # Alignment score threshold (0-1)
LINE_CLUSTERING_TOLERANCE = 10  # Pixels tolerance for grouping words into same line
MIN_IMAGE_AREA = 1000  # Minimum area (pixels) to consider as image

# IOU threshold for image deduplication
IOU_THRESHOLD = 0.8

# Caption detection
CAPTION_PATTERNS = [
    r"^Figure\s+\d+",
    r"^Table\s+\d+",
    r"^Elevation\s+\d+",
    r"^Fllorplan\s+\d+",


]
OCR_LANG = "en"
LINE_CLUSTERING_TOLERANCE = 12
MIN_TEXT_CHARS = 30
MIN_TEXT_LINES = 2
TEXT_ALIGNMENT_THRESHOLD = 0.5


DEBUG_REFINE_DIR = "debug_refine"  


BIG_REGION_AREA_RATIO = 0.35


INJECT_FULL_PAGE_CANDIDATE = False

OCR_CLAHE_CLIP = 2.0
OCR_CLAHE_GRID = (8, 8)
OCR_ADAPTIVE_BLOCK_SIZE = 31
OCR_ADAPTIVE_C = 10
OCR_AUTO_INVERT = True
OCR_INVERT_WHITE_RATIO_THRESHOLD = 0.5

MIN_REGION_SIDE_PX = 0

# codex update: page keyword gate config (floorplan detector)
PAGE_KEYWORDS = ["floor plan", "floorplan"]
PAGE_KEYWORD_LANG = "en"
GATE_USE_OCR = True
GATE_MAX_SIDE = 3840
GATE_OCR_PARAMS = dict(
    lang=PAGE_KEYWORD_LANG,
    use_textline_orientation=True,
    text_det_limit_type="max",
    text_det_limit_side_len=GATE_MAX_SIDE,
    text_det_thresh=0.2,
    text_det_box_thresh=0.3,
    text_det_unclip_ratio=1.8,
)

# codex update: OCR gate preprocessing (align with floorplan script)
KW_STRIP_Y0_RATIO = 0.80
KW_OCR_MAX_SIDE = 3840
KW_OCR_MIN_UPSCALE = 1.0
KW_OCR_MAX_UPSCALE = 4.0
GATE_PREPROCESS_DIR = os.getenv('GATE_PREPROCESS_DIR', os.path.join(OUTPUT_DIR, 'gate_preprocess'))
GATE_KEEP_PREPROCESS = os.getenv('GATE_KEEP_PREPROCESS', 'false').lower() == 'true'

# codex update: page-level visual fallback for keyword gate
GATE_USE_VISUAL_FALLBACK = os.getenv('GATE_USE_VISUAL_FALLBACK', 'true').lower() == 'true'
GATE_VIS_MAX_SIDE = int(os.getenv('GATE_VIS_MAX_SIDE', '2000'))
GATE_VIS_EDGE_DENSITY_MIN = float(os.getenv('GATE_VIS_EDGE_DENSITY_MIN', '0.010'))
GATE_VIS_COLOR_MAX = float(os.getenv('GATE_VIS_COLOR_MAX', '0.30'))

# codex update: PDF render settings
RENDER_DPI = int(os.getenv("RENDER_DPI", "200"))

# codex update: blueprint visual prefilter
BLUEPRINT_FILTER_ENABLED = True
BLUEPRINT_MIN_AREA_RATIO = 0.35
BLUEPRINT_EDGE_DENSITY_MIN = 0.010
BLUEPRINT_COLOR_MAX = 0.30

# codex update: door detection (YOLO)
DOOR_DETECT_ENABLED = os.getenv("DOOR_DETECT_ENABLED", "true").lower() == "true"
DOOR_MODEL_PATH = os.getenv("DOOR_MODEL_PATH", "")
DOOR_CONF_THRESHOLD = float(os.getenv("DOOR_CONF_THRESHOLD", "0.25"))
DOOR_IOU_THRESHOLD = float(os.getenv("DOOR_IOU_THRESHOLD", "0.45"))
DOOR_CLASS_NAMES = [name.strip().lower() for name in os.getenv("DOOR_CLASS_NAMES", "door").split(",") if name.strip()]
DOOR_MIN_COUNT = int(os.getenv("DOOR_MIN_COUNT", "2"))

# codex update: candidate padding expansion
CANDIDATE_PAD_RATIO = float(os.getenv("CANDIDATE_PAD_RATIO", "0.2"))
CANDIDATE_MIN_PAD = int(os.getenv("CANDIDATE_MIN_PAD", "32"))
CANDIDATE_MAX_PAD = int(os.getenv("CANDIDATE_MAX_PAD", "512"))

# codex update: table keyword extraction
TABLE_KEYWORDS = [
    "area schedule",
    "square",
    "floor area's (sq.ft.)",
    "sq.ft.",
]

# codex update: test output root (ensure shared volume visibility)
TEST_OUTPUT_DIR = os.getenv('TEST_OUTPUT_DIR', os.path.join(OUTPUT_DIR, 'test_outputs'))

# codex update: candidate preprocessing (align with floorplan script)
CANDIDATE_MIN_W = 12
CANDIDATE_MIN_H = 12
CANDIDATE_OVERLAP_TH = 0.30
CANDIDATE_MIN_AREA_RATIO = 0.2
SIDEBAR_PARAMS = dict(side_margin_ratio=0.12, min_height_ratio=0.55, max_width_ratio=0.22)
