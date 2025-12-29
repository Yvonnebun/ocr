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
LAYOUT_SERVICE_URL = os.getenv('LAYOUT_SERVICE_URL', 'http://localhost:8001')

# Shared Volume Configuration
# Path contract: All cross-service file paths use shared volume absolute paths
SHARED_VOLUME_ROOT = os.getenv('SHARED_VOLUME_ROOT', '/app/shared_data')

# Layout Service Timeout Configuration
LAYOUT_CONNECT_TIMEOUT = float(os.getenv('LAYOUT_CONNECT_TIMEOUT', '5'))  # seconds
LAYOUT_READ_TIMEOUT = float(os.getenv('LAYOUT_READ_TIMEOUT', '30'))  # seconds
LAYOUT_MAX_RETRIES = int(os.getenv('LAYOUT_MAX_RETRIES', '2'))  # number of retries

# OCR config
OCR_LANG = "eng+chi_sim"  # English + Simplified Chinese
TESSERACT_CMD = "C:/Program Files/Tesseract-OCR/tesseract.exe" # Auto-detect, or set path if needed

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
