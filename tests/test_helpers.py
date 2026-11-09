import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alpr.pipeline import OCR

def test_normalize_plate_text():
    assert OCR.normalize("ka 03 AB-1234") == "KA03AB1234"
