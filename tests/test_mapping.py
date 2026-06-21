import json
from pathlib import Path

from src.mapping import CODES, GROUPS, group_of, is_dropped

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def codes_in_data() -> set[str]:
    codes = set()
    for split in ("train", "dev", "test"):
        with open(DATA_DIR / f"{split}_v1.jsonl", encoding="utf-8") as f:
            for line in f:
                codes.add(json.loads(line)["code"])
    return codes


def test_all_data_codes_are_mapped():
    unmapped = codes_in_data() - set(CODES)
    assert not unmapped, unmapped


def test_eight_groups():
    assert len(GROUPS) == 8


def test_drop_codes():
    assert is_dropped("D50")
    assert is_dropped("H65")
    assert is_dropped("Z00")
    assert is_dropped("F41")


def test_known_mappings():
    assert group_of("M54") == "ПОЗВОНОЧНИК"
    assert group_of("I11") == "КАРДИО+НЕВРО"
    assert group_of("G20") == "КАРДИО+НЕВРО"
    assert not is_dropped("M54")
