import pytest

from shoppilot.transform import normalize as n


@pytest.mark.parametrize("raw, expected", [
    ("23,448", 23448.0), ("18145", 18145.0), (12, 12.0), (3.5, 3.5),
    ("5.05%", 0.0505), ("0.81%", 0.0081), (" 1,523.08 ", 1523.08),
])
def test_to_number_values(raw, expected):
    assert n.to_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["-", "", None, "abc", float("nan"), True])
def test_to_number_missing(raw):
    assert n.to_number(raw) is None


def test_to_int():
    assert n.to_int("1,234") == 1234
    assert n.to_int("2.6") == 3
    assert n.to_int("-") is None


@pytest.mark.parametrize("raw, expected", [
    (11111111101, "11111111101"), (11111111101.0, "11111111101"), (" 123 ", "123"),
    ("", None), (None, None), (float("nan"), None),
])
def test_to_id(raw, expected):
    assert n.to_id(raw) == expected


def test_to_text():
    assert n.to_text("-") is None
    assert n.to_text(" A9992 ") == "A9992"
    assert n.to_text(None) is None


@pytest.mark.parametrize("raw, expected", [
    ("$299", (299.0, 299.0)), ("$250 - $399", (250.0, 399.0)), ("NT$1,290", (1290.0, 1290.0)),
    ("250~399", (250.0, 399.0)), ("$399 - $250", (250.0, 399.0)), (299, (299.0, 299.0)),
    ("$199 $299", (199.0, 199.0)), ("", (None, None)), (None, (None, None)), ("問賣家", (None, None)),
])
def test_parse_price_text(raw, expected):
    assert n.parse_price_text(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("已售出 1.2萬", 12000), ("已售出 3,456", 3456), ("1.2k", 1200), ("10萬+", 100000),
    ("已售出 999+", 999), (345, 345), (12.0, 12), ("", None), (None, None), ("尚無銷量", None),
])
def test_parse_sold_text(raw, expected):
    assert n.parse_sold_text(raw) == expected
