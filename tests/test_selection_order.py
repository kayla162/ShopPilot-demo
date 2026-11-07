import pandas as pd

from shoppilot.analytics.products import selection_order


def test_regular_first_then_units_then_impressions():
    df = pd.DataFrame({"product_id": ["a", "b", "c", "d"], "is_special": [True, False, False, False],
                       "units": [99, 5, 5, 7], "impressions": [1, 10, 20, 0]})
    assert selection_order(df) == ["d", "c", "b", "a"]


def test_empty():
    assert selection_order(pd.DataFrame(columns=["product_id", "is_special", "units", "impressions"])) == []
