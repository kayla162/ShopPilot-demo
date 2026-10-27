import pandas as pd
import pytest

from shoppilot.analytics import inventory as inv

CFG = {"preorder_bases": [10, 100, 300, 500], "preorder_window_bases": [100, 300, 500], "preorder_window": 10}


@pytest.mark.parametrize("stock, name, expected", [
    (0, "咖色-預購,XL", False), (10, "咖色-預購,XL", True), (10, "黑色", False), (96, "黑色", True),
    (100, "", True), (90, "", False), (91, "", True), (297, "", True), (290, "", False), (498, None, True),
    (5, "", False), (7, "現貨", False), (9975199, "", False), (None, "預購", False), (float("nan"), "預購", False),
])
def test_is_preorder_sku(stock, name, expected):
    assert inv.is_preorder_sku(stock, name, CFG) is expected


@pytest.mark.parametrize("stock, expected", [(96, 4), (99, 1), (100, 0), (297, 3), (498, 2), (10, 0), (0, 0), (600, 0), (5, 5)])
def test_preorder_sold(stock, expected):
    assert inv.preorder_sold(stock, CFG) == expected


def test_product_inventory():
    skus = pd.DataFrame({"product_id": ["P", "P", "P", "Q"], "variation_name": ["黑", "預購-白", "灰", "F"],
                         "stock": [2, 97, 0, "3"]})
    table = inv.product_inventory(inv.sku_inventory(skus, CFG)).set_index("product_id")
    assert table.loc["P"].to_dict() == {"preorder_skus": 1, "preorder_sold": 3, "instock_units": 2}
    assert table.loc["Q"].to_dict() == {"preorder_skus": 0, "preorder_sold": 0, "instock_units": 3}
