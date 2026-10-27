from dashboard import charts


def test_hbar_highlights_one_bar():
    fig = charts.hbar(["我的商品", "競品平均"], [599, 649], " 元", highlight="我的商品")
    assert list(fig.data[0].marker.color) == [charts.PALETTE[False]["accent"], charts.PALETTE[False]["context"]]
    assert fig.layout.showlegend is False and fig.data[0].marker.cornerradius == 4


def test_price_histogram_and_box():
    hist = charts.price_histogram([199, 299, 350], 282.7)
    assert hist.data[0].type == "histogram" and hist.data[0].marker.cornerradius == 4
    fig = charts.box_by_group({"現貨": [1, 2, 3], "預購": [1, 2]}, "曝光", log_y=True, dark=True)
    assert len(fig.data) == 2 and fig.layout.yaxis.type == "log"
    assert fig.data[0].marker.color == charts.PALETTE[True]["accent"]
