"""Plotly 圖表。規則：只用參考色票第 1 色＋灰色做強調；單一系列不放圖例；格線 1px 實線；長條圓角 4px。"""
from __future__ import annotations

import plotly.graph_objects as go

PALETTE = {
    False: {"accent": "#2a78d6", "context": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    True: {"accent": "#3987e5", "context": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
FONT = 'system-ui, -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif'


def _layout(fig: go.Figure, dark: bool, height: int) -> go.Figure:
    p = PALETTE[dark]
    fig.update_layout(height=height, margin=dict(l=8, r=24, t=8, b=8), font=dict(family=FONT), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_family=FONT))
    fig.update_xaxes(gridcolor=p["grid"], linecolor=p["axis"], gridwidth=1, zeroline=False)
    fig.update_yaxes(gridcolor=p["grid"], linecolor=p["axis"], gridwidth=1, zeroline=False)
    return fig


def price_histogram(prices: list[float], avg: float, dark: bool = False) -> go.Figure:
    p = PALETTE[dark]
    fig = go.Figure(go.Histogram(x=prices, xbins=dict(size=50), marker=dict(color=p["accent"], cornerradius=4),
                                 hovertemplate="價格 %{x}<br>商品數 %{y}<extra></extra>"))
    fig.add_vline(x=avg, line_width=1, line_color=p["context"], annotation_text=f"平均 ${avg:,.0f}",
                  annotation_position="top right")
    fig.update_layout(bargap=0.06)
    fig.update_xaxes(title="價格（元）")
    fig.update_yaxes(title="商品數")
    return _layout(fig, dark, 300)


def hbar(labels: list[str], values: list[float], unit: str, highlight: str | None = None,
         dark: bool = False) -> go.Figure:
    """水平長條圖。有 highlight 時只有那一條用強調色，其他用灰色；數值標在長條尾端。"""
    p = PALETTE[dark]
    colors = [p["accent"] if highlight is None or name == highlight else p["context"] for name in labels]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[f"{v:,.0f}{unit}" for v in values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}：%{x:,.0f}" + unit + "<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(bargap=0.4)
    return _layout(fig, dark, max(160, 34 * len(labels) + 40))


def box_by_group(groups: dict[str, list[float]], y_title: str, log_y: bool = False, dark: bool = False) -> go.Figure:
    p = PALETTE[dark]
    fig = go.Figure()
    for name, values in groups.items():
        fig.add_trace(go.Box(y=values, name=name, marker_color=p["accent"], line=dict(width=2), boxpoints="outliers"))
    fig.update_yaxes(title=y_title, type="log" if log_y else "linear")
    return _layout(fig, dark, 340)
