import warnings, openpyxl
import openpyxl.reader.excel as _rx
from openpyxl.worksheet.views import SheetViewList
warnings.filterwarnings("ignore")
_rx.apply_stylesheet = lambda a, w: None           # broken stylesheet (product_diagnostics)
_orig = SheetViewList.from_tree.__func__
def _safe(cls, node):
    try: return _orig(cls, node)
    except Exception: return cls()                  # broken sheetView (mass_update_basic_info)
SheetViewList.from_tree = classmethod(_safe)
def load(f):
    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
    for ws in wb.worksheets:
        ws.reset_dimensions()                       # wrong declared range (mass_update_basic_info)
    return wb
def rows(f, sheet=None):
    wb = load(f)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    out = []
    for r in ws.iter_rows(values_only=True):
        r = list(r)
        while r and r[-1] is None: r.pop()
        out.append(r)
    return out
