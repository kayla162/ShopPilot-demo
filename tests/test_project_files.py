import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_streamlit_config_is_local_only():
    # 私人版只在這台電腦上使用：區域網路上的其他人不能打開頁面、上傳檔案或用賣家的 Key 呼叫 AI。
    # 公開版（根目錄有 streamlit_app.py）是唯讀的示範資料，要在雲端主機上對外開放，所以不綁 localhost。
    cfg = _load(ROOT / ".streamlit" / "config.toml")
    public = (ROOT / "streamlit_app.py").exists()
    assert cfg.get("server", {}).get("address") == (None if public else "localhost")
    assert cfg["browser"]["gatherUsageStats"] is False


def test_public_streamlit_config_has_no_localhost():
    path = ROOT / "public" / ".streamlit" / "config.toml"
    if not path.exists():           # 公開版本身沒有 public/ 資料夾
        return
    cfg = _load(path)
    assert "address" not in cfg.get("server", {}) and cfg["browser"]["gatherUsageStats"] is False


def test_ci_workflow_matches_docker_python_and_readme_badge():
    # 公開版每次 push 都跑測試：Python 版本和 Docker 映像（以及 Streamlit Community Cloud）一致；只裝 requirements.txt；
    # demo/shoppilot.db 用 Git LFS 存放，checkout 要下載 LFS。README 開頭的徽章指向這個 workflow
    import re

    import yaml
    base = ROOT if (ROOT / "streamlit_app.py").exists() else ROOT / "public"
    wf = yaml.safe_load((base / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
    assert "push" in wf.get("on", wf.get(True))            # PyYAML 把 on: 讀成 True
    steps = wf["jobs"]["pytest"]["steps"]
    by_action = {s["uses"].split("@")[0]: s.get("with", {}) for s in steps if "uses" in s}
    assert by_action["actions/checkout"]["lfs"] is True
    docker = re.search(r"^FROM python:(\d+\.\d+)", (base / "Dockerfile").read_text(encoding="utf-8"), re.M)
    assert by_action["actions/setup-python"]["python-version"] == docker.group(1)
    runs = [s["run"] for s in steps if "run" in s]
    assert runs == ["pip install -r requirements.txt", "python -m pytest -q"]
    head = (base / "README.md").read_text(encoding="utf-8").split("\n## ")[0]
    assert "github.com/kayla162/ShopPilot-demo/actions/workflows/tests.yml/badge.svg" in head


def _requirements(path: Path) -> dict:
    from packaging.requirements import Requirement
    lines = (line.split("#")[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {r.name.lower(): r.specifier for r in (Requirement(line) for line in lines if line)}


def test_public_requirements_are_pinned_within_private_bounds():
    # 公開版（雲端 demo、CI、Docker）用固定版本，雲端重新部署才不會裝到不相容的新版。私人版只寫下限：
    # 兩邊是同一組套件，固定的版本要滿足私人版的下限（私人版提高下限時，公開版要跟著升級）
    public = (ROOT / "streamlit_app.py").exists()
    pinned = _requirements((ROOT if public else ROOT / "public") / "requirements.txt")
    assert pinned and all(len(s) == 1 and next(iter(s)).operator == "==" for s in pinned.values()), pinned
    if public:                      # 公開版本身沒有私人版的 requirements.txt 可以比對
        return
    bounds = _requirements(ROOT / "requirements.txt")
    assert bounds.keys() == pinned.keys()
    for name, spec in pinned.items():
        assert bounds[name].contains(next(iter(spec)).version), name


def test_external_competitors_included_by_default():
    # 2026-10-16 用賣家填好的範本校準：確認是同款的外部競品，標題相似度只有 0.02–0.14，所以預設納入、填「否」才排除
    from shoppilot import config
    assert config.load_yaml("rules")["peers"]["external_threshold"] == 0
