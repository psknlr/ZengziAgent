#!/usr/bin/env python3
"""Execute the Colab notebook's code cells offline (mock provider) without Jupyter.

    python scripts/test_notebook_offline.py [--workdir DIR] [--notebook notebooks/ZengziAgent_Colab.ipynb]

Cells are executed in one namespace; `!cmd` lines become subprocess calls; PROVIDER is forced to
"mock" and IPython display helpers are stubbed when IPython is unavailable.  Use a scratch clone as
--workdir so that results do not land in your working tree.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import types


def transform(src: str) -> str:
    out = []
    for line in src.splitlines():
        m = re.match(r"^(\s*)!(.*)$", line)
        if m:
            out.append(f"{m.group(1)}subprocess.run({m.group(2).strip()!r}, shell=True, check=True)")
        elif line.startswith("%"):
            continue
        else:
            out.append(line)
    code = "\n".join(out)
    code = re.sub(r'^PROVIDER\s*=\s*"[^"]+"', 'PROVIDER = "mock"', code, flags=re.M)
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="notebooks/ZengziAgent_Colab.ipynb")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    nb = json.load(open(args.notebook, encoding="utf-8"))
    if args.workdir:
        os.environ["ZENGZI_WORKDIR"] = os.path.abspath(args.workdir)
    os.environ["ZENGZI_PROVIDER"] = "mock"  # the notebook honours this override; no API key needed
    try:
        import IPython.display  # noqa: F401
    except ImportError:
        ipy = types.ModuleType("IPython")
        disp = types.ModuleType("IPython.display")
        disp.display = lambda *a, **k: print(*[str(x)[:2000] for x in a])
        disp.Markdown = lambda s: s
        disp.Image = lambda p, **k: f"<Image {p}>"
        ipy.display = disp
        sys.modules["IPython"] = ipy
        sys.modules["IPython.display"] = disp
    ns: dict = {"__name__": "__main__", "subprocess": subprocess}
    n_code = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        code = transform(src)
        print(f"\n===== cell {i} =====\n{code[:300]}{'...' if len(code) > 300 else ''}\n-----")
        t0 = time.time()
        exec(compile(code, f"<cell {i}>", "exec"), ns)
        print(f"----- cell {i} ok ({time.time() - t0:.0f}s)")
    print(f"\nALL {n_code} CODE CELLS EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
