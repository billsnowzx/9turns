# run_hstech30.py
import re
import subprocess
import sys
from pathlib import Path

import requests

URLS = [
    "https://m.aastocks.com/sc/stocks/market/index/hk-index-con.aspx?index=HSTECH",
    "https://www.aastocks.com/sc/stocks/market/index/hk-index-con.aspx?index=HSTECH",
]

# 兜底名单（HSTECH常见30只代码，按5位港股格式）
FALLBACK_30 = [
    "00020.HK","00241.HK","00268.HK","00285.HK","00300.HK","00700.HK","00780.HK","00981.HK","00992.HK","01024.HK",
    "01211.HK","01347.HK","01698.HK","01810.HK","02015.HK","02382.HK","03690.HK","03888.HK","06618.HK","06690.HK",
    "09618.HK","09626.HK","09660.HK","09863.HK","09866.HK","09868.HK","09888.HK","09961.HK","09988.HK","09999.HK",
]

def fetch_hstech_codes():
    all_codes = set()
    headers = {"User-Agent": "Mozilla/5.0"}
    patterns = [
        r'(\d{5}\.HK)',
        r'code[=/:"\']+(\d{5})',  # 兼容不同页面字段
    ]
    for u in URLS:
        try:
            html = requests.get(u, headers=headers, timeout=20).text
            for p in patterns:
                for x in re.findall(p, html, flags=re.IGNORECASE):
                    if isinstance(x, tuple):
                        x = x[0]
                    if len(x) == 5 and x.isdigit():
                        all_codes.add(f"{x}.HK")
                    elif isinstance(x, str) and x.endswith(".HK") and len(x) == 8:
                        all_codes.add(x.upper())
        except Exception:
            pass

    # 过滤异常代码后排序
    codes = sorted(c for c in all_codes if re.fullmatch(r"\d{5}\.HK", c))
    if len(codes) >= 25:
        return codes[:30]  # 抓太多时只取前30（页面一般就是30）
    return FALLBACK_30

def run_one(symbol: str):
    cmd = [
        sys.executable, "-c",
        (
            "from main import run_full_research; "
            f"run_full_research(market='hk', symbol='{symbol}', "
            "freq='daily', start='2018-01-01', end='2026-04-28')"
        ),
    ]
    return subprocess.run(cmd, check=False).returncode

def main():
    codes = fetch_hstech_codes()
    print(f"Using {len(codes)} symbols")
    out = Path("output/hstech30_runs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = ["symbol,ok"]
    for s in codes:
        print(f"Running {s} ...")
        rc = run_one(s)
        rows.append(f"{s},{1 if rc == 0 else 0}")

    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"Done. Saved: {out}")

if __name__ == "__main__":
    main()
