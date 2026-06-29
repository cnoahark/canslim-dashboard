#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANSLIM Dashboard - Data Refresher
===================================
Saves TDX sector K-line data to JSON files.
Run this daily after market close to update the dashboard.

Requires: tdx-connector MCP is connected
Usage: python refresh_data.py
"""

import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# All sector codes to track
SECTORS = {
    "881319": {"name": "半导体", "level1": "电子", "setcode": "1"},
    "881333": {"name": "元器件", "level1": "电子", "setcode": "1"},
    "881338": {"name": "通信设备", "level1": "通信", "setcode": "1"},
    "881015": {"name": "化工", "level1": "化工", "setcode": "1"},
    "881125": {"name": "电力设备", "level1": "电力设备", "setcode": "1"},
    "881007": {"name": "机械设备", "level1": "机械设备", "setcode": "1"},
    "881258": {"name": "IT设备", "level1": "计算机", "setcode": "1"},
    "881160": {"name": "医药医疗", "level1": "医药医疗", "setcode": "1"},
    "881087": {"name": "食品饮料", "level1": "食品饮料", "setcode": "1"},
    "881061": {"name": "汽车", "level1": "汽车", "setcode": "1"},
    "881134": {"name": "国防军工", "level1": "国防军工", "setcode": "1"},
    "881056": {"name": "传媒", "level1": "传媒", "setcode": "1"},
    "881200": {"name": "软件服务", "level1": "计算机", "setcode": "1"},
    "881343": {"name": "消费电子", "level1": "电子", "setcode": "1"},
    "881168": {"name": "建筑", "level1": "建筑", "setcode": "1"},
    "881059": {"name": "商贸", "level1": "商贸", "setcode": "1"},
    "881069": {"name": "交通运输", "level1": "交通运输", "setcode": "1"},
    "881148": {"name": "家电", "level1": "家电", "setcode": "1"},
    "881044": {"name": "环保", "level1": "环保", "setcode": "1"},
    "881161": {"name": "纺织服饰", "level1": "纺织服饰", "setcode": "1"},
    "881138": {"name": "农林牧渔", "level1": "农林牧渔", "setcode": "1"},
    "881062": {"name": "房地产", "level1": "房地产", "setcode": "1"},
    "881133": {"name": "社会服务", "level1": "社会服务", "setcode": "1"},
    "881113": {"name": "轻工制造", "level1": "轻工制造", "setcode": "1"},
    "881146": {"name": "非银金融", "level1": "非银金融", "setcode": "1"},
}

BENCHMARK = {"000001": {"name": "上证指数", "setcode": "1"}}


def print_instructions():
    """Print instructions for the user to fetch data via WorkBuddy."""
    print("""
============================================================
CANSLIM Dashboard - Data Refresh Instructions
============================================================

This script cannot directly call TDX MCP tools.
Please use WorkBuddy to fetch data by running the following
DeferExecuteTool calls for each sector:

For BENCHMARK:
  mcp__tdx-connector__tdx_kline code="000001" setcode="1" period="4" wantNum="60"

For SECTORS (repeat for each):
  mcp__tdx-connector__tdx_kline code="881319" setcode="1" period="4" wantNum="60"

Then save the resulting JSON ListItem data to:
  data/kline_000001.json
  data/kline_881319.json
  ... etc.

After all data files are saved, run:
  python compute.py

============================================================

Currently tracked sectors:
""")
    for code, info in sorted(SECTORS.items()):
        print(f"  {code} - {info['name']} ({info['level1']})")

    print(f"\nTotal: {len(SECTORS)} sectors + 1 benchmark")
    print(f"Data directory: {DATA_DIR}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print_instructions()
