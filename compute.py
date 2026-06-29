#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANSLIM Dashboard - Data Computation Engine (minitick-tushare)
================================================================
Fetches sector/index data via minitick.top REST API and computes:
  RPS, RRG coordinates, market status, multi-period returns

Usage: python compute.py

Data sources: minitick.top (Tushare proxy)
Output: data/dashboard.json
"""

import json, os, sys, time, io, math
import numpy as np
import requests
from datetime import datetime, timedelta

# Ensure script directory is in path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_factors, annotate_stocks, CANSLIM_FACTOR_SPECS

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

# ─── Configuration ───
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TOKEN = os.environ.get('MINITICK_TOKEN', '293924f5f020255acfca1838d81bb45076de58ef5e2d96b079474a76')
BASE_URL = os.environ.get('MINITICK_BASE_URL', 'https://minitick.top/')
BENCHMARK = '000001.SH'  # 上证指数
BENCHMARK_NAME = '上证指数'
RS_WINDOW = 10
MA5_WEIGHT = 0.6
LOOKBACK_DAYS = 60

# ─── THS Sector Codes (同花顺行业板块 86xxxx.TI) ───
SECTORS = [
    ("861100.TI", "半导体", "电子"),
    ("861101.TI", "元器件", "电子"),
    ("861054.TI", "通信设备", "通信"),
    ("861112.TI", "化工", "化工"),
    ("861067.TI", "电力", "电力设备"),
    ("861137.TI", "机械设备", "机械设备"),
    ("861098.TI", "软件服务", "计算机"),
    ("861116.TI", "医药医疗", "医药医疗"),
    ("861122.TI", "食品饮料", "食品饮料"),
    ("861113.TI", "汽车", "汽车"),
    ("861108.TI", "国防军工", "国防军工"),
    ("861102.TI", "传媒", "传媒"),
    ("861158.TI", "消费电子", "电子"),
    ("861095.TI", "银行", "金融"),
    ("861097.TI", "保险", "金融"),
    ("861126.TI", "钢铁", "钢铁"),
    ("861160.TI", "建筑", "建筑"),
    ("861166.TI", "纺织服饰", "纺织服饰"),
    ("861136.TI", "农林牧渔", "农林牧渔"),
    ("861104.TI", "房地产", "房地产"),
]


def call_tushare(api_name, params=None, fields=None, retries=2):
    """Generic Tushare API call with retry"""
    payload = {'api_name': api_name, 'token': TOKEN, 'params': params or {}}
    if fields: payload['fields'] = fields
    for i in range(retries):
        try:
            resp = requests.post(BASE_URL, json=payload, timeout=30)
            data = resp.json()
            if data.get('code') == 0 and data.get('data'):
                return data['data']
            if 'ip' in str(data.get('msg', '')):
                time.sleep(1)
                continue
        except Exception as e:
            if i < retries - 1: time.sleep(1)
            else: raise e
    return None


def fetch_benchmark():
    """Fetch benchmark index daily data"""
    print(f"  Fetching benchmark: {BENCHMARK} ({BENCHMARK_NAME})...")
    data = call_tushare('index_daily', {'ts_code': BENCHMARK,
        'start_date': '20260301', 'end_date': '20260630'})
    if not data: return []
    items = data.get('items', [])
    # Items come in reverse chronological (newest first), reverse to chronological
    items.reverse()
    bars = []
    for item in items:
        bars.append({
            'date': item[1], 'open': float(item[3]), 'high': float(item[4]),
            'low': float(item[5]), 'close': float(item[2]),
            'amount': float(item[10]) * 1e4, 'volume': float(item[9]),
            'pct_chg': float(item[8]) if len(item) > 8 else 0,
        })
    return bars


def fetch_sector_klines(code, name):
    """Fetch sector daily K-line data"""
    data = call_tushare('ths_daily', {'ts_code': code,
        'start_date': '20260301', 'end_date': '20260630'})
    if not data: return []
    items = data.get('items', [])
    items.reverse()
    bars = []
    for item in items:
        days = item[0]  # trade_date is first field
        bars.append({
            'date': item[1] if len(item) > 1 else days,
            'open': float(item[2]) if len(item) > 2 else 0,
            'high': float(item[3]) if len(item) > 3 else 0,
            'low': float(item[4]) if len(item) > 4 else 0,
            'close': float(item[5]) if len(item) > 5 else 0,
            'amount': float(item[10]) if len(item) > 10 else 0,
            'volume': float(item[9]) if len(item) > 9 else 0,
        })
    return bars


# ─── Core Calculations (unchanged from original) ───
def compute_rs_ratio(sector_bars, bench_bars, window=RS_WINDOW):
    if len(sector_bars) < window or len(bench_bars) < window:
        return 100.0
    dates = [b["date"] for b in bench_bars]
    s_by_date = {b["date"]: b["close"] for b in sector_bars}
    b_by_date = {b["date"]: b["close"] for b in bench_bars}
    ratios = []
    for d in dates:
        if d in s_by_date and d in b_by_date and b_by_date[d] > 0 and s_by_date[d] > 0:
            ratios.append(s_by_date[d] / b_by_date[d])
    if len(ratios) < window + 10: return 100.0
    current = ratios[-1]
    recent = ratios[-window:]
    min_r, max_r = min(recent), max(recent)
    if max_r == min_r: return 100.0
    short_term = sum(ratios[-5:]) / 5
    long_term = sum(ratios[-window:]) / window
    short_norm = (short_term - min_r) / (max_r - min_r) * 200 - 100
    long_norm = (long_term - min_r) / (max_r - min_r) * 200 - 100
    return short_norm * MA5_WEIGHT + long_norm * (1 - MA5_WEIGHT)


def compute_rs_momentum(sector_bars, bench_bars, window=RS_WINDOW):
    if len(sector_bars) < window + 5 or len(bench_bars) < window: return 0.0
    rs_ratios = []
    for i in range(len(sector_bars) - 5, len(sector_bars)):
        ratio = compute_rs_ratio(sector_bars[:i+1], bench_bars[:min(i+1, len(bench_bars))], window)
        rs_ratios.append(ratio)
    if len(rs_ratios) < 2: return 0.0
    n = len(rs_ratios)
    x_mean = (n - 1) / 2
    y_mean = sum(rs_ratios) / n
    num = sum((i - x_mean) * (rs_ratios[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return (num / den * 10) if den != 0 else 0


def compute_distribution_days(bench_bars):
    if len(bench_bars) < 25: return {"distribution": 0, "follow_through": 0, "ma50": "unknown"}
    closes = [b["close"] for b in bench_bars[-50:]]
    ma50 = sum(closes) / len(closes)
    current = bench_bars[-1]["close"]
    ma50_position = "above" if current > ma50 else "below"
    recent = bench_bars[-25:]
    distribution, follow_through = 0, 0
    for i in range(1, len(recent)):
        pc = (recent[i]["close"] - recent[i-1]["close"]) / recent[i-1]["close"] if recent[i-1]["close"] > 0 else 0
        vr = recent[i]["volume"] / recent[i-1]["volume"] if recent[i-1]["volume"] > 0 else 1
        if pc < -0.002 and vr > 1.05: distribution += 1
        elif pc > 0.017 and vr > 1.05: follow_through += 1
    return {"distribution": distribution, "follow_through": follow_through, "ma50": ma50_position, "ma50_value": round(ma50, 2), "close": round(current, 2)}


def compute_market_status(dist_data):
    d, ft, ma = dist_data["distribution"], dist_data["follow_through"], dist_data["ma50"]
    if ma == "below" or d >= 7: return "red"
    if d >= 4: return "yellow_defense"
    return "green_attack"


def compute_period_returns(bars, days):
    if len(bars) < days + 1: return None
    start, end = bars[-(days+1)]["close"], bars[-1]["close"]
    return round((end - start) / start * 100, 2) if start > 0 else None


def compute_fund_flow(bars):
    """Compute 5-day fund flow proxy: daily volume vs its 20-day MA ratio trend"""
    if len(bars) < 25: return {"daily": [], "trend": 0, "net_label": "N/A"}
    vols = [b["volume"] for b in bars]
    vol_ma20 = sum(vols[-25:-5]) / 20 if len(vols) >= 25 else sum(vols[-20:]) / len(vols[-20:])
    daily = []
    for i in range(-5, 0):
        ratio = vols[i] / vol_ma20 if vol_ma20 > 0 else 1
        chg = (bars[i]["close"] - bars[i-1]["close"]) / bars[i-1]["close"] * 100 if i > -len(bars) and bars[i-1]["close"] > 0 else 0
        daily.append(round(ratio, 2))
    trend = sum(daily) / len(daily) if daily else 1
    # Net label: >1.3=heavy inflow, 1.1-1.3=inflow, 0.9-1.1=neutral, <0.9=outflow
    if trend > 1.3: net_label = "大幅流入"
    elif trend > 1.1: net_label = "流入"
    elif trend >= 0.9: net_label = "中性"
    elif trend >= 0.7: net_label = "流出"
    else: net_label = "大幅流出"
    return {"daily": daily, "trend": round(trend, 2), "net_label": net_label}


def compute_rs_momentum_safe(sector_bars, bench_bars, window=RS_WINDOW):
    """Compute RS-Momentum with safety clipping"""
    raw = compute_rs_momentum(sector_bars, bench_bars, window)
    return max(-80, min(80, raw))  # Clip to [-80, 80]


def compute_trajectory_angle(rs_ratio, rs_momentum):
    """Compute trajectory angle in degrees from positive RS-Ratio axis.
    0 = pure strength, 90 = pure momentum, 180/270 = weakness.
    Label: <15=强度驱动, 15-75=混合, >75=动量驱动"""
    if abs(rs_ratio) < 0.01 and abs(rs_momentum) < 0.01:
        return {"degrees": 0.0, "label": "N/A"}
    deg = math.degrees(math.atan2(rs_momentum, rs_ratio))
    if deg < 0: deg += 360
    if deg <= 15 or deg >= 345: label = "强度驱动"
    elif 15 < deg <= 75: label = "混合偏强"
    elif 75 < deg <= 105: label = "动量驱动"
    elif 105 < deg <= 165: label = "混合偏弱"
    elif 165 < deg <= 195: label = "强度衰减"
    elif 195 < deg <= 255: label = "弱势加速"
    elif 255 < deg <= 285: label = "动量衰减"
    else: label = "弱势混合"
    return {"degrees": round(deg, 1), "label": label}


def compute_momentum_acceleration(sector_bars, bench_bars, window=RS_WINDOW):
    """Compute RS-Momentum acceleration (2nd derivative).
    Positive = momentum speeding up, Negative = slowing down."""
    if len(sector_bars) < window + 10 or len(bench_bars) < window + 10:
        return {"value": 0.0, "signal": "无数据"}
    mtm_vals = []
    for i in range(len(sector_bars) - 7, len(sector_bars)):
        mtm = compute_rs_momentum_safe(sector_bars[:i+1], bench_bars[:min(i+1, len(bench_bars))], window)
        mtm_vals.append(mtm)
    if len(mtm_vals) < 3: return {"value": 0.0, "signal": "无数据"}
    n = len(mtm_vals)
    x_mean = (n - 1) / 2.0
    y_mean = sum(mtm_vals) / n
    num = sum((i - x_mean) * (mtm_vals[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = (num / den) if den != 0 else 0
    accel = round(slope, 2)
    if accel > 1.5: signal = "加速"
    elif accel < -1.5: signal = "减速"
    else: signal = "平稳"
    return {"value": accel, "signal": signal}


def compute_bubble_radius(volume, amount, all_volumes, all_amounts, min_r=6, max_r=32):
    max_v = max(all_volumes) if all_volumes else 1
    max_a = max(all_amounts) if all_amounts else 1
    ratio_v = volume / max_v if max_v > 0 else 0
    ratio_a = amount / max_a if max_a > 0 else 0
    return round(min_r + (ratio_v * 0.4 + ratio_a * 0.6) * (max_r - min_r), 1)


# ─── Main ───
def compute_all():
    print("=" * 60)
    print("CANSLIM Dashboard - Data Engine (minitick-tushare)")
    print("=" * 60)

    # 1. Benchmark
    print("\n[1/4] Benchmark data...")
    bench_bars = fetch_benchmark()
    if not bench_bars:
        print("  [!] Failed! Using embedded fallback.")
        return generate_fallback()
    print(f"  [OK] {len(bench_bars)} bars, latest close: {bench_bars[-1]['close']:.2f}")

    # 2. Market status
    print("\n[2/4] Market status...")
    dist_data = compute_distribution_days(bench_bars)
    market_status = compute_market_status(dist_data)
    print(f"  分布日:{dist_data['distribution']} 跟进日:{dist_data['follow_through']} MA50:{dist_data['ma50']} 状态:{market_status}")

    # 3. Sectors
    print(f"\n[3/4] Sector RRG ({len(SECTORS)} sectors)...")
    all_vols, all_amts, rrg_data = [], [], []

    for code, name, level1 in SECTORS:
        time.sleep(0.35)  # Rate limiting
        bars = fetch_sector_klines(code, name)
        if len(bars) < 20:
            print(f"  [!] {name} ({code}): {len(bars)} bars, using fallback")
            rrg_data.append({
                "code": code, "name": name, "level1": level1,
                "x": 0.0, "y": 0.0, "r": 12,
                "angle": 0.0, "angle_label": "N/A",
                "accel": 0.0, "accel_signal": "无数据",
                "close": bars[-1]["close"] if bars else 0,
                "r5": compute_period_returns(bars, 5) if bars else None,
                "r10": compute_period_returns(bars, 10) if bars else None,
                "r20": compute_period_returns(bars, 20) if bars else None,
                "_placeholder": True,
            })
            continue

        rs_ratio = compute_rs_ratio(bars, bench_bars)
        rs_momentum = compute_rs_momentum_safe(bars, bench_bars)
        angle = compute_trajectory_angle(rs_ratio, rs_momentum)
        accel = compute_momentum_acceleration(bars, bench_bars)
        avg_vol = sum(b["volume"] for b in bars[-5:]) / 5
        avg_amt = sum(b["amount"] for b in bars[-5:]) / 5
        fund_flow = compute_fund_flow(bars)
        all_vols.append(avg_vol)
        all_amts.append(avg_amt)

        rrg_data.append({
            "code": code, "name": name, "level1": level1,
            "x": round(rs_ratio, 2),
            "y": round(rs_momentum, 2),
            "angle": angle["degrees"], "angle_label": angle["label"],
            "accel": accel["value"], "accel_signal": accel["signal"],
            "avg_vol": avg_vol, "avg_amount": avg_amt,
            "close": round(bars[-1]["close"], 2),
            "r5": compute_period_returns(bars, 5),
            "r10": compute_period_returns(bars, 10),
            "r20": compute_period_returns(bars, 20),
            "fund_flow": fund_flow["daily"],
            "fund_trend": fund_flow["net_label"],
        })
        print(f"  [OK] {name:8s} RS:{rs_ratio:6.1f} Mtm:{rs_momentum:6.1f} Ang:{angle['degrees']}° {accel['signal']} 5d:{rrg_data[-1]['r5'] or 'N/A'}% 资金:{fund_flow['net_label']}")

    # Bubble radii
    for item in rrg_data:
        if not item.get("_placeholder"):
            item["r"] = compute_bubble_radius(item.get("avg_vol", 0), item.get("avg_amount", 0), all_vols, all_amts)

    # 4. Output
    print("\n[4/4] Generating output...")

    # Fetch individual stock RPS from CANSLIM Pro free API for advanced features
    stock_data = {"leading_sectors": [], "golden_radar": [], "strongest": {}}
    try:
        print("  Fetching stock RPS from CANSLIM Pro...")
        stocks_resp = requests.get('http://118.25.186.36/api/all_stocks', timeout=30)
        all_stocks = stocks_resp.json().get('stocks', [])

        # Leading sectors: group by level1, rank by avg RPS
        l1_groups = {}
        for s in all_stocks:
            l1 = s['level1']
            if l1 not in l1_groups: l1_groups[l1] = {'count':0, 'sum':0, 'high':0, 'l3_groups':{}}
            l1_groups[l1]['count'] += 1
            l1_groups[l1]['sum'] += s['rps']
            if s['rps'] >= 85: l1_groups[l1]['high'] += 1
            l3 = s.get('level3','未知')
            if l3 not in l1_groups[l1]['l3_groups']: l1_groups[l1]['l3_groups'][l3] = 0
            l1_groups[l1]['l3_groups'][l3] += 1
        for name, g in l1_groups.items():
            g['avg_rps'] = round(g['sum']/g['count'], 1)
            g['strength'] = round(g['high']/g['count']*100, 1)

        stock_data['leading_sectors'] = [
            {"name": name, "avg_rps": g['avg_rps'], "strength": g['strength'],
             "count": g['count'], "high_count": g['high'],
             "hot_l3": sorted(g['l3_groups'].items(), key=lambda x: x[1], reverse=True)[:3]}
            for name, g in sorted(l1_groups.items(), key=lambda x: x[1]['avg_rps'], reverse=True)[:12]
        ]

        # Sector concentration check
        rps99 = [s for s in all_stocks if s['rps'] >= 99]
        conc = {}
        for s in rps99: conc[s['level1']] = conc.get(s['level1'], 0) + 1
        top_conc = sorted(conc.items(), key=lambda x: -x[1])[0] if conc else ('',0)
        stock_data['concentration'] = {
            'total_rps99': len(rps99),
            'top_sector': top_conc[0],
            'top_pct': round(top_conc[1]/max(len(rps99),1)*100, 1),
            'warning': top_conc[1]/max(len(rps99),1) > 0.3
        }

        # Alpha computation: sector avg RPS for each industry
        sector_avg = {name: g['avg_rps'] for name, g in l1_groups.items()}

        # ─── RPS × RRG Cross-Validation ───
        print("  Computing RPS-RRG consensus signals...")
        rps_sector_map = {name: g['avg_rps'] for name, g in l1_groups.items()}
        for item in rrg_data:
            l1 = item['level1']
            rps_val = rps_sector_map.get(l1, 50)
            item['rps_val'] = round(rps_val, 1)
            # Consensus signal
            rrg_bull = item['x'] >= 0 and item['y'] >= 0
            rps_bull = rps_val > 50
            if rrg_bull and rps_bull:
                item['consensus'] = '强烈看多'
            elif rrg_bull and not rps_bull:
                item['consensus'] = 'RRG偏多'
            elif not rrg_bull and rps_bull:
                item['consensus'] = 'RPS背离'
            elif not rrg_bull and not rps_bull:
                item['consensus'] = '双弱'
            else:
                item['consensus'] = '中性'
        consensus_count = sum(1 for i in rrg_data if i.get('consensus') == '强烈看多')
        print(f"  [OK] RPS-RRG共识: {consensus_count} strongly bullish sectors")

        # Golden Radar: RPS > 80 with Alpha calculation
        rrgs = {item['level1']: item for item in rrg_data}
        golden_items = []
        for s in all_stocks:
            if s['rps'] >= 85:
                sa = sector_avg.get(s['level1'], 50)
                alpha = round(s['rps'] - sa, 1)
                # Smart score: RPS*0.4 + Alpha*0.4 + (100-sector_avg)*0.2
                smarts = round(s['rps']*0.4 + alpha*0.4 + (100-sa)*0.2, 1)
                golden_items.append({
                    "rps": s['rps'], "code": s['code'], "name": s['name'],
                    "level1": s['level1'], "level2": s.get('level2',''), "level3": s.get('level3',''),
                    "alpha": alpha, "sector_avg": sa, "smart": smarts,
                    "sector_rps": rps_sector_map.get(s['level1'], 50),
                    "sector_consensus": rrgs[s['level1']]['consensus'] if s['level1'] in rrgs else '',
                })
        golden_items.sort(key=lambda x: (-x['rps'], -x['alpha']))
        stock_data['golden_radar'] = golden_items[:50]

        # ─── RPS × RRG Consensus Picks ───
        cpicks = []
        for s in all_stocks:
            if s['rps'] >= 85:
                sa = sector_avg.get(s['level1'], 50)
                alpha = round(s['rps'] - sa, 1)
                scons = rrgs.get(s['level1'], {}).get('consensus', '')
                boost = 1.5 if scons == '强烈看多' else 1.2 if scons == 'RRG偏多' else 1.0
                cscore = round(s['rps'] * 0.3 + alpha * 0.3 + (100 - sa) * 0.2 + (10 if scons == '强烈看多' else 5 if scons == 'RRG偏多' else 0), 1)
                cpicks.append({
                    "rps": s['rps'], "code": s['code'], "name": s['name'],
                    "level1": s['level1'], "level3": s.get('level3', ''),
                    "alpha": alpha, "sector_avg": sa,
                    "consensus": scons, "cscore": cscore,
                })
        cpicks.sort(key=lambda x: (-x['cscore'], -x['rps']))
        stock_data['consensus_picks'] = cpicks[:30]

        # Best Alpha picks (independent of sector strength)
        alpha_picks = []
        for s in all_stocks:
            if s['rps'] >= 80:
                sa = sector_avg.get(s['level1'], 50)
                alpha = round(s['rps'] - sa, 1)
                if alpha > 30:  # Only show true outliers
                    alpha_picks.append({
                        "rps": s['rps'], "code": s['code'], "name": s['name'],
                        "level1": s['level1'], "alpha": alpha, "sector_avg": sa,
                        "smart": round(s['rps']*0.4 + alpha*0.4 + (100-sa)*0.2, 1)
                    })
        alpha_picks.sort(key=lambda x: -x['alpha'])
        stock_data['alpha_picks'] = alpha_picks[:20]

        # Strongest-in-sector
        for code, name, level1 in SECTORS:
            sector_stocks = []
            for s in all_stocks:
                if s['level1'] == level1 and s['rps'] >= 85:
                    sa = sector_avg.get(level1, 50)
                    alpha = round(s['rps'] - sa, 1)
                    smart = round(s['rps']*0.4 + alpha*0.4 + (100-sa)*0.2, 1)
                    sector_stocks.append({
                        "rps": s['rps'], "code": s['code'], "name": s['name'],
                        "l3": s.get('level3',''), "alpha": alpha, "sector_avg": sa, "smart": smart
                    })
            sector_stocks.sort(key=lambda x: (-x['rps'], -x['alpha']))
            stock_data['strongest'][name] = sector_stocks[:12]

        print(f"  [OK] Stocks: {len(all_stocks)}, Leading sectors: {len(stock_data['leading_sectors'])}, Golden radar: {len(stock_data['golden_radar'])}")

        # ─── Sector Extra (PE, moneyflow, sentiment) ───
        print("  Enhancing sector data (PE, fund flow, sentiment)...")
        try:
            fetch_sector_extra(rrg_data, all_stocks)
        except Exception as e:
            print(f"  [!] Sector extra failed (non-blocking): {e}")

        # ─── Multi-Version CANSLIM Scoring ───
        print("  Computing multi-version scores...")
        versions = initialize_version_structures()
        stock_data['broker_consensus'] = fetch_broker_consensus()

        # ─── Fundamental Data (EPS/ROE) for top stocks ───
        print("  Fetching fundamentals for top 40 stocks...")
        fund_codes = list(set(s['code'] for s in stock_data['golden_radar'][:40]))
        fundamentals = fetch_fundamentals_batch(fund_codes)
        for item in stock_data['golden_radar']:
            item['fin'] = fundamentals.get(item['code'], {})
        # Re-score with fundamentals
        versions = rescore_with_fundamentals(versions, stock_data, sector_avg)
        for ver in versions.values():
            for item in ver.get('picks', []):
                item['fin'] = fundamentals.get(item['code'], {})
        stock_data['versions'] = versions
        print(f"  [OK] Fundamentals fetched for {len(fundamentals)} stocks, 4 versions scored")

        # ─── Snapshot Comparison ───
        print("  Computing day-over-day changes...")
        changes = compute_day_changes(stock_data['leading_sectors'], stock_data, sector_avg)
        stock_data['changes'] = changes
        stock_data['concentration_warning'] = stock_data.get('concentration', {}).get('warning', False)

        # Save snapshot for next day comparison
        save_snapshot({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'leading': [(s['name'], s['avg_rps'], s['strength']) for s in stock_data['leading_sectors']],
            'rps99_total': stock_data.get('concentration', {}).get('total_rps99', 0),
        })

    except Exception as e:
        print(f"  [!] Stock RPS fetch failed: {e}")

    output = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "minitick-tushare + CANSLIM Pro RPS",
        "stock_data": stock_data,
        "benchmark": {"code": BENCHMARK, "name": BENCHMARK_NAME,
            "close": dist_data["close"], "ma50": dist_data["ma50_value"], "ma50_position": dist_data["ma50"]},
        "market_status": {
            "distribution_days": dist_data["distribution"],
            "follow_through_days": dist_data["follow_through"],
            "ma50_position": dist_data["ma50"],
            "light": market_status,
            "label": {"green_attack": "绿 · 进攻", "yellow_defense": "黄 · 防御", "red": "红 · 避险"}.get(market_status, market_status),
        },
        "rrg_data": rrg_data,
    }

    # Merge with fallback for missing sectors
    fallback = generate_fallback()
    fb_map = {s["code"]: s for s in fallback["rrg_data"]}
    for item in rrg_data:
        if item.get("_placeholder"):
            fb = fb_map.get(item["code"])
            if fb:
                item["x"] = fb["x"]
                item["y"] = fb["y"]
                item["r"] = fb["r"]
                item.pop("_placeholder", None)

    output["rrg_data"] = rrg_data

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "dashboard.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Written: {output_path}")
    print(f"   Live sectors: {sum(1 for s in rrg_data if not s.get('_placeholder'))}/{len(rrg_data)}")

    # Generate self-contained HTML (no fetch required for file://)
    generate_self_contained(output)
    return output


def generate_self_contained(output_data):
    """Inject dashboard JSON into the HTML so it works with file:// protocol.
    Replaces the INLINE_DATA placeholder with actual JSON data."""
    template_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(template_path):
        print("  [!] Template not found, skipping self-contained HTML")
        return

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    data_json = json.dumps(output_data, ensure_ascii=False)
    injection = f"<script>var INLINE_DATA={data_json};</script>"

    # Insert right before </body>
    if "<script>var INLINE_DATA=" in html:
        # Update existing injection
        html = __import__('re').sub(
            r"<script>var INLINE_DATA=.*?</script>",
            injection,
            html
        )
    else:
        # First-time injection
        html = html.replace("</body>", f"{injection}\n</body>")

    output_html = os.path.join(os.path.dirname(__file__), "index.html")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [OK] Self-contained HTML updated: {output_html}")


def initialize_version_structures():
    """Initialize empty version structures before scoring."""
    return {
        "classic": {"name": "CANSLIM Classic", "desc": "O'Neil L+M+Alpha+EPS", "picks": []},
        "car_ln": {"name": "CAR_LN 成长版", "desc": "长城基金 C+A+R+L+N", "picks": []},
        "rotation": {"name": "CANSLIM 轮动版", "desc": "国信 S+L+I+M", "picks": []},
        "fesc": {"name": "FESC 精选版", "desc": "华创 4维+基本面", "picks": []},
    }


def fetch_fundamentals_batch(codes):
    """Fetch EPS/ROE for a list of stock codes."""
    results = {}
    for i, code in enumerate(codes):
        try:
            time.sleep(0.33)
            d = call_tushare('fina_indicator',
                params={'ts_code': code, 'start_date': '20260101', 'end_date': '20260630'},
                fields='basic_eps_yoy,roe,netprofit_yoy,op_yoy,netprofit_margin,grossprofit_margin,debt_to_assets')
            if d and d.get('items'):
                # Get latest quarter with data
                for item in d['items']:
                    row = dict(zip(d['fields'], item))
                    if row.get('basic_eps_yoy') is not None:
                        results[code] = {
                            'eps_yoy': round(row['basic_eps_yoy'], 1),
                            'roe': round(row['roe'], 1) if row['roe'] else None,
                            'profit_yoy': round(row['netprofit_yoy'], 1) if row['netprofit_yoy'] else None,
                            'op_yoy': round(row['op_yoy'], 1) if row['op_yoy'] else None,
                            'margin': round(row['netprofit_margin'], 1) if row['netprofit_margin'] else None,
                            'debt': round(row['debt_to_assets'], 1) if row['debt_to_assets'] else None,
                        }
                        break
        except Exception:
            pass
    return results


def fetch_sector_extra(rrg_data, all_stocks):
    """Fetch PE valuation, moneyflow for each sector (sample 3 stocks per sector)."""
    if not all_stocks: return
    print("  Fetching sector PE & moneyflow (sample)...")
    l1_stocks = {}
    for s in all_stocks:
        l1 = s['level1']
        if l1 not in l1_stocks: l1_stocks[l1] = []
        l1_stocks[l1].append(s['code'])
    # Try today, yesterday, day-before for date fallback
    today = datetime.now()
    dates_try = [(today - timedelta(days=d)).strftime('%Y%m%d') for d in range(4)]
    for item in rrg_data:
        l1 = item['level1']
        codes = l1_stocks.get(l1, [])[:3]
        if not codes: continue
        pe_vals, mf_total = [], 0
        for code in codes:
            for try_date in dates_try:
                try:
                    time.sleep(0.33)
                    db = call_tushare('daily_basic',
                        params={'ts_code': code, 'trade_date': try_date},
                        fields='pe_ttm')
                    if db and db.get('items'):
                        for j, f in enumerate(db['fields']):
                            if f == 'pe_ttm':
                                v = db['items'][0][j]
                                if v is not None:
                                    pe = float(v)
                                    if 0 < pe < 5000:
                                        pe_vals.append(pe)
                                        break
                        if pe_vals: break
                except: pass
            for try_date in dates_try:
                try:
                    time.sleep(0.33)
                    mf = call_tushare('moneyflow',
                        params={'ts_code': code, 'trade_date': try_date},
                        fields='net_mf_amount')
                    if mf and mf.get('items'):
                        for j, f in enumerate(mf['fields']):
                            if f == 'net_mf_amount':
                                v = mf['items'][0][j]
                                if v is not None:
                                    mf_total += float(v)
                                    break
                        break
                except: pass
        item['pe_ttm'] = round(sorted(pe_vals)[len(pe_vals)//2], 1) if pe_vals else None
        item['net_mf'] = round(mf_total / 1e8, 2)
        item['up_cnt'] = item.get('up_cnt', 0)
        item['down_cnt'] = item.get('down_cnt', 0)
        item['limit_up'] = item.get('limit_up', 0)
        item['sector_total'] = len(l1_stocks.get(l1, []))
    print(f"  [OK] PE for {sum(1 for i in rrg_data if i.get('pe_ttm'))} sectors")


def rescore_with_fundamentals(versions, stock_data, sector_avg):
    """Re-compute version scores with real fundamental data.
    
    v2: 使用 winsorize + zscore + rank 预处理因子，
    消除不同因子量纲差异对融合结果的影响。
    同时保留传统线性映射作为对比（_linear 后缀）。
    """
    items = stock_data['golden_radar']
    
    # ─── 保留传统线性映射（对比用）───
    for item in items:
        fin = item.get('fin', {})
        eps_yoy = fin.get('eps_yoy') or 0
        item['_eps_score_linear'] = round(max(0, min(100, (eps_yoy + 50) * 0.8)), 1)
        roe = fin.get('roe') or 5
        item['_roe_score_linear'] = round(max(0, min(100, roe * 5)), 1)
        profit_yoy = fin.get('profit_yoy') or 0
        item['_profit_score_linear'] = round(max(0, min(100, (profit_yoy + 30) * 0.6)), 1)
    
    # ─── 步骤 1: 提取原始因子值 ───
    raw_stocks = []
    for s in items:
        fin = s.get('fin', {})
        raw_stocks.append({
            'rps': s.get('rps', 50),
            'alpha': s.get('alpha', 0),
            'smart': s.get('smart', 50),
            'eps_yoy': fin.get('eps_yoy'),
            'roe': fin.get('roe'),
            'profit_yoy': fin.get('profit_yoy'),
            'op_yoy': fin.get('op_yoy'),
            'margin': fin.get('margin'),
            'debt': fin.get('debt'),
        })
    
    # ─── 步骤 2: Winsorize + Z-Score + Rank → 0-100 ───
    # 只对有意义的基本面因子做标准化，RPS/Alpha 已天然 0-100
    factor_specs = {
        'eps_yoy': 'eps_yoy',
        'roe': 'roe',
        'profit_yoy': 'profit_yoy',
        'op_yoy': 'op_yoy',
        'margin': 'margin',
    }
    
    zscores, ranks, pre_report = preprocess_factors(raw_stocks, factor_specs,
                                                     winsor_lower=1.0,
                                                     winsor_upper=99.0)
    
    # ─── 步骤 3: 将 rank 评分注回 stock_data ───
    for i, s in enumerate(items):
        for key, arr in ranks.items():
            if i < len(arr) and not np.isnan(arr[i]):
                s[key] = round(float(arr[i]), 1)
        for key, arr in zscores.items():
            if i < len(arr) and not np.isnan(arr[i]):
                s[key] = round(float(arr[i]), 4)
        # 向后兼容的别名
        s['_eps_score'] = s.get('eps_yoy_rank', 50)
        s['_roe_score'] = s.get('roe_rank', 50)
        s['_profit_score'] = s.get('profit_yoy_rank', 50)
        # 也保留传统线性映射
        s['_eps_score_linear'] = s.get('_eps_score_linear', 50)
        s['_roe_score_linear'] = s.get('_roe_score_linear', 50)
    
    # ─── 步骤 4: 用 rank 评分重算四个版本 ───
    # Version 1: Classic — L(momentum) + Alpha + C(eps_rank)
    classic = []
    for s in items:
        sc = s['rps']*0.45 + s.get('alpha',0)*0.25 + s.get('_eps_score',50)*0.30
        classic.append((sc, s))
    classic.sort(key=lambda x: -x[0])
    versions['classic']['picks'] = [dict(s, score=round(sc, 1)) for sc, s in classic[:30]]
    
    # Version 2: CAR_LN — C + A(annual=profit) + R(ROE) + L + N(rps/alpha)
    car_ln = []
    for s in items:
        sc = (s.get('_eps_score',50)*0.25 + 
              s.get('_roe_score',50)*0.25 + 
              s['rps']*0.20 + 
              s.get('alpha',0)*0.15 + 
              s.get('_profit_score',50)*0.15)
        car_ln.append((sc, s))
    car_ln.sort(key=lambda x: -x[0])
    versions['car_ln']['picks'] = [dict(s, score=round(sc, 1)) for sc, s in car_ln[:30]]
    
    # Version 3: Rotation — sector-level rotation
    rotation_picks = []
    for s in items:
        ls = stock_data['leading_sectors']
        sector_rank = next((i for i, x in enumerate(ls) if x['name'] == s['level1']), 10)
        rot_score = s['rps']*0.2 + s.get('alpha',0)*0.2 + max(0, 50-sector_rank*5)*0.4 + s.get('smart',0)*0.2
        rotation_picks.append((rot_score, s))
    rotation_picks.sort(key=lambda x: -x[0])
    versions['rotation']['picks'] = [dict(s, score=round(sc, 1)) for sc, s in rotation_picks[:30]]
    
    # Version 4: FESC — 4D fundamental + momentum
    fesc = []
    for s in items:
        sc = (s['rps']*0.20 + 
              s.get('_eps_score',50)*0.20 + 
              s.get('_roe_score',50)*0.20 + 
              s.get('_profit_score',50)*0.15 + 
              s.get('alpha',0)*0.10 + 
              s.get('smart',50)*0.15)
        fesc.append((sc, s))
    fesc.sort(key=lambda x: -x[0])
    versions['fesc']['picks'] = [dict(s, score=round(sc, 1)) for sc, s in fesc[:30]]
    
    # 记录预处理统计
    versions['_preprocess'] = {
        'n_factors': len(pre_report['factors_processed']),
        'factors_processed': pre_report['factors_processed'],
        'skipped': pre_report.get('factors_skipped', []),
    }
    
    return versions


def fetch_broker_consensus():
    """Fetch current month broker stock picks for consensus signal."""
    try:
        r = requests.post(BASE_URL, json={
            'api_name': 'broker_recommend',
            'params': {'month': datetime.now().strftime('%Y%m')},
            'token': TOKEN
        }, timeout=15)
        d = r.json()
        if d.get('code') == 0 and d.get('data'):
            items = d['data']['items']
            # Count picks per stock
            consensus = {}
            for item in items:
                code = item[2]
                consensus[code] = consensus.get(code, 0) + 1
            # Return top consensus stocks
            top = sorted(consensus.items(), key=lambda x: -x[1])[:30]
            return [{"code": c, "brokers": n} for c, n in top]
        return []
    except Exception:
        return []
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "embedded-fallback",
        "benchmark": {"code": "000001.SH","name":"上证指数","close":4090.48,"ma50":4060.0,"ma50_position":"above"},
        "market_status": {"distribution_days":5,"follow_through_days":0,"ma50_position":"above","light":"yellow_defense","label":"黄 · 防御"},
        "rrg_data": [
            {"code":"861100.TI","name":"半导体","level1":"电子","x":65.2,"y":12.5,"r":28.3,"close":1997,"r5":8.5,"r10":15.2,"r20":42.1},
            {"code":"861101.TI","name":"元器件","level1":"电子","x":52.1,"y":8.9,"r":24.6,"close":2850,"r5":5.8,"r10":12.4,"r20":35.6},
            {"code":"861054.TI","name":"通信设备","level1":"通信","x":48.7,"y":7.2,"r":22.1,"close":1980,"r5":4.2,"r10":10.8,"r20":28.9},
            {"code":"861112.TI","name":"化工","level1":"化工","x":15.3,"y":-2.1,"r":18.5,"close":1620,"r5":-1.2,"r10":-3.5,"r20":-8.7},
            {"code":"861067.TI","name":"电力","level1":"电力设备","x":20.5,"y":-5.8,"r":19.2,"close":2100,"r5":-2.8,"r10":-5.1,"r20":-9.0},
            {"code":"861137.TI","name":"机械设备","level1":"机械设备","x":35.2,"y":3.1,"r":20.8,"close":1850,"r5":2.1,"r10":4.5,"r20":3.9},
            {"code":"861098.TI","name":"软件服务","level1":"计算机","x":28.4,"y":-1.5,"r":16.7,"close":1750,"r5":-0.8,"r10":2.1,"r20":-1.5},
            {"code":"861116.TI","name":"医药医疗","level1":"医药医疗","x":-35.6,"y":-18.3,"r":15.3,"close":980,"r5":-6.8,"r10":-12.4,"r20":-22.7},
            {"code":"861122.TI","name":"食品饮料","level1":"食品饮料","x":-42.1,"y":-22.7,"r":14.8,"close":1250,"r5":-4.5,"r10":-9.8,"r20":-18.3},
            {"code":"861113.TI","name":"汽车","level1":"汽车","x":-10.2,"y":-8.5,"r":17.1,"close":1680,"r5":-3.2,"r10":-6.7,"r20":-12.1},
            {"code":"861108.TI","name":"国防军工","level1":"国防军工","x":-5.8,"y":1.2,"r":16.0,"close":1420,"r5":1.2,"r10":-2.3,"r20":1.8},
            {"code":"861102.TI","name":"传媒","level1":"传媒","x":-25.3,"y":-12.8,"r":13.5,"close":890,"r5":-5.1,"r10":-10.2,"r20":-15.4},
            {"code":"861158.TI","name":"消费电子","level1":"电子","x":38.9,"y":5.4,"r":21.3,"close":1920,"r5":3.8,"r10":8.2,"r20":24.5},
            {"code":"861126.TI","name":"钢铁","level1":"钢铁","x":-15.2,"y":-10.5,"r":12.3,"close":830,"r5":-3.0,"r10":-6.5,"r20":-10.2},
        ],
    }


def compute_day_changes(leading, stock_data, sector_avg):
    """Compare today's leading sectors with yesterday's snapshot."""
    snapshot_path = os.path.join(DATA_DIR, "snapshot.json")
    if not os.path.exists(snapshot_path):
        return {s['name']: {'change': 0, 'arrow': 'new'} for s in leading}

    with open(snapshot_path, 'r', encoding='utf-8') as f:
        prev = json.load(f)

    prev_map = {s[0]: {'avg': s[1], 'strength': s[2]} for s in prev.get('leading', [])}
    changes = {}
    for s in leading:
        name = s['name']
        if name in prev_map:
            diff = round(s['avg_rps'] - prev_map[name]['avg'], 1)
            arrow = 'up' if diff > 1 else 'down' if diff < -1 else 'flat'
        else:
            diff = 0
            arrow = 'new'
        changes[name] = {'change': diff, 'arrow': arrow, 'prev_avg': prev_map.get(name, {}).get('avg')}

    return changes


def save_snapshot(data):
    """Save today's data for tomorrow's comparison."""
    path = os.path.join(DATA_DIR, "snapshot.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def generate_fallback():
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "embedded-fallback",
        "benchmark": {"code": "000001.SH","name":"上证指数","close":4090.48,"ma50":4060.0,"ma50_position":"above"},
        "market_status": {"distribution_days":5,"follow_through_days":0,"ma50_position":"above","light":"yellow_defense","label":"黄 · 防御"},
        "rrg_data": [
            {"code":"861100.TI","name":"半导体","level1":"电子","x":65.2,"y":12.5,"r":28.3,"close":1997,"r5":8.5,"r10":15.2,"r20":42.1},
            {"code":"861101.TI","name":"元器件","level1":"电子","x":52.1,"y":8.9,"r":24.6,"close":2850,"r5":5.8,"r10":12.4,"r20":35.6},
            {"code":"861054.TI","name":"通信设备","level1":"通信","x":48.7,"y":7.2,"r":22.1,"close":1980,"r5":4.2,"r10":10.8,"r20":28.9},
            {"code":"861112.TI","name":"化工","level1":"化工","x":15.3,"y":-2.1,"r":18.5,"close":1620,"r5":-1.2,"r10":-3.5,"r20":-8.7},
            {"code":"861067.TI","name":"电力","level1":"电力设备","x":20.5,"y":-5.8,"r":19.2,"close":2100,"r5":-2.8,"r10":-5.1,"r20":-9.0},
            {"code":"861137.TI","name":"机械设备","level1":"机械设备","x":35.2,"y":3.1,"r":20.8,"close":1850,"r5":2.1,"r10":4.5,"r20":3.9},
            {"code":"861098.TI","name":"软件服务","level1":"计算机","x":28.4,"y":-1.5,"r":16.7,"close":1750,"r5":-0.8,"r10":2.1,"r20":-1.5},
            {"code":"861116.TI","name":"医药医疗","level1":"医药医疗","x":-35.6,"y":-18.3,"r":15.3,"close":980,"r5":-6.8,"r10":-12.4,"r20":-22.7},
            {"code":"861122.TI","name":"食品饮料","level1":"食品饮料","x":-42.1,"y":-22.7,"r":14.8,"close":1250,"r5":-4.5,"r10":-9.8,"r20":-18.3},
            {"code":"861113.TI","name":"汽车","level1":"汽车","x":-10.2,"y":-8.5,"r":17.1,"close":1680,"r5":-3.2,"r10":-6.7,"r20":-12.1},
            {"code":"861108.TI","name":"国防军工","level1":"国防军工","x":-5.8,"y":1.2,"r":16.0,"close":1420,"r5":1.2,"r10":-2.3,"r20":1.8},
            {"code":"861102.TI","name":"传媒","level1":"传媒","x":-25.3,"y":-12.8,"r":13.5,"close":890,"r5":-5.1,"r10":-10.2,"r20":-15.4},
            {"code":"861158.TI","name":"消费电子","level1":"电子","x":38.9,"y":5.4,"r":21.3,"close":1920,"r5":3.8,"r10":8.2,"r20":24.5},
            {"code":"861126.TI","name":"钢铁","level1":"钢铁","x":-15.2,"y":-10.5,"r":12.3,"close":830,"r5":-3.0,"r10":-6.5,"r20":-10.2},
        ],
    }


if __name__ == "__main__":
    compute_all()
