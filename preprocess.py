"""
因子预处理模块 — 轻量版（适配寻龙诀看板）
==========================================
三步流水线：
1. Winsorize — 极端值截尾
2. Cross-sectional Z-Score — 横截面标准化
3. preprocess_factor_panel — 统一入口

输入: 股票因子列表 [{code, rps, alpha, eps_yoy, roe, ...}, ...]
输出: 含 zscore 后缀的因子面板 + 预处理报告

参考: xuyafei/quant_strategy 第96天文章
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def winsorize_series(values: np.ndarray, lower_pct: float = 1.0,
                     upper_pct: float = 99.0) -> np.ndarray:
    """
    对一组因子值做百分位截尾。
    
    参数:
        values: 原始因子值数组（含 NaN）
        lower_pct: 截尾下界百分位（默认 1%）
        upper_pct: 截尾上界百分位（默认 99%）
    
    返回:
        截尾后的数组，NaN 保持不变
    """
    valid = values[~np.isnan(values)]
    if len(valid) < 5:
        return values.copy()
    
    if np.unique(valid).size <= 2:
        return values.copy()
    
    lo = np.percentile(valid, lower_pct)
    hi = np.percentile(valid, upper_pct)
    
    if lo >= hi:
        return values.copy()
    
    result = values.copy()
    mask = ~np.isnan(result)
    result[mask] = np.clip(result[mask], lo, hi)
    return result


def cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """
    横截面 Z-Score: (x - mean) / std
    
    参数:
        values: 因子值数组（含 NaN，已 winsorize 过）
    
    返回:
        zscore 标准化后的数组，NaN 不变
    """
    valid_mask = ~np.isnan(values)
    valid = values[valid_mask]
    
    if len(valid) < 3:
        return np.full_like(values, np.nan)
    
    mean_val = np.mean(valid)
    std_val = np.std(valid)
    
    if std_val < 1e-10:
        return np.full_like(values, np.nan)
    
    result = np.full_like(values, np.nan)
    result[valid_mask] = (valid - mean_val) / std_val
    return result


def zscore_to_rank_score(zscore: np.ndarray) -> np.ndarray:
    """
    将 zscore 映射到 1-100 的排序评分。
    使用百分位排名，而非线性映射。
    
    参数:
        zscore: zscore 标准化值
    
    返回:
        1-100 评分，NaN 映射为 50
    """
    result = np.full_like(zscore, 50.0, dtype=float)
    valid_mask = ~np.isnan(zscore)
    valid = zscore[valid_mask]
    
    if len(valid) < 3:
        return result
    
    # 百分位排名 → 0-1 → 1-100
    ranks = np.argsort(np.argsort(valid))  # 0-based rank
    pct = (ranks + 1) / (len(valid) + 1)  # 避免 0 和 1
    result[valid_mask] = pct * 100
    return result


def build_factor_dict(stocks: List[Dict], factor_specs: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    从股票列表构建因子数组字典。
    
    参数:
        stocks: [{code, rps, alpha, eps_yoy, roe, ...}, ...]
        factor_specs: {因子名: 股票字典中的键}
    
    返回:
        {因子名: np.array}
    """
    n = len(stocks)
    factors = {}
    
    for factor_name, key in factor_specs.items():
        values = np.full(n, np.nan)
        for i, s in enumerate(stocks):
            v = s.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                try:
                    values[i] = float(v)
                except (ValueError, TypeError):
                    pass
        factors[factor_name] = values
    
    return factors


def preprocess_factors(stocks: List[Dict],
                       factor_specs: Dict[str, str],
                       winsor_lower: float = 1.0,
                       winsor_upper: float = 99.0,
                       min_valid: int = 5) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict]:
    """
    因子预处理统一入口 — 对股票因子列表做 winsorize → zscore → rank。
    
    参数:
        stocks: 股票字典列表
        factor_specs: {因子名: 键} 映射
        winsor_lower: 截尾下界百分位
        winsor_upper: 截尾上界百分位
        min_valid: 有效值最少数量，不足则跳过该因子
    
    返回:
        (zscore_dict, rank_dict, report)
        - zscore_dict: {因子名_zscore: np.array} 标准化值
        - rank_dict: {因子名_rank: np.array} 0-100 排名评分
        - report: 处理报告
    """
    # Step 0: 提取因子
    raw = build_factor_dict(stocks, factor_specs)
    report = {
        'n_stocks': len(stocks),
        'factors_processed': [],
        'factors_skipped': [],
        'details': {},
    }
    
    zscores = {}
    ranks = {}
    
    for name, values in raw.items():
        n_valid = np.sum(~np.isnan(values))
        
        if n_valid < min_valid:
            report['factors_skipped'].append(f'{name} (有效值{n_valid}<{min_valid})')
            continue
        
        # 检查是否单一值
        if np.unique(values[~np.isnan(values)]).size <= 2:
            report['factors_skipped'].append(f'{name} (值域过窄)')
            continue
        
        # Step 1: Winsorize
        winsored = winsorize_series(values, winsor_lower, winsor_upper)
        
        # Step 2: Z-Score
        z = cross_sectional_zscore(winsored)
        
        if np.all(np.isnan(z)):
            report['factors_skipped'].append(f'{name} (zscore 全部 NaN)')
            continue
        
        # Step 3: Rank to 0-100
        r = zscore_to_rank_score(z)
        
        zscores[f'{name}_zscore'] = z
        ranks[f'{name}_rank'] = r
        report['factors_processed'].append(name)
        
        # 统计
        valid_z = z[~np.isnan(z)]
        report['details'][name] = {
            'n_valid': int(n_valid),
            'mean_raw': round(float(np.nanmean(values)), 4),
            'std_raw': round(float(np.nanstd(values)), 4),
            'mean_z': round(float(np.mean(valid_z)), 4),
            'std_z': round(float(np.std(valid_z)), 4),
            'n_clamped': int(np.sum(~np.isnan(winsored) & (values != winsored))),
        }
    
    print(f"\n[Preprocess] {len(report['factors_processed'])} 个因子完成标准化")
    for name in report['factors_processed']:
        d = report['details'][name]
        print(f"  {name:>15s}: raw μ={d['mean_raw']:.1f} σ={d['std_raw']:.1f} → "
              f"z μ={d['mean_z']:.3f} σ={d['std_z']:.3f} | clamped={d['n_clamped']}")
    
    if report['factors_skipped']:
        print(f"[Preprocess] 跳过 {len(report['factors_skipped'])} 个因子: {', '.join(report['factors_skipped'])}")
    
    return zscores, ranks, report


def annotate_stocks(stocks: List[Dict],
                    zscores: Dict[str, np.ndarray],
                    ranks: Dict[str, np.ndarray]) -> List[Dict]:
    """
    将预处理结果注回股票字典。
    每个股票新增 {factor}_zscore 和 {factor}_rank 字段。
    """
    for i, s in enumerate(stocks):
        for key, arr in {**zscores, **ranks}.items():
            if i < len(arr) and not np.isnan(arr[i]):
                s[key] = round(float(arr[i]), 4)
    return stocks


# ============================================================
#  CANSLIM 看板专用因子规格
# ============================================================

CANSLIM_FACTOR_SPECS = {
    # RPS & Alpha
    'rps':       'rps',
    'alpha':     'alpha',
    'smart':     'smart',
    # 基本面（原始值）
    'eps_yoy':   'eps_yoy',
    'roe':       'roe',
    'profit_yoy':'profit_yoy',
    'op_yoy':    'op_yoy',
    'margin':    'margin',
    'debt':      'debt',
    # 之前已映射的 0-100 分（保留用于对比）
    'eps_score': '_eps_score',
    'roe_score': '_roe_score',
    'profit_score': '_profit_score',
}


if __name__ == '__main__':
    # 自测
    np.random.seed(42)
    test_stocks = []
    for i in range(100):
        test_stocks.append({
            'code': f'{i:06d}',
            'name': f'Stock{i}',
            'rps': np.random.randint(30, 99),
            'alpha': np.random.normal(0, 15),
            'eps_yoy': np.random.normal(20, 30),
            'roe': np.random.normal(12, 8),
            'profit_yoy': np.random.normal(15, 40),
        })
    
    # 注入极端值
    test_stocks[0]['eps_yoy'] = 500
    test_stocks[1]['roe'] = -200
    test_stocks[2]['rps'] = 2
    test_stocks[50]['eps_yoy'] = None
    
    print("=" * 60)
    print("  因子预处理模块 — 自测")
    print("=" * 60)
    
    specs = {
        'rps': 'rps',
        'alpha': 'alpha',
        'eps_yoy': 'eps_yoy',
        'roe': 'roe',
    }
    
    zscores, ranks, report = preprocess_factors(test_stocks, specs)
    
    print(f"\n处理报告: {report['n_stocks']} 股票, "
          f"{len(report['factors_processed'])} 因子标准化, "
          f"{len(report['factors_skipped'])} 跳过")
    
    # 验证
    for name, arr in zscores.items():
        valid = arr[~np.isnan(arr)]
        assert abs(np.mean(valid)) < 0.01, f"{name}: mean={np.mean(valid):.4f} ≠ 0"
        assert abs(np.std(valid) - 1.0) < 0.05, f"{name}: std={np.std(valid):.4f} ≠ 1"
        print(f"  ✅ {name}: μ={np.mean(valid):.4f}, σ={np.std(valid):.4f}")
    
    # 验证排名
    for name, arr in ranks.items():
        valid = arr[~np.isnan(arr)]
        print(f"  📊 {name}: min={valid.min():.1f}, max={valid.max():.1f}, median={np.median(valid):.1f}")
    
    print("\n✅ 所有测试通过")
