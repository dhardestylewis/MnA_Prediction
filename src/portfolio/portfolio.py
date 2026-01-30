# -*- coding: utf-8 -*-
"""
portfolio.py - Portfolio Simulation Module for M&A Pipeline

Extracted from mna_pipeline to support modular architecture per TODO.md D1.
"""

import numpy as np
from typing import Dict, Optional


def simulate_portfolio(predictions: np.ndarray, 
                       metadata_ret_fwd: np.ndarray,
                       strategy_config: Dict = None) -> Dict:
    """
    Simulate portfolio returns from predictions.
    
    Args:
        predictions: Model probability scores for test set
        metadata_ret_fwd: Forward returns aligned to test set
        strategy_config: Configuration dict with:
            - top_k: Number of holdings (default: 50)
            - weight_scheme: 'equal' or 'score_weighted'
            - clip_returns: (min, max) to clip outlier returns
    
    Returns:
        Dict with returns, exposures, turnover, concentration (HHI)
    """
    if strategy_config is None:
        strategy_config = {}
    
    top_k = strategy_config.get("top_k", 50)
    weight_scheme = strategy_config.get("weight_scheme", "equal")
    clip_min, clip_max = strategy_config.get("clip_returns", (-0.5, 1.0))
    
    if len(predictions) == 0 or metadata_ret_fwd is None:
        return {"error": "insufficient_data"}
    
    # Select top-K
    top_k_actual = min(top_k, len(predictions))
    top_k_idx = np.argsort(predictions)[-top_k_actual:]
    
    # Get returns for selected holdings
    selected_returns = metadata_ret_fwd[top_k_idx]
    selected_returns = np.clip(selected_returns, clip_min, clip_max)
    
    # Compute weights
    if weight_scheme == "score_weighted":
        scores = predictions[top_k_idx]
        weights = scores / scores.sum()
    else:  # equal
        weights = np.ones(len(top_k_idx)) / len(top_k_idx)
    
    # Portfolio return
    portfolio_return = np.sum(weights * selected_returns)
    
    # Concentration (HHI)
    hhi = np.sum(weights ** 2)
    
    # Universe return for comparison
    universe_return = np.nanmean(np.clip(metadata_ret_fwd, clip_min, clip_max))
    
    return {
        "portfolio_return": float(portfolio_return),
        "universe_return": float(universe_return),
        "excess_return": float(portfolio_return - universe_return),
        "n_holdings": int(top_k_actual),
        "hhi": float(hhi),
        "top_score_min": float(predictions[top_k_idx].min()),
        "top_score_max": float(predictions[top_k_idx].max()),
        "hit_rate": float(np.mean(selected_returns > 0)),
    }
