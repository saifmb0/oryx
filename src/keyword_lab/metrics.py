import logging
import math
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from scipy import stats


# =============================================================================
# Commercial Intent Indicators (CPC Proxy Heuristics)
# =============================================================================

# High-value commercial keywords indicating purchase/lead intent
# These typically have higher CPC in paid search
COMMERCIAL_TRIGGERS = {
    "quote", "quotes", "price", "prices", "pricing", "cost", "costs",
    "rates", "estimate", "estimates", "estimation",
    "companies", "company", "contractors", "contractor", "services",
    "hire", "hiring", "agency", "agencies", "firm", "firms",
    "consultation", "consult", "consultant", "consultants",
    "package", "packages", "plan", "plans",
}

# Transactional triggers (highest commercial value)
TRANSACTIONAL_TRIGGERS = {
    "buy", "purchase", "order", "book", "booking",
    "get quote", "request quote", "free quote",
    "near me", "in my area", "local",
    "for sale", "deals", "discount", "offer",
}

# B2B/Enterprise indicators (high-value leads)
B2B_TRIGGERS = {
    "enterprise", "business", "commercial", "corporate", "b2b",
    "wholesale", "bulk", "industrial", "professional",
    "solutions", "platform", "software", "system",
}

# UAE/Gulf-specific commercial terms
UAE_COMMERCIAL_TRIGGERS = {
    "fit out", "fitout", "turnkey", "mep", "hvac",
    "renovation", "construction", "contracting",
    "interior design", "landscaping",
    "license", "approval", "permit",
    "villa", "warehouse", "office",
}


def raw_volume_proxy(
    keyword: str, 
    freq: int, 
    is_question: bool,
    is_validated: bool = False,
) -> float:
    """
    Calculate raw volume proxy for a keyword.
    
    Args:
        keyword: The keyword string
        freq: Document frequency count
        is_question: Whether keyword is a question
        is_validated: Whether keyword was validated via autocomplete (2x boost)
        
    Returns:
        Raw volume score (not normalized)
    """
    base = 1.0 + math.log1p(max(1, freq))  # grows slowly
    if is_question:
        base *= 1.2
    # small boost for longer long-tails
    base *= (1.0 + 0.05 * max(0, len(keyword.split()) - 2))
    # 2x multiplier for autocomplete-validated keywords
    if is_validated:
        base *= 2.0
    return float(base)


def raw_difficulty_proxy(keyword: str, total_results: Optional[int] = None) -> float:
    if total_results:
        return float(math.log10(max(10, total_results)))  # ~1..10
    length_penalty = 1.0 if len(keyword.split()) <= 2 else 0.6
    head_terms = any(h in keyword for h in ["best", "top", "review", "compare", "pricing"])  # slightly harder
    term_penalty = 1.2 if head_terms else 1.0
    return float(length_penalty * term_penalty)


def business_relevance(intent: str, goals: str) -> float:
    g = goals.lower()
    # Match sales/revenue/lead (catches 'leads', 'lead generation', etc.)
    if any(k in g for k in ["sales", "revenue", "lead"]):
        if intent in ("transactional", "commercial"):
            return 1.0
    # Match traffic/brand awareness goals
    if any(k in g for k in ["traffic", "brand", "awareness", "exposure"]):
        if intent in ("informational", "navigational"):
            return 0.8
    # Default fallback
    return 0.6


def commercial_value(
    keyword: str,
    intent: str = "informational",
    geo: str = "global",
    niche: Optional[str] = None,
) -> float:
    """
    Calculate commercial value score (CPC proxy heuristic).
    
    Since we don't have paid API access to CPC data, we use trigger-word
    heuristics to estimate the commercial value of a keyword.
    
    High commercial value keywords are prioritized for lead generation
    over traffic-focused informational keywords.
    
    Args:
        keyword: The keyword to evaluate
        intent: Pre-classified intent ('transactional', 'commercial', etc.)
        geo: Geographic target (enables locale-specific triggers)
        niche: Optional niche for specialized triggers
        
    Returns:
        Commercial value score (0.0 - 1.0)
        - 0.0-0.3: Low commercial value (informational)
        - 0.3-0.6: Medium commercial value (research phase)
        - 0.6-0.8: High commercial value (comparison/evaluation)
        - 0.8-1.0: Very high commercial value (ready to convert)
    """
    kw_lower = keyword.lower()
    tokens = set(kw_lower.split())
    
    score = 0.0
    
    # Base score from intent
    intent_scores = {
        "transactional": 0.6,
        "commercial": 0.4,
        "comparative": 0.35,
        "local": 0.3,
        "complex_research": 0.2,
        "direct_answer": 0.1,
        "informational": 0.1,
        "navigational": 0.0,
    }
    score += intent_scores.get(intent, 0.1)
    
    # Transactional trigger boost (highest value)
    for trigger in TRANSACTIONAL_TRIGGERS:
        if trigger in kw_lower:
            score += 0.3
            break
    
    # Commercial trigger boost
    commercial_matches = len(tokens & COMMERCIAL_TRIGGERS)
    score += min(0.2, commercial_matches * 0.1)
    
    # B2B/Enterprise trigger boost
    b2b_matches = len(tokens & B2B_TRIGGERS)
    score += min(0.15, b2b_matches * 0.1)
    
    # UAE/Gulf-specific commercial triggers
    if geo.lower() in ("ae", "sa", "qa", "kw", "bh", "om"):
        for trigger in UAE_COMMERCIAL_TRIGGERS:
            if trigger in kw_lower:
                score += 0.15
                break
    
    # Contracting niche boost for specific service terms
    if niche and niche.lower() in ("contracting", "construction"):
        construction_terms = {"villa", "warehouse", "fit out", "renovation", "mep"}
        for term in construction_terms:
            if term in kw_lower:
                score += 0.1
                break
    
    # Cap at 1.0
    return float(min(1.0, max(0.0, score)))


def compute_metrics(
    keywords: List[str],
    clusters: Dict[str, List[str]],
    intents: Dict[str, str],
    freq: Dict[str, int],
    questions: set,
    provider: str,
    serp_total_results: Optional[Dict[str, int]] = None,
    validated_keywords: Optional[Dict[str, bool]] = None,
) -> Dict[str, Dict]:
    """
    Compute SEO metrics for keywords.
    
    Args:
        keywords: List of keywords to score
        clusters: Cluster assignments
        intents: Intent classifications
        freq: Document frequency counts
        questions: Set of question-style keywords
        provider: SERP provider used
        serp_total_results: Optional dict of keyword -> total SERP results
        validated_keywords: Optional dict of keyword -> autocomplete validation
        
    Returns:
        Dict mapping keyword to metrics dict
    """
    validated = validated_keywords or {}
    serp_results = serp_total_results or {}
    
    # Raw proxies with autocomplete validation boost
    v_raw = {
        k: raw_volume_proxy(k, freq.get(k, 1), k in questions, validated.get(k, False)) 
        for k in keywords
    }
    d_raw = {k: raw_difficulty_proxy(k, serp_results.get(k)) for k in keywords}

    # Percentile ranking normalization (preserves mid-tier keyword value)
    def normalize_percentile(d: Dict[str, float]) -> Dict[str, float]:
        """Normalize using percentile ranking instead of min-max."""
        if not d:
            return {}
        keys = list(d.keys())
        vals = np.array([d[k] for k in keys], dtype=float)
        
        if len(vals) <= 1:
            return {k: 0.5 for k in keys}
        
        # Use percentile ranking: each value gets its percentile position
        percentiles = stats.rankdata(vals, method='average') / len(vals)
        return {k: float(p) for k, p in zip(keys, percentiles)}
    
    # Legacy min-max normalization for difficulty (competitive metric)
    def normalize_minmax(d: Dict[str, float]) -> Dict[str, float]:
        vals = np.array(list(d.values()), dtype=float)
        if len(vals) == 0:
            return {}
        vmin, vmax = float(vals.min()), float(vals.max())
        denom = (vmax - vmin) if vmax > vmin else 1.0
        return {k: float((d[k] - vmin) / denom) for k in d}

    # Use percentile for volume (preserves long-tail value)
    v_norm = normalize_percentile(v_raw)
    # Use min-max for difficulty (competitive comparison)
    d_norm = normalize_minmax(d_raw)

    # Mark as estimated unless we have real SERP data
    has_real_data = bool(serp_results)

    metrics: Dict[str, Dict] = {}
    for k in keywords:
        metrics[k] = {
            "search_volume": float(max(0.0, min(1.0, v_norm.get(k, 0.5)))),
            "difficulty": float(max(0.0, min(1.0, d_norm.get(k, 0.5)))),
            "estimated": not has_real_data,
            "validated": validated.get(k, False),
        }
    return metrics


def opportunity_scores(
    metrics: Dict[str, Dict],
    intents: Dict[str, str],
    goals: str,
    geo: str = "global",
    niche: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculate opportunity scores for keywords.
    
    Formula: search_volume * (1 - difficulty) * (business_relevance + commercial_boost)
    
    For lead-generation goals, commercial_value provides additional boost
    to high-value transactional keywords.
    
    Args:
        metrics: Dict of keyword -> metrics dict
        intents: Dict of keyword -> intent classification
        goals: Business goals string ('traffic', 'leads', etc.)
        geo: Geographic target for locale-specific scoring
        niche: Optional niche for specialized scoring
        
    Returns:
        Dict mapping keyword -> opportunity score (0.0 - 1.0)
    """
    scores = {}
    goals_lower = goals.lower()
    
    # Determine if we should prioritize commercial value
    prioritize_leads = any(k in goals_lower for k in ["lead", "sales", "revenue", "conversion"])
    
    for k, m in metrics.items():
        intent = intents.get(k, "informational")
        br = business_relevance(intent, goals)  # 0.6..1.0
        v = m.get("search_volume", 0.0)
        d = m.get("difficulty", 0.5)
        
        # Base score formula
        base_score = v * (1.0 - d) * br
        
        # Add commercial boost for lead-focused goals
        if prioritize_leads:
            cv = commercial_value(k, intent, geo, niche)
            # Commercial boost adds up to 50% to the score
            commercial_boost = cv * 0.5
            score = base_score + (base_score * commercial_boost)
        else:
            score = base_score
        
        scores[k] = float(max(0.0, min(1.0, score)))
    
    return scores
