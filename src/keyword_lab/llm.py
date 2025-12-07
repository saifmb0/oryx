"""LLM integration for keyword expansion.

Supports multiple providers via LiteLLM or direct API calls:
- Gemini (GEMINI_API_KEY)
- OpenAI (OPENAI_API_KEY)  
- Anthropic (ANTHROPIC_API_KEY)
- Any LiteLLM-supported provider
"""

import logging
import os
from typing import List, Optional


# LiteLLM availability
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

# Direct Gemini availability  
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def _build_prompt(seed: str, audience: str, language: str, geo: str, max_results: int) -> str:
    """Build the keyword expansion prompt."""
    return (
        "Generate a diverse, high-quality list of long-tail SEO keywords and questions as plain text, one per line.\n"
        f"Seed topic: {seed}\nAudience: {audience}\nLanguage: {language}\nGeo: {geo}\n"
        "Rules: lowercase, bigrams/trigrams or longer, focus on search intent, include questions "
        "(who/what/why/how/best/vs/for/near me/beginner/advanced/guide/checklist/template).\n"
        f"Return at most {max_results} items. No numbering, no extra prose."
    )


def _parse_response(text: str, max_results: int) -> List[str]:
    """Parse LLM response into keyword list."""
    lines = [ln.strip().lower() for ln in text.splitlines()]
    items = [ln for ln in lines if ln and len(ln.split()) >= 2]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for k in items:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:max_results]


def _detect_provider() -> Optional[str]:
    """Auto-detect available LLM provider based on environment variables."""
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def _expand_with_litellm(
    seed: str,
    audience: str,
    language: str,
    geo: str,
    max_results: int,
    model: Optional[str] = None,
) -> List[str]:
    """Expand keywords using LiteLLM (supports OpenAI, Anthropic, etc.)."""
    if not HAS_LITELLM:
        return []
    
    # Default models by detected provider
    if model is None:
        if os.getenv("OPENAI_API_KEY"):
            model = "gpt-4o-mini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            model = "claude-3-haiku-20240307"
        else:
            return []
    
    prompt = _build_prompt(seed, audience, language, geo, max_results)
    
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        return _parse_response(text, max_results)
    except Exception as e:
        logging.debug(f"LiteLLM expansion failed: {e}")
        return []


def _expand_with_gemini(
    seed: str,
    audience: str,
    language: str,
    geo: str,
    max_results: int,
    model: Optional[str] = None,
) -> List[str]:
    """Expand keywords using Google Gemini directly."""
    if not HAS_GENAI:
        return []
    
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return []
    
    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    prompt = _build_prompt(seed, audience, language, geo, max_results)
    
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model_name)
        resp = model_instance.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return _parse_response(text, max_results)
    except Exception as e:
        logging.debug(f"Gemini expansion failed: {e}")
        return []


def expand_with_llm(
    seed: str,
    audience: str = "",
    language: str = "en",
    geo: str = "global",
    max_results: int = 30,
    provider: str = "auto",
    model: Optional[str] = None,
) -> List[str]:
    """
    Expand keywords using an LLM provider.
    
    Args:
        seed: Seed topic for keyword expansion
        audience: Target audience description
        language: Language code (e.g., 'en')
        geo: Geographic target (e.g., 'global', 'us')
        max_results: Maximum keywords to generate
        provider: LLM provider ('auto', 'gemini', 'openai', 'anthropic', 'none')
        model: Optional specific model name
        
    Returns:
        List of generated keyword strings
        
    Supported environment variables:
        - GEMINI_API_KEY: For Gemini provider
        - OPENAI_API_KEY: For OpenAI provider
        - ANTHROPIC_API_KEY: For Anthropic provider
    """
    if provider == "none":
        return []
    
    # Auto-detect provider
    if provider == "auto":
        provider = _detect_provider()
        if provider is None:
            logging.warning("LLM expansion skipped: No API keys found (GEMINI_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY).")
            return []
    
    # Route to appropriate provider
    if provider == "gemini":
        return _expand_with_gemini(seed, audience, language, geo, max_results, model)
    elif provider in ("openai", "anthropic"):
        return _expand_with_litellm(seed, audience, language, geo, max_results, model)
    else:
        # Try LiteLLM for unknown providers
        if HAS_LITELLM:
            return _expand_with_litellm(seed, audience, language, geo, max_results, model or provider)
        return []


# Backward compatibility alias
def expand_with_gemini(
    seed: str,
    audience: str = "",
    language: str = "en", 
    geo: str = "global",
    max_results: int = 30,
) -> List[str]:
    """Legacy function - use expand_with_llm instead."""
    return expand_with_llm(seed, audience, language, geo, max_results, provider="gemini")
