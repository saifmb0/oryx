"""Configuration loading and merging for Keyword Lab."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_default_config() -> Dict[str, Any]:
    """Load the bundled default configuration."""
    default_path = Path(__file__).parent / "default_config.yaml"
    if default_path.exists():
        return yaml.safe_load(default_path.read_text()) or {}
    return {}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration, merging user config with defaults.
    
    Priority (highest to lowest):
    1. User config file (explicit --config or ./config.yaml)
    2. Default config bundled with package
    
    Args:
        config_path: Optional path to user config file
        
    Returns:
        Merged configuration dictionary
    """
    # Start with defaults
    config = load_default_config()
    
    # Determine user config path
    user_config_path = Path(config_path) if config_path else Path("config.yaml")
    
    # Merge user config if it exists
    if user_config_path.exists():
        try:
            user_config = yaml.safe_load(user_config_path.read_text()) or {}
            config = _deep_merge(config, user_config)
            logging.debug(f"Loaded user config from {user_config_path}")
        except Exception as e:
            logging.warning(f"Failed to read config {user_config_path}: {e}")
    
    return config


def get_intent_rules(config: Dict[str, Any]) -> Dict[str, list]:
    """Get intent rules from config, with fallback defaults."""
    return config.get("intent_rules", {
        "informational": ["who", "what", "why", "how", "guide", "tutorial", "tips", "checklist", "template"],
        "commercial": ["best", "top", "review", "compare", "vs", "alternatives", "pricing"],
        "transactional": ["buy", "discount", "coupon", "deal", "near me"],
        "navigational": [],
    })


def get_question_prefixes(config: Dict[str, Any]) -> list:
    """Get question prefixes from config, with fallback defaults."""
    return config.get("question_prefixes", [
        "how", "what", "best", "vs", "for", "near me", 
        "beginner", "advanced", "guide", "checklist", "template", "why"
    ])
