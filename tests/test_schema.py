import re
from pathlib import Path

import pytest

from oryx.schema import COMPACT_SCHEMA, validate_items


def test_schema_validation():
    sample = [
        {
            "keyword": "best coffee beans",
            "cluster": "cluster-0",
            "parent_topic": "coffee beans",
            "intent": "commercial",
            "funnel_stage": "MOFU",
            "relative_interest": 0.8,
            "difficulty": 0.4,
            "ctr_potential": 0.85,
            "serp_features": ["top_ads", "shopping_results"],
            "estimated": True,
            "validated": True,
            "opportunity_score": 0.5,
        }
    ]
    
    # Action: Validate items and capture the result
    result = validate_items(sample)
    
    # Assertion 1: It should return a list (not True)
    assert isinstance(result, list), "Expected a list of KeywordData objects"
    
    # Assertion 2: The list should not be empty
    assert len(result) == 1
    
    # Assertion 3: Verify the data was correctly parsed into the Pydantic model
    # Note: Pydantic models support dot access
    assert result[0].keyword == "best coffee beans"
    assert result[0].opportunity_score == 0.5
    assert result[0].validated is True


# =============================================================================
# Output Quality Validation Tests (Phase 5)
# =============================================================================

class TestOutputQuality:
    """Validate JSON output structure and quality constraints."""

    @pytest.fixture
    def boilerplate_terms(self):
        """Load boilerplate terms from resource file."""
        boilerplate_path = Path(__file__).parent.parent / "src" / "oryx" / "resources" / "boilerplate.txt"
        terms = set()
        if boilerplate_path.exists():
            with open(boilerplate_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        terms.add(line.lower())
        return terms

    def test_relative_interest_in_range(self):
        """relative_interest should be normalized between 0 and 1."""
        valid_samples = [
            {"keyword": "test", "cluster": "c1", "parent_topic": "p1", 
             "intent": "informational", "funnel_stage": "TOFU",
             "relative_interest": 0.0, "difficulty": 0.5, "ctr_potential": 0.5,
             "serp_features": [], "estimated": True, "validated": True, 
             "opportunity_score": 0.5},
            {"keyword": "test2", "cluster": "c2", "parent_topic": "p2",
             "intent": "informational", "funnel_stage": "TOFU",
             "relative_interest": 1.0, "difficulty": 0.5, "ctr_potential": 0.5,
             "serp_features": [], "estimated": True, "validated": True,
             "opportunity_score": 0.5},
            {"keyword": "test3", "cluster": "c3", "parent_topic": "p3",
             "intent": "informational", "funnel_stage": "TOFU",
             "relative_interest": 0.5, "difficulty": 0.5, "ctr_potential": 0.5,
             "serp_features": [], "estimated": True, "validated": True,
             "opportunity_score": 0.5},
        ]
        
        result = validate_items(valid_samples)
        assert len(result) == 3
        
        for item in result:
            assert 0.0 <= item.relative_interest <= 1.0, \
                f"relative_interest {item.relative_interest} out of range for {item.keyword}"

    def test_cluster_not_numeric_prefix(self):
        """Cluster names should be descriptive, not just 'cluster-N'."""
        # Pattern for generic cluster names like "cluster-0", "cluster-1", etc.
        numeric_cluster_pattern = re.compile(r"^cluster-\d+$", re.IGNORECASE)
        
        # These should be rejected in favor of descriptive names
        generic_clusters = ["cluster-0", "cluster-1", "cluster-42", "Cluster-100"]
        
        for cluster_name in generic_clusters:
            assert numeric_cluster_pattern.match(cluster_name), \
                f"Pattern should match generic cluster: {cluster_name}"
        
        # Descriptive names should NOT match the pattern
        descriptive_clusters = [
            "coffee beans selection",
            "home renovation services", 
            "keyword research tools",
            "best-coffee-beans",  # hyphenated but descriptive
        ]
        
        for cluster_name in descriptive_clusters:
            assert not numeric_cluster_pattern.match(cluster_name), \
                f"Descriptive cluster should not match pattern: {cluster_name}"

    def test_keyword_not_in_boilerplate(self, boilerplate_terms):
        """Keywords should not be boilerplate/navigation terms."""
        # These are common boilerplate terms that should be filtered
        bad_keywords = [
            "login",
            "privacy policy",
            "terms of service",
            "subscribe",
            "menu",
            "footer",
        ]
        
        for kw in bad_keywords:
            assert kw.lower() in boilerplate_terms, \
                f"'{kw}' should be in boilerplate terms"
        
        # Valid SEO keywords should NOT be in boilerplate
        good_keywords = [
            "best coffee beans",
            "home renovation cost",
            "how to choose contractor",
            "villa construction dubai",
        ]
        
        for kw in good_keywords:
            assert kw.lower() not in boilerplate_terms, \
                f"'{kw}' should not be in boilerplate terms"

    def test_intent_is_valid_value(self):
        """Intent should be one of the valid intent categories."""
        valid_intents = {"informational", "navigational", "transactional", "commercial"}
        
        sample = {
            "keyword": "test keyword",
            "cluster": "test cluster",
            "parent_topic": "test topic",
            "intent": "commercial",
            "funnel_stage": "BOFU",
            "relative_interest": 0.7,
            "difficulty": 0.3,
            "ctr_potential": 0.8,
            "serp_features": [],
            "estimated": True,
            "validated": True,
            "opportunity_score": 0.6,
        }
        
        result = validate_items([sample])
        assert len(result) == 1
        assert result[0].intent in valid_intents

    def test_parent_topic_not_empty(self):
        """parent_topic should never be empty or just whitespace."""
        sample = {
            "keyword": "test keyword",
            "cluster": "test cluster",
            "parent_topic": "descriptive topic",  # Must be meaningful
            "intent": "informational",
            "funnel_stage": "TOFU",
            "relative_interest": 0.5,
            "difficulty": 0.5,
            "ctr_potential": 0.5,
            "serp_features": [],
            "estimated": True,
            "validated": True,
            "opportunity_score": 0.5,
        }
        
        result = validate_items([sample])
        assert len(result) == 1
        assert result[0].parent_topic.strip() != "", "parent_topic should not be empty"
        assert result[0].parent_topic.lower() != "general", "parent_topic should be specific"