"""Tests for LLM verification and classification functions."""

import pytest

from keyword_lab.llm import (
    verify_candidates_with_llm,
    filter_llm_verified,
    classify_intents_with_llm,
    _parse_verification_response,
    _parse_intent_response,
    GEO_INTENT_CATEGORIES,
)


class TestVerificationParsing:
    """Test the verification response parsing logic."""
    
    def test_parse_valid_json_response(self):
        """Test parsing a well-formed JSON response."""
        keywords = ["best villa dubai", "word salad nonsense"]
        response = '{"best villa dubai": "valid", "word salad nonsense": "invalid"}'
        
        result = _parse_verification_response(response, keywords)
        
        assert result["best villa dubai"] is True
        assert result["word salad nonsense"] is False
    
    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON embedded in prose."""
        keywords = ["good keyword"]
        response = 'Here is the result:\n{"good keyword": "valid"}\nDone.'
        
        result = _parse_verification_response(response, keywords)
        
        assert result["good keyword"] is True
    
    def test_parse_boolean_values(self):
        """Test parsing boolean values in JSON."""
        keywords = ["kw1", "kw2"]
        response = '{"kw1": true, "kw2": false}'
        
        result = _parse_verification_response(response, keywords)
        
        assert result["kw1"] is True
        assert result["kw2"] is False
    
    def test_fallback_on_invalid_json(self):
        """Test fallback to all-valid on parse failure."""
        keywords = ["kw1", "kw2"]
        response = 'This is not valid JSON at all'
        
        result = _parse_verification_response(response, keywords)
        
        # Should default to True (valid) for all
        assert result["kw1"] is True
        assert result["kw2"] is True


class TestIntentParsing:
    """Test the intent classification response parsing logic."""
    
    def test_parse_valid_intent_response(self):
        """Test parsing a valid intent classification response."""
        keywords = ["buy coffee beans", "coffee shops near me"]
        response = '{"buy coffee beans": "transactional", "coffee shops near me": "local"}'
        
        result = _parse_intent_response(response, keywords)
        
        assert result["buy coffee beans"] == "transactional"
        assert result["coffee shops near me"] == "local"
    
    def test_unknown_intent_defaults_to_informational(self):
        """Test that unknown intents default to informational."""
        keywords = ["test keyword"]
        response = '{"test keyword": "unknown_category"}'
        
        result = _parse_intent_response(response, keywords)
        
        assert result["test keyword"] == "informational"


class TestVerifyWithNoProvider:
    """Test verification behavior when no LLM provider is available."""
    
    def test_provider_none_returns_all_valid(self):
        """With provider='none', all keywords should be marked valid."""
        keywords = ["kw1", "kw2", "kw3"]
        
        result = verify_candidates_with_llm(keywords, provider="none")
        
        assert all(result.values())
        assert len(result) == 3
    
    def test_empty_keywords_returns_empty_dict(self):
        """Empty input should return empty output."""
        result = verify_candidates_with_llm([], provider="none")
        
        assert result == {}
    
    def test_filter_llm_verified_with_none_provider(self):
        """filter_llm_verified with provider='none' returns all keywords."""
        keywords = ["kw1", "kw2"]
        
        result = filter_llm_verified(keywords, provider="none")
        
        assert result == keywords


class TestClassifyIntentsNoProvider:
    """Test intent classification when no LLM provider is available."""
    
    def test_provider_none_returns_informational(self):
        """With provider='none', all keywords should be classified as informational."""
        keywords = ["best coffee", "coffee near me"]
        
        result = classify_intents_with_llm(keywords, provider="none")
        
        assert all(intent == "informational" for intent in result.values())
    
    def test_empty_keywords_returns_empty_dict(self):
        """Empty input should return empty output."""
        result = classify_intents_with_llm([], provider="none")
        
        assert result == {}


class TestGEOIntentCategories:
    """Test that GEO intent categories are properly defined."""
    
    def test_required_categories_exist(self):
        """Verify all required intent categories are defined."""
        required = ["direct_answer", "complex_research", "transactional", "local", "comparative"]
        
        for cat in required:
            assert cat in GEO_INTENT_CATEGORIES
    
    def test_categories_have_descriptions(self):
        """All categories should have non-empty descriptions."""
        for cat, desc in GEO_INTENT_CATEGORIES.items():
            assert isinstance(desc, str)
            assert len(desc) > 10  # Meaningful description
