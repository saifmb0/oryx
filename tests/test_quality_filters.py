"""Tests for LinguisticsValidator garbage filtering."""

import pytest

from oryx.linguistics import LinguisticsValidator


class TestLinguisticsValidator:
    """Test suite for linguistic quality filters."""

    @pytest.fixture
    def validator(self):
        """Get singleton LinguisticsValidator instance."""
        return LinguisticsValidator()

    def test_rejects_pure_temporal(self, validator):
        """Phrases that are purely temporal patterns should be rejected."""
        temporal_phrases = [
            "2024",
            "january 2024",
            "q1 2023",
            "2023-2024",
            "march",
        ]
        for phrase in temporal_phrases:
            assert not validator.is_valid_phrase(phrase), f"Should reject: {phrase}"

    def test_rejects_navigational_garbage(self, validator):
        """Navigational or scraping artifacts should be rejected."""
        garbage_phrases = [
            "skip to content",
            "read more",
            "click here",
            "learn more",
            "previous page",
            "next article",
        ]
        for phrase in garbage_phrases:
            assert not validator.is_valid_phrase(phrase), f"Should reject: {phrase}"

    def test_rejects_ending_preposition(self, validator):
        """Phrases ending with dangling prepositions should be rejected."""
        dangling_phrases = [
            "services in",
            "company for",
            "solutions with",
            "products from",
        ]
        for phrase in dangling_phrases:
            assert not validator.is_valid_phrase(phrase), f"Should reject: {phrase}"

    def test_rejects_no_noun(self, validator):
        """Phrases without nouns or proper nouns should be rejected."""
        no_noun_phrases = [
            "very good",
            "quickly and",
            "is the",
        ]
        for phrase in no_noun_phrases:
            assert not validator.is_valid_phrase(phrase), f"Should reject: {phrase}"

    def test_accepts_valid_keywords(self, validator):
        """Valid SEO keywords should be accepted."""
        valid_phrases = [
            "home renovation services",
            "Dubai contractors",
            "kitchen remodeling cost",
            "best plumbing company",
            "commercial HVAC installation",
            "waterproofing solutions",
        ]
        for phrase in valid_phrases:
            assert validator.is_valid_phrase(phrase), f"Should accept: {phrase}"

    def test_singleton_pattern(self):
        """Validator should follow singleton pattern."""
        v1 = LinguisticsValidator()
        v2 = LinguisticsValidator()
        assert v1 is v2, "Should return same instance"

    def test_empty_phrase(self, validator):
        """Empty phrases should be rejected."""
        assert not validator.is_valid_phrase("")
        assert not validator.is_valid_phrase("   ")

    def test_single_word_noun(self, validator):
        """Single word nouns should be accepted."""
        assert validator.is_valid_phrase("contractor")
        assert validator.is_valid_phrase("renovation")
        assert validator.is_valid_phrase("Dubai")
