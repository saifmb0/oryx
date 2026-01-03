"""
Candidate Generator module for ORYX.

Encapsulates the logic for generating keyword candidates from multiple sources:
- Document extraction (n-grams from scraped content)
- LLM-based expansion (grammar-aware keyword generation)
- PAA questions (real queries from search engines)

This module is part of the Phase 3 architecture modularization.
"""

import logging
from typing import Dict, List, Optional

from .nlp import generate_candidates as extract_from_docs
from .llm import expand_with_llm
from .scrape import get_paa_questions


class CandidateGenerator:
    """
    Generates keyword candidates from multiple sources.
    
    Combines document extraction, LLM expansion, and PAA questions
    to build a comprehensive list of keyword candidates.
    
    Usage:
        generator = CandidateGenerator(
            seed_topic="villa construction",
            audience="homeowners in Dubai",
            language="en",
            geo="ae",
        )
        candidates = generator.generate(docs)
    """
    
    def __init__(
        self,
        seed_topic: str,
        audience: str,
        language: str = "en",
        geo: str = "global",
        provider: str = "none",
        llm_provider: str = "auto",
        llm_model: Optional[str] = None,
        max_llm_results: int = 50,
        ngram_min_df: int = 2,
        top_terms_per_doc: int = 10,
    ):
        """
        Initialize the candidate generator.
        
        Args:
            seed_topic: The main topic to generate keywords for
            audience: Target audience description
            language: Language code (e.g., "en", "ar")
            geo: Geographic target (e.g., "ae", "us", "global")
            provider: SERP provider for PAA questions ("duckduckgo", "none")
            llm_provider: LLM provider for expansion ("auto", "gemini", etc.)
            llm_model: Specific LLM model to use
            max_llm_results: Maximum keywords from LLM expansion
            ngram_min_df: Minimum document frequency for n-grams
            top_terms_per_doc: Top terms to extract per document
        """
        self.seed_topic = seed_topic
        self.audience = audience
        self.language = language
        self.geo = geo
        self.provider = provider
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_llm_results = max_llm_results
        self.ngram_min_df = ngram_min_df
        self.top_terms_per_doc = top_terms_per_doc
    
    def generate(
        self,
        docs: List[Dict],
        query: Optional[str] = None,
    ) -> List[str]:
        """
        Generate keyword candidates from all sources.
        
        Args:
            docs: List of document dicts with 'url', 'title', 'text' keys
            query: Optional search query (defaults to seed_topic)
            
        Returns:
            Deduplicated list of keyword candidates
        """
        all_candidates = []
        
        # Source 1: Extract from documents
        doc_candidates = self._extract_from_documents(docs)
        all_candidates.extend(doc_candidates)
        logging.debug(f"Extracted {len(doc_candidates)} candidates from documents")
        
        # Source 2: Seed topic itself
        seed_candidates = self._get_seed_candidates()
        all_candidates.extend(seed_candidates)
        
        # Source 3: LLM expansion
        llm_candidates = self._expand_with_llm()
        all_candidates.extend(llm_candidates)
        logging.debug(f"Generated {len(llm_candidates)} candidates via LLM")
        
        # Source 4: PAA questions
        paa_candidates = self._get_paa_questions(query)
        all_candidates.extend(paa_candidates)
        if paa_candidates:
            logging.info(f"Acquired {len(paa_candidates)} real PAA questions from {self.provider}")
        
        # Deduplicate while preserving order
        return list(dict.fromkeys(all_candidates))
    
    def _extract_from_documents(self, docs: List[Dict]) -> List[str]:
        """Extract keyword candidates from document content."""
        if not docs:
            return []
        
        return extract_from_docs(
            docs,
            ngram_min_df=self.ngram_min_df,
            top_terms_per_doc=self.top_terms_per_doc,
        )
    
    def _get_seed_candidates(self) -> List[str]:
        """Get candidates from the seed topic itself."""
        return [self.seed_topic.lower()]
    
    def _expand_with_llm(self) -> List[str]:
        """Generate candidates using LLM expansion."""
        return expand_with_llm(
            self.seed_topic,
            self.audience,
            self.language,
            self.geo,
            max_results=self.max_llm_results,
            provider=self.llm_provider,
            model=self.llm_model,
        )
    
    def _get_paa_questions(self, query: Optional[str] = None) -> List[str]:
        """Fetch PAA questions from search engines."""
        if not self.provider or self.provider == "none":
            return []
        
        return get_paa_questions(
            query or self.seed_topic,
            provider=self.provider,
        )
