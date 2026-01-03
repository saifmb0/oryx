"""
GEO (Generative Engine Optimization) module for ORYX.

Optimizes content for AI-powered search engines (Google SGE, Perplexity, ChatGPT).
Provides information gain scoring, schema markup generation, and trust signal analysis.

Key Concepts:
- Information Gain: How much unique value does your content add vs. competitors?
- GEO Suitability: How well does a query match AI search patterns?
- Trust Signals: E-E-A-T signals for AI citation likelihood
"""

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# Pattern Loading from External JSON
# =============================================================================

_geo_patterns_cache: Optional[Dict] = None


def _get_geo_patterns_path() -> Path:
    """Get path to geo_patterns.json resource file."""
    return Path(__file__).parent / "resources" / "geo_patterns.json"


def _load_geo_patterns() -> Dict:
    """Load geo patterns from JSON file with caching."""
    global _geo_patterns_cache
    
    if _geo_patterns_cache is not None:
        return _geo_patterns_cache
    
    patterns_path = _get_geo_patterns_path()
    
    if patterns_path.exists():
        try:
            with open(patterns_path, "r", encoding="utf-8") as f:
                _geo_patterns_cache = json.load(f)
            logging.debug(f"Loaded geo patterns from {patterns_path}")
            return _geo_patterns_cache
        except Exception as e:
            logging.warning(f"Failed to load geo patterns from {patterns_path}: {e}")
    
    # Return empty dict if file not found or error
    _geo_patterns_cache = {"query_patterns": {}, "trust_signals": {}}
    return _geo_patterns_cache


def get_query_patterns(locale: str = "global") -> Dict[str, List[str]]:
    """
    Get query patterns for a specific locale.
    
    Merges global patterns with locale-specific patterns.
    
    Args:
        locale: Locale code (e.g., "en-AE", "en-US", "global")
        
    Returns:
        Dict mapping query type to list of regex patterns
    """
    data = _load_geo_patterns()
    patterns_data = data.get("query_patterns", {})
    
    # Start with global patterns
    result = dict(patterns_data.get("global", {}))
    
    # Merge locale-specific patterns
    if locale != "global" and locale in patterns_data:
        locale_patterns = patterns_data[locale]
        for key, patterns in locale_patterns.items():
            if key in result:
                result[key] = result[key] + patterns
            else:
                result[key] = patterns
    
    return result


def get_trust_signals(locale: str = "global") -> Dict[str, List[Tuple[str, str]]]:
    """
    Get trust signal patterns for a specific locale.
    
    Merges global signals with locale-specific signals.
    
    Args:
        locale: Locale code (e.g., "en-AE", "en-US", "global")
        
    Returns:
        Dict mapping category to list of (pattern, description) tuples
    """
    data = _load_geo_patterns()
    signals_data = data.get("trust_signals", {})
    
    # Start with global signals
    result = {}
    for category, patterns in signals_data.get("global", {}).items():
        result[category] = [(p["pattern"], p["description"]) for p in patterns]
    
    # Merge locale-specific signals
    if locale != "global" and locale in signals_data:
        locale_signals = signals_data[locale]
        for category, patterns in locale_signals.items():
            if category in result:
                result[category] = result[category] + [(p["pattern"], p["description"]) for p in patterns]
            else:
                result[category] = [(p["pattern"], p["description"]) for p in patterns]
    
    return result


# =============================================================================
# GEO Query Classification
# =============================================================================

class GeoQueryType(str, Enum):
    """Types of queries suitable for AI/generative search."""
    DIRECT_ANSWER = "direct_answer"       # Simple factual questions
    COMPLEX_RESEARCH = "complex_research"  # Multi-step research queries
    COMPARATIVE = "comparative"            # vs/comparison queries
    LOCAL = "local"                        # Location-specific queries
    TUTORIAL = "tutorial"                  # How-to queries
    LIST = "list"                          # Best/top X queries
    DEFINITION = "definition"              # What is X queries


# Default fallback patterns (used if JSON file not found)
_DEFAULT_GEO_QUERY_PATTERNS = {
    GeoQueryType.DIRECT_ANSWER: [
        r"^what is\b", r"^how much\b", r"^how long\b", r"^when did\b",
        r"\bcost of\b", r"\bprice of\b", r"\bmeaning of\b",
    ],
    GeoQueryType.COMPLEX_RESEARCH: [
        r"\bprocess of\b", r"\bsteps to\b", r"\bhow does\b.*work",
        r"\bwhat are the\b.*considerations", r"\bfactors\b",
    ],
    GeoQueryType.COMPARATIVE: [
        r"\bvs\b", r"\bversus\b", r"\bcompared to\b", r"\bdifference between\b",
        r"\bor\b(?=.*\bwhich\b)", r"\bbetter\b",
    ],
    GeoQueryType.LOCAL: [
        r"\bin dubai\b", r"\bin abu dhabi\b", r"\bin uae\b", r"\bnear me\b",
        r"\bin sharjah\b", r"\bal ain\b", r"\bmussafah\b",
    ],
    GeoQueryType.TUTORIAL: [
        r"^how to\b", r"\bguide\b", r"\btutorial\b", r"\bstep by step\b",
        r"\btips for\b", r"\bways to\b",
    ],
    GeoQueryType.LIST: [
        r"^best\b", r"^top\s+\d+", r"\blist of\b", r"\btypes of\b",
        r"\brecommended\b", r"\bpopular\b",
    ],
    GeoQueryType.DEFINITION: [
        r"^what is\b", r"^define\b", r"\bdefinition of\b",
        r"^what are\b", r"\bexplain\b",
    ],
}

# Backward compatibility alias (deprecated - use get_query_patterns())
GEO_QUERY_PATTERNS = _DEFAULT_GEO_QUERY_PATTERNS


def _build_query_patterns(locale: str = "global") -> Dict[GeoQueryType, List[str]]:
    """
    Build query patterns dict with GeoQueryType keys from JSON.
    
    Args:
        locale: Locale code for locale-specific patterns
        
    Returns:
        Dict mapping GeoQueryType to list of regex patterns
    """
    json_patterns = get_query_patterns(locale)
    
    if not json_patterns:
        return _DEFAULT_GEO_QUERY_PATTERNS
    
    # Map string keys to GeoQueryType enum
    result = {}
    for type_str, patterns in json_patterns.items():
        try:
            query_type = GeoQueryType(type_str)
            result[query_type] = patterns
        except ValueError:
            logging.debug(f"Unknown query type: {type_str}")
    
    # Fill in any missing types from defaults
    for query_type in GeoQueryType:
        if query_type not in result:
            result[query_type] = _DEFAULT_GEO_QUERY_PATTERNS.get(query_type, [])
    
    return result


def classify_geo_query(query: str, locale: str = "global") -> GeoQueryType:
    """
    Classify query type for GEO optimization.
    
    Args:
        query: Search query to classify
        locale: Locale for locale-specific patterns
        
    Returns:
        GeoQueryType classification
    """
    query_lower = query.lower()
    
    # Build patterns from JSON file (with locale support)
    patterns = _build_query_patterns(locale)
    
    # Check patterns in priority order
    for query_type, type_patterns in patterns.items():
        for pattern in type_patterns:
            if re.search(pattern, query_lower):
                return query_type
    
    return GeoQueryType.COMPLEX_RESEARCH  # Default


def calculate_geo_suitability(query: str) -> float:
    """
    Calculate how suitable a query is for AI search optimization.
    
    Higher scores indicate queries more likely to trigger AI-generated answers.
    
    Args:
        query: Search query to analyze
        
    Returns:
        Suitability score from 0.0 to 1.0
    """
    score = 0.5  # Base score
    query_lower = query.lower()
    
    # Question words boost
    question_words = ["what", "how", "why", "when", "where", "which", "who"]
    if any(query_lower.startswith(qw) for qw in question_words):
        score += 0.15
    
    # Complex query indicators
    if len(query.split()) >= 5:
        score += 0.1  # Longer queries often trigger AI
    
    if any(word in query_lower for word in ["vs", "versus", "compare", "difference"]):
        score += 0.15  # Comparison queries often get AI answers
    
    if any(word in query_lower for word in ["best", "top", "recommended"]):
        score += 0.1  # List queries trigger AI
    
    # Local intent (less likely for AI, more for maps/local pack)
    if any(word in query_lower for word in ["near me", "in dubai", "in abu dhabi"]):
        score -= 0.1
    
    # Commercial/transactional (less likely for AI answers)
    if any(word in query_lower for word in ["buy", "price", "discount", "book", "hire"]):
        score -= 0.15
    
    return max(0.0, min(1.0, score))


# =============================================================================
# Information Gain Scoring
# =============================================================================

@dataclass
class InfoGainResult:
    """Result of information gain analysis."""
    score: float  # 0.0 to 1.0
    unique_concepts: List[str]  # Concepts unique to your content
    common_concepts: List[str]  # Concepts shared with competitors
    missing_concepts: List[str]  # Concepts in competitors but not yours
    recommendations: List[str]  # Suggestions for improvement


def calculate_information_gain(
    your_content: str,
    competitor_contents: List[str],
    top_n_concepts: int = 20,
) -> InfoGainResult:
    """
    Calculate information gain of your content vs. competitors.
    
    Information gain measures how much unique value your content adds
    compared to what's already available in search results.
    
    Args:
        your_content: Your content text
        competitor_contents: List of competitor content texts
        top_n_concepts: Number of top concepts to analyze
        
    Returns:
        InfoGainResult with score and analysis
    """
    if not HAS_SKLEARN:
        return InfoGainResult(
            score=0.5,
            unique_concepts=[],
            common_concepts=[],
            missing_concepts=[],
            recommendations=["Install scikit-learn for full analysis"]
        )
    
    if not competitor_contents:
        return InfoGainResult(
            score=1.0,
            unique_concepts=[],
            common_concepts=[],
            missing_concepts=[],
            recommendations=["No competitor content to compare"]
        )
    
    # Combine all competitor content
    all_competitor_text = " ".join(competitor_contents)
    
    # Extract key concepts using TF-IDF
    all_texts = [your_content, all_competitor_text]
    
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=500,
            stop_words="english",
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top terms for each document
        your_scores = tfidf_matrix[0].toarray().flatten()
        comp_scores = tfidf_matrix[1].toarray().flatten()
        
        your_top_indices = your_scores.argsort()[-top_n_concepts:][::-1]
        comp_top_indices = comp_scores.argsort()[-top_n_concepts:][::-1]
        
        your_concepts = set(feature_names[i] for i in your_top_indices if your_scores[i] > 0)
        comp_concepts = set(feature_names[i] for i in comp_top_indices if comp_scores[i] > 0)
        
        # Calculate sets
        unique = your_concepts - comp_concepts
        common = your_concepts & comp_concepts
        missing = comp_concepts - your_concepts
        
        # Calculate similarity (lower = more unique content)
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        
        # Information gain score (1 - similarity, with adjustments)
        uniqueness_ratio = len(unique) / max(len(your_concepts), 1)
        coverage_ratio = len(common) / max(len(comp_concepts), 1)
        
        # Weighted score: reward uniqueness, penalize missing coverage
        score = 0.4 * (1 - similarity) + 0.4 * uniqueness_ratio + 0.2 * coverage_ratio
        score = max(0.0, min(1.0, score))
        
        # Generate recommendations
        recommendations = []
        if len(missing) > 5:
            recommendations.append(f"Consider covering these topics: {', '.join(list(missing)[:5])}")
        if len(unique) < 3:
            recommendations.append("Add more unique insights, data, or perspectives")
        if similarity > 0.7:
            recommendations.append("Content is very similar to competitors - differentiate more")
        
        return InfoGainResult(
            score=round(score, 3),
            unique_concepts=list(unique)[:10],
            common_concepts=list(common)[:10],
            missing_concepts=list(missing)[:10],
            recommendations=recommendations,
        )
        
    except Exception as e:
        return InfoGainResult(
            score=0.5,
            unique_concepts=[],
            common_concepts=[],
            missing_concepts=[],
            recommendations=[f"Analysis error: {str(e)}"]
        )


# =============================================================================
# Structured Data Schema Generation
# =============================================================================

@dataclass
class SchemaMarkup:
    """Generated schema markup for a page."""
    schema_type: str
    json_ld: Dict[str, Any]
    
    def to_html(self) -> str:
        """Generate HTML script tag for schema."""
        return f'<script type="application/ld+json">\n{json.dumps(self.json_ld, indent=2)}\n</script>'


def generate_faq_schema(faqs: List[Tuple[str, str]]) -> SchemaMarkup:
    """
    Generate FAQPage schema markup.
    
    Args:
        faqs: List of (question, answer) tuples
        
    Returns:
        SchemaMarkup for FAQPage
    """
    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": q,
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": a
                }
            }
            for q, a in faqs
        ]
    }
    return SchemaMarkup(schema_type="FAQPage", json_ld=schema)


def generate_howto_schema(
    title: str,
    steps: List[str],
    description: str = "",
    total_time: str = "",
    image_url: str = "",
) -> SchemaMarkup:
    """
    Generate HowTo schema markup.
    
    Args:
        title: Title of the how-to guide
        steps: List of step descriptions
        description: Overall description
        total_time: ISO 8601 duration (e.g., "PT2H30M")
        image_url: Optional image URL
        
    Returns:
        SchemaMarkup for HowTo
    """
    schema = {
        "@context": "https://schema.org",
        "@type": "HowTo",
        "name": title,
        "step": [
            {
                "@type": "HowToStep",
                "position": i + 1,
                "text": step
            }
            for i, step in enumerate(steps)
        ]
    }
    
    if description:
        schema["description"] = description
    if total_time:
        schema["totalTime"] = total_time
    if image_url:
        schema["image"] = image_url
    
    return SchemaMarkup(schema_type="HowTo", json_ld=schema)


def generate_local_business_schema(
    name: str,
    business_type: str = "HomeAndConstructionBusiness",
    address: Dict[str, str] = None,
    phone: str = "",
    url: str = "",
    price_range: str = "",
    rating: Optional[float] = None,
    review_count: int = 0,
    services: List[str] = None,
) -> SchemaMarkup:
    """
    Generate LocalBusiness schema markup.
    
    Optimized for UAE contracting businesses.
    
    Args:
        name: Business name
        business_type: Schema.org business type
        address: Dict with streetAddress, addressLocality, addressRegion, addressCountry
        phone: Contact phone
        url: Website URL
        price_range: Price range ($ to $$$$)
        rating: Average rating (1-5)
        review_count: Number of reviews
        services: List of services offered
        
    Returns:
        SchemaMarkup for LocalBusiness
    """
    schema = {
        "@context": "https://schema.org",
        "@type": business_type,
        "name": name,
    }
    
    if address:
        schema["address"] = {
            "@type": "PostalAddress",
            **address,
            "addressCountry": address.get("addressCountry", "AE"),
        }
    
    if phone:
        schema["telephone"] = phone
    if url:
        schema["url"] = url
    if price_range:
        schema["priceRange"] = price_range
    
    if rating and review_count > 0:
        schema["aggregateRating"] = {
            "@type": "AggregateRating",
            "ratingValue": rating,
            "reviewCount": review_count,
            "bestRating": 5,
        }
    
    if services:
        schema["hasOfferCatalog"] = {
            "@type": "OfferCatalog",
            "name": "Services",
            "itemListElement": [
                {"@type": "Offer", "itemOffered": {"@type": "Service", "name": s}}
                for s in services
            ]
        }
    
    return SchemaMarkup(schema_type="LocalBusiness", json_ld=schema)


def generate_service_schema(
    service_name: str,
    provider_name: str,
    description: str = "",
    area_served: List[str] = None,
    price_from: float = None,
    price_currency: str = "AED",
) -> SchemaMarkup:
    """
    Generate Service schema markup.
    
    Args:
        service_name: Name of the service
        provider_name: Name of the provider
        description: Service description
        area_served: List of areas served
        price_from: Starting price
        price_currency: Currency code
        
    Returns:
        SchemaMarkup for Service
    """
    schema = {
        "@context": "https://schema.org",
        "@type": "Service",
        "name": service_name,
        "provider": {
            "@type": "Organization",
            "name": provider_name,
        },
    }
    
    if description:
        schema["description"] = description
    
    if area_served:
        schema["areaServed"] = [
            {"@type": "City", "name": area} for area in area_served
        ]
    
    if price_from is not None:
        schema["offers"] = {
            "@type": "Offer",
            "priceSpecification": {
                "@type": "PriceSpecification",
                "price": price_from,
                "priceCurrency": price_currency,
                "minPrice": price_from,
            }
        }
    
    return SchemaMarkup(schema_type="Service", json_ld=schema)


# =============================================================================
# Trust Signal Analysis (E-E-A-T for AI)
# =============================================================================

@dataclass
class TrustSignalResult:
    """Result of trust signal analysis."""
    overall_score: float  # 0.0 to 1.0
    experience_score: float
    expertise_score: float
    authority_score: float
    trust_score: float
    signals_found: List[str]
    signals_missing: List[str]
    recommendations: List[str]


# Trust signal patterns
TRUST_SIGNAL_PATTERNS = {
    "experience": [
        (r"\byears? of experience\b", "Years of experience mentioned"),
        (r"\bcompleted\s+\d+\s+projects?\b", "Project count mentioned"),
        (r"\bcase stud(?:y|ies)\b", "Case studies present"),
        (r"\bbefore\s+and\s+after\b", "Before/after examples"),
        (r"\bclient testimonial", "Client testimonials"),
    ],
    "expertise": [
        (r"\blicensed\b", "Licensed contractor"),
        (r"\bcertified\b", "Certifications mentioned"),
        (r"\bspeciali[sz]e\b", "Specialization stated"),
        (r"\bestidama\b", "Estidama certification"),
        (r"\biso\s*\d+", "ISO certification"),
        (r"\bpearl rated?\b", "Pearl rating mentioned"),
    ],
    "authority": [
        (r"\bapproved by\b", "Approval mentioned"),
        (r"\bgovernment\s+approved\b", "Government approval"),
        (r"\btamm\s+registered\b", "TAMM registration"),
        (r"\bded\s+(?:licensed|registered)\b", "DED license"),
        (r"\baward(?:s|ed)?\b", "Awards mentioned"),
        (r"\bmember of\b", "Professional membership"),
    ],
    "trust": [
        (r"\bguarantee[ds]?\b", "Guarantee offered"),
        (r"\bwarrant(?:y|ies)\b", "Warranty mentioned"),
        (r"\binsured\b", "Insurance mentioned"),
        (r"\btransparent pricing\b", "Price transparency"),
        (r"\bno hidden\s+(fees?|costs?)\b", "No hidden costs"),
        (r"\bfree\s+(?:quote|estimate|consultation)\b", "Free quote/consultation"),
    ],
}


def _build_trust_signals(locale: str = "global") -> Dict[str, List[Tuple[str, str]]]:
    """
    Build trust signal patterns from JSON file with locale support.
    
    Args:
        locale: Locale code for locale-specific signals
        
    Returns:
        Dict mapping category to list of (pattern, description) tuples
    """
    json_signals = get_trust_signals(locale)
    
    if not json_signals:
        return _DEFAULT_TRUST_SIGNAL_PATTERNS
    
    return json_signals


# Backward compatibility alias (deprecated - use get_trust_signals())
_DEFAULT_TRUST_SIGNAL_PATTERNS = TRUST_SIGNAL_PATTERNS


def analyze_trust_signals(content: str, locale: str = "global") -> TrustSignalResult:
    """
    Analyze content for E-E-A-T trust signals.
    
    E-E-A-T: Experience, Expertise, Authority, Trust
    Important for AI citation likelihood.
    
    Args:
        content: Content to analyze
        locale: Locale for locale-specific trust signals
        
    Returns:
        TrustSignalResult with scores and recommendations
    """
    content_lower = content.lower()
    
    # Build patterns from JSON file (with locale support)
    trust_patterns = _build_trust_signals(locale)
    
    signals_found = []
    signals_by_category = {cat: [] for cat in trust_patterns}
    
    for category, patterns in trust_patterns.items():
        for pattern, description in patterns:
            if re.search(pattern, content_lower):
                signals_found.append(description)
                signals_by_category[category].append(description)
    
    # Calculate scores per category
    max_signals = {
        "experience": 5,
        "expertise": 6,
        "authority": 6,
        "trust": 6,
    }
    
    experience_score = min(len(signals_by_category.get("experience", [])) / max_signals["experience"], 1.0)
    expertise_score = min(len(signals_by_category.get("expertise", [])) / max_signals["expertise"], 1.0)
    authority_score = min(len(signals_by_category.get("authority", [])) / max_signals["authority"], 1.0)
    trust_score = min(len(signals_by_category.get("trust", [])) / max_signals["trust"], 1.0)
    
    # Overall score (weighted)
    overall_score = (
        0.2 * experience_score +
        0.3 * expertise_score +
        0.25 * authority_score +
        0.25 * trust_score
    )
    
    # Identify missing signals
    all_signals = set()
    for patterns in trust_patterns.values():
        for _, desc in patterns:
            all_signals.add(desc)
    signals_missing = list(all_signals - set(signals_found))
    
    # Generate recommendations
    recommendations = []
    if experience_score < 0.4:
        recommendations.append("Add project examples, case studies, or years of experience")
    if expertise_score < 0.4:
        recommendations.append("Highlight certifications (Estidama, ISO) and specializations")
    if authority_score < 0.4:
        recommendations.append("Mention government approvals, TAMM registration, or awards")
    if trust_score < 0.4:
        recommendations.append("Add guarantees, warranties, and transparent pricing info")
    
    return TrustSignalResult(
        overall_score=round(overall_score, 3),
        experience_score=round(experience_score, 3),
        expertise_score=round(expertise_score, 3),
        authority_score=round(authority_score, 3),
        trust_score=round(trust_score, 3),
        signals_found=signals_found,
        signals_missing=signals_missing[:10],  # Limit to top 10
        recommendations=recommendations,
    )


# =============================================================================
# Snippet Optimization for AI Citation
# =============================================================================

def generate_featured_snippet_content(
    query: str,
    query_type: GeoQueryType,
    key_points: List[str],
    source_name: str = "",
) -> str:
    """
    Generate content optimized for featured snippets and AI citations.
    
    Args:
        query: Target query
        query_type: Type of query
        key_points: Key points to include
        source_name: Name to attribute
        
    Returns:
        Optimized content snippet
    """
    query_title = query.title()
    
    if query_type == GeoQueryType.DIRECT_ANSWER:
        # Concise answer format
        answer = key_points[0] if key_points else ""
        return f"{query_title}? {answer}"
    
    elif query_type == GeoQueryType.LIST:
        # Numbered list format
        lines = [f"**{query_title}**\n"]
        for i, point in enumerate(key_points[:10], 1):
            lines.append(f"{i}. {point}")
        return "\n".join(lines)
    
    elif query_type == GeoQueryType.TUTORIAL:
        # Step-by-step format
        lines = [f"## {query_title}\n"]
        for i, point in enumerate(key_points, 1):
            lines.append(f"**Step {i}:** {point}")
        return "\n\n".join(lines)
    
    elif query_type == GeoQueryType.COMPARATIVE:
        # Comparison format
        lines = [f"## {query_title}\n"]
        for point in key_points:
            lines.append(f"- {point}")
        return "\n".join(lines)
    
    else:
        # Default paragraph format
        intro = f"{query_title} involves the following key considerations:\n\n"
        points = "\n".join(f"• {p}" for p in key_points)
        return intro + points


def optimize_for_ai_citation(
    content: str,
    target_query: str,
    max_length: int = 300,
) -> str:
    """
    Optimize a content snippet for AI citation.
    
    AI systems prefer:
    - Clear, authoritative statements
    - Specific data and numbers
    - Structured information
    - Source attribution
    
    Args:
        content: Original content
        target_query: Query to optimize for
        max_length: Maximum snippet length
        
    Returns:
        Optimized content snippet
    """
    # Clean up content
    content = content.strip()
    
    # Ensure it starts with a clear statement
    if not content[0].isupper():
        content = content.capitalize()
    
    # Truncate to optimal length
    if len(content) > max_length:
        # Find a good break point
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.6:
            content = truncated[:last_period + 1]
        else:
            content = truncated.rsplit(' ', 1)[0] + '...'
    
    return content
