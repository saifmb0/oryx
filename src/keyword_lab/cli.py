import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

from .pipeline import run_pipeline
from .config import load_config, ConfigValidationError

# Initialize rich console
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    add_completion=False, 
    no_args_is_help=False,  # Changed for interactive mode
    help="🔬 Keyword Lab - Discover and cluster SEO keywords"
)


def setup_logging(verbose: bool):
    """Setup logging with rich handler for pretty output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True, show_path=verbose)]
    )


def display_results_table(items: List[dict], max_rows: int = 25):
    """Display results in a pretty ASCII table."""
    if not items:
        console.print("[yellow]No keywords found.[/yellow]")
        return
    
    table = Table(
        title="🔑 Keyword Results",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    
    table.add_column("Keyword", style="cyan", no_wrap=False, max_width=40)
    table.add_column("Cluster", style="green")
    table.add_column("Intent", style="yellow")
    table.add_column("Funnel", style="blue")
    table.add_column("Vol", justify="right", style="white")
    table.add_column("Diff", justify="right", style="white")
    table.add_column("Opp", justify="right", style="bold green")
    
    for item in items[:max_rows]:
        table.add_row(
            item["keyword"],
            item["cluster"],
            item["intent"],
            item["funnel_stage"],
            f"{item['search_volume']:.2f}",
            f"{item['difficulty']:.2f}",
            f"{item['opportunity_score']:.2f}",
        )
    
    if len(items) > max_rows:
        table.add_row(
            f"[dim]... and {len(items) - max_rows} more[/dim]",
            "", "", "", "", "", ""
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(items)} keywords[/dim]")


@app.command()
def run(
    seed_topic: Optional[str] = typer.Option(None, "--seed-topic", help="Seed topic"),
    audience: Optional[str] = typer.Option(None, "--audience", help="Audience"),
    geo: str = typer.Option("global", "--geo"),
    language: str = typer.Option("en", "--language"),
    competitors: str = typer.Option("", "--competitors", help="Comma-separated domains"),
    business_goals: str = typer.Option("traffic, leads", "--business-goals"),
    capabilities: str = typer.Option("no-paid-apis", "--capabilities"),
    time_horizon: str = typer.Option("quarter", "--time-horizon"),
    max_clusters: int = typer.Option(8, "--max-clusters"),
    max_keywords_per_cluster: int = typer.Option(12, "--max-keywords-per-cluster"),
    sources: Optional[str] = typer.Option(None, "--sources"),
    query: Optional[str] = typer.Option(None, "--query"),
    provider: str = typer.Option("none", "--provider", case_sensitive=False),
    output: str = typer.Option("keywords.json", "--output", help="Path to JSON/CSV/XLSX file or '-' for stdout"),
    save_csv: Optional[str] = typer.Option(None, "--save-csv"),
    verbose: bool = typer.Option(False, "--verbose", help="DEBUG logs to stderr"),
    config_path: Optional[str] = typer.Option(None, "--config", help="YAML config (default: ./config.yaml if present)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log steps without network calls"),
    table_output: bool = typer.Option(False, "--table", help="Display results as a table instead of JSON"),
):
    """
    Run the keyword discovery pipeline.
    
    If --seed-topic or --audience are not provided, you'll be prompted interactively.
    """
    setup_logging(verbose)

    # Interactive mode: prompt for required fields if not provided
    if seed_topic is None:
        seed_topic = typer.prompt("🌱 Enter seed topic")
    if audience is None:
        audience = typer.prompt("👥 Enter target audience")

    # Load config with defaults and validation
    try:
        cfg = load_config(config_path)
    except ConfigValidationError as e:
        err_console.print(f"[bold red]Configuration Error:[/bold red]")
        for error in e.errors:
            err_console.print(f"  [red]•[/red] {error}")
        err_console.print("\n[dim]Check your config.yaml for typos or invalid values.[/dim]")
        sys.exit(2)

    comp_list: List[str] = [c.strip() for c in competitors.split(",") if c.strip()]

    # Show a panel with run configuration
    if not verbose:
        console.print(Panel(
            f"[bold]Seed:[/bold] {seed_topic}\n"
            f"[bold]Audience:[/bold] {audience}\n"
            f"[bold]Geo:[/bold] {geo} | [bold]Language:[/bold] {language}",
            title="🔬 Keyword Lab",
            border_style="blue",
        ))

    items = run_pipeline(
        seed_topic=seed_topic,
        audience=audience,
        geo=geo,
        language=language,
        competitors=comp_list,
        business_goals=business_goals,
        capabilities=capabilities,
        time_horizon=time_horizon,
        max_clusters=max_clusters,
        max_keywords_per_cluster=max_keywords_per_cluster,
        sources=sources,
        query=query or seed_topic,
        provider=provider,
        output=output,
        save_csv=save_csv,
        verbose=verbose,
        config=cfg,
        dry_run=dry_run,
    )

    # Output display
    if output == "-":
        # Already printed by io.write_json
        pass
    elif table_output:
        display_results_table(items)
    else:
        # Print formatted JSON to stdout
        console.print_json(json.dumps(items, ensure_ascii=False))
    
    # Summary
    if items:
        console.print(f"\n[green]✓[/green] Generated {len(items)} keywords in {len(set(i['cluster'] for i in items))} clusters")
        if output != "-":
            console.print(f"[dim]Saved to: {output}[/dim]")
    else:
        console.print("[yellow]⚠ No keywords generated. Try adding more sources or adjusting settings.[/yellow]")
        sys.exit(1)


@app.command()
def brief(
    input_file: str = typer.Argument(..., help="Path to keywords JSON file"),
    cluster: Optional[str] = typer.Option(None, "--cluster", "-c", help="Specific cluster to generate brief for"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output markdown file (default: stdout)"),
    provider: str = typer.Option("auto", "--provider", help="LLM provider for brief generation"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Generate a content brief from clustered keywords.
    
    Creates an SEO-optimized content brief with:
    - Suggested H1 title
    - Primary intent analysis
    - Key questions to answer
    - Entity/topic recommendations
    
    Example:
        keyword-lab brief keywords.json --cluster cluster-0 -o brief.md
    """
    from .llm import _detect_provider, HAS_GENAI, HAS_LITELLM
    
    setup_logging(verbose)
    
    # Load keywords file
    try:
        with open(input_file, 'r') as f:
            items = json.load(f)
    except Exception as e:
        err_console.print(f"[red]Error loading {input_file}: {e}[/red]")
        sys.exit(1)
    
    if not items:
        err_console.print("[yellow]No keywords found in file.[/yellow]")
        sys.exit(1)
    
    # Filter by cluster if specified
    if cluster:
        items = [i for i in items if i.get("cluster", "").lower() == cluster.lower()]
        if not items:
            err_console.print(f"[yellow]No keywords found in cluster '{cluster}'[/yellow]")
            sys.exit(1)
    
    # Group by cluster
    clusters_data: Dict[str, List[dict]] = {}
    for item in items:
        c = item.get("cluster", "unknown")
        if c not in clusters_data:
            clusters_data[c] = []
        clusters_data[c].append(item)
    
    # Generate briefs
    briefs = []
    for cluster_name, keywords in clusters_data.items():
        brief_md = _generate_content_brief(cluster_name, keywords, provider)
        briefs.append(brief_md)
    
    full_output = "\n\n---\n\n".join(briefs)
    
    if output:
        Path(output).write_text(full_output)
        console.print(f"[green]✓[/green] Saved brief to {output}")
    else:
        console.print(full_output)


def _generate_content_brief(cluster_name: str, keywords: List[dict], provider: str) -> str:
    """Generate a content brief for a cluster of keywords."""
    from .llm import _detect_provider, HAS_GENAI, HAS_LITELLM
    import os
    
    # Sort by opportunity score
    keywords = sorted(keywords, key=lambda k: k.get("opportunity_score", 0), reverse=True)
    
    # Extract key data
    kw_list = [k["keyword"] for k in keywords]
    intents = set(k.get("intent", "informational") for k in keywords)
    primary_intent = max(intents, key=lambda i: sum(1 for k in keywords if k.get("intent") == i))
    avg_volume = sum(k.get("search_volume", 0) for k in keywords) / len(keywords)
    avg_difficulty = sum(k.get("difficulty", 0) for k in keywords) / len(keywords)
    questions = [k["keyword"] for k in keywords if any(
        k["keyword"].startswith(q) for q in ["how", "what", "why", "when", "where", "which", "who"]
    )]
    
    # Build brief with or without LLM
    detected_provider = _detect_provider() if provider == "auto" else provider
    
    h1_suggestion = kw_list[0].title() if kw_list else cluster_name.title()
    
    brief = f"""# Content Brief: {cluster_name.replace('-', ' ').title()}

## Overview
- **Primary Intent:** {primary_intent.title()}
- **Keywords:** {len(keywords)}
- **Avg. Search Volume Score:** {avg_volume:.2f}
- **Avg. Difficulty Score:** {avg_difficulty:.2f}

## Suggested H1
> {h1_suggestion}

## Target Keywords
{chr(10).join(f"- {kw}" for kw in kw_list[:10])}
{"" if len(kw_list) <= 10 else f"- ... and {len(kw_list) - 10} more"}

## Questions to Answer
{chr(10).join(f"- {q}" for q in questions[:8]) if questions else "- No question-style keywords detected"}

## Content Recommendations
- **Funnel Stage:** {_get_dominant_funnel(keywords)}
- **Word Count:** {_suggest_word_count(primary_intent)}
- **Content Type:** {_suggest_content_type(primary_intent)}

## SEO Checklist
- [ ] Include primary keyword in H1
- [ ] Address top questions in subheadings
- [ ] Add internal links to related content
- [ ] Include relevant entities/topics
- [ ] Optimize meta description with key terms
"""
    
    # Try to enhance with LLM if available
    if detected_provider and detected_provider != "none":
        try:
            enhanced = _enhance_brief_with_llm(cluster_name, kw_list, detected_provider)
            if enhanced:
                brief += f"\n## AI-Generated Insights\n{enhanced}\n"
        except Exception as e:
            logging.debug(f"LLM brief enhancement failed: {e}")
    
    return brief


def _get_dominant_funnel(keywords: List[dict]) -> str:
    """Get the dominant funnel stage from keywords."""
    stages = [k.get("funnel_stage", "MOFU") for k in keywords]
    from collections import Counter
    return Counter(stages).most_common(1)[0][0]


def _suggest_word_count(intent: str) -> str:
    """Suggest word count based on intent."""
    counts = {
        "informational": "1500-2500 words (comprehensive guide)",
        "commercial": "1000-1500 words (comparison/review)",
        "transactional": "500-800 words (product-focused)",
        "navigational": "300-500 words (reference page)",
        "direct_answer": "300-600 words (quick answer + context)",
        "complex_research": "2000-3500 words (in-depth analysis)",
        "comparative": "1200-2000 words (detailed comparison)",
        "local": "600-1000 words (local landing page)",
    }
    return counts.get(intent, "1000-1500 words")


def _suggest_content_type(intent: str) -> str:
    """Suggest content type based on intent."""
    types = {
        "informational": "How-to Guide, Tutorial, or Explainer",
        "commercial": "Comparison Post, Review, or Listicle",
        "transactional": "Product Page, Landing Page, or Pricing Page",
        "navigational": "About Page, Contact, or FAQ",
        "direct_answer": "FAQ Entry, Snippet-optimized Post",
        "complex_research": "Ultimate Guide, Whitepaper, or Long-form Article",
        "comparative": "Versus Post, Comparison Table, or Buyer's Guide",
        "local": "Local Landing Page, Store Locator, or City Guide",
    }
    return types.get(intent, "Blog Post or Article")


def _enhance_brief_with_llm(cluster_name: str, keywords: List[str], provider: str) -> str:
    """Use LLM to add insights to the brief."""
    from .llm import HAS_GENAI, HAS_LITELLM
    import os
    
    prompt = f"""Analyze these SEO keywords for the topic "{cluster_name}" and provide:
1. A compelling meta description (150-160 chars)
2. 3 key entities/topics that MUST be mentioned for topical authority
3. 1 content angle that competitors likely miss

Keywords: {', '.join(keywords[:15])}

Be concise. Format as bullet points."""

    try:
        if provider == "gemini" and HAS_GENAI:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                resp = model.generate_content(prompt)
                return getattr(resp, "text", "")
        elif HAS_LITELLM:
            import litellm
            model = "gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "claude-3-haiku-20240307"
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.choices[0].message.content or ""
    except Exception as e:
        logging.debug(f"LLM enhancement failed: {e}")
    
    return ""


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    🔬 Keyword Lab - Discover and cluster SEO keywords from any content.
    
    Run 'keyword-lab run' to start the pipeline.
    """
    if ctx.invoked_subcommand is None:
        # If no command provided, show help
        console.print(ctx.get_help())


if __name__ == "__main__":
    app()
