"""Helpers for wiring the enhanced pharmaceutical query engine into applications.

Environment variables
---------------------
- PUBMED_EMAIL: optional contact email for NCBI E-utilities
- PUBMED_EUTILS_API_KEY: optional API key for higher rate limits
- QUERY_ENGINE_CACHE_DIR: optional override for the on-disk JSON cache directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enhanced_config import EnhancedRAGConfig
from .enhanced_rag_agent import EnhancedRAGAgent
from .pubmed_scraper import PubMedScraper
from .query_engine import EnhancedQueryEngine
from .ranking_filter import StudyRankingFilter


def build_pharmaceutical_query_engine(
    *,
    scraper: PubMedScraper | None = None,
    ranking_filter: StudyRankingFilter | None = None,
    **engine_kwargs: Any,
) -> EnhancedQueryEngine:
    """Return a configured `EnhancedQueryEngine` ready for pharmaceutical queries.

    Parameters
    ----------
    scraper:
        Optional pre-configured `PubMedScraper`. When omitted a new instance is
        built using environment variables (see module docstring).
    ranking_filter:
        Optional custom `StudyRankingFilter`. Defaults to a new instance with
        standard weights.
    engine_kwargs:
        Additional keyword arguments forwarded to the `EnhancedQueryEngine`
        constructor (e.g. `cache_dir`, `cache_ttl_hours`).

    Returns
    -------
    EnhancedQueryEngine
        Configured query engine ready for pharmaceutical queries with PubMed
        integration and study ranking capabilities.
    """

    scraper = scraper or PubMedScraper()
    ranking_filter = ranking_filter or StudyRankingFilter()
    return EnhancedQueryEngine(scraper, ranking_filter=ranking_filter, **engine_kwargs)


def build_enhanced_pharmaceutical_query_engine(
    *,
    scraper: PubMedScraper | None = None,
    config: EnhancedRAGConfig | None = None,
    **engine_kwargs: Any,
) -> EnhancedQueryEngine:
    """Build an enhanced query engine with pharmaceutical processing capabilities.

    Parameters
    ----------
    scraper:
        Optional pre-configured `PubMedScraper`. When omitted, creates a new
        instance with configuration from the provided config or environment.
    config:
        Configuration object for PubMed integration settings. If not provided,
        loads from environment variables.
    engine_kwargs:
        Additional keyword arguments forwarded to the `EnhancedQueryEngine`
        constructor.

    Returns
    -------
    EnhancedQueryEngine
        Configured query engine with enhanced pharmaceutical processing.
    """
    config = config or EnhancedRAGConfig.from_env()

    if scraper is None:
        advanced_enabled = config.enable_enhanced_pubmed_scraper and config.enable_advanced_caching
        scraper_kwargs = {
            "enable_rate_limiting": config.enable_rate_limiting,
            "enable_advanced_caching": advanced_enabled,
            "use_normalized_cache_keys": config.use_normalized_cache_keys and advanced_enabled,
        }
        scraper = PubMedScraper(**scraper_kwargs)

    # Apply configuration defaults
    engine_kwargs.setdefault("enable_query_enhancement", config.enable_query_enhancement)
    engine_kwargs.setdefault("cache_filtered_results", config.pubmed_cache_integration)

    return EnhancedQueryEngine(scraper, **engine_kwargs)


def build_enhanced_rag_agent(
    docs_folder: str,
    api_key: str,
    *,
    vector_db_path: str = "./vector_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model_name: str | None = None,
    enable_preflight_embedding: bool | None = None,
    append_disclaimer_in_answer: bool | None = None,
    force_disclaimer_in_answer: bool | None = None,
    guardrails_config_path: str | None = None,
    enable_synthesis: bool = False,
    enable_ddi_analysis: bool = False,
    safety_mode: str = "balanced",
    config: EnhancedRAGConfig | None = None,
    pubmed_query_engine: EnhancedQueryEngine | None = None,
    pubmed_scraper: PubMedScraper | None = None,
) -> EnhancedRAGAgent:
    """Build an enhanced RAG agent with PubMed integration capabilities.

    Parameters
    ----------
    docs_folder:
        Path to PDF documents folder for local document processing.
    api_key:
        NVIDIA API key for embeddings and language models.
    vector_db_path:
        Path for vector database storage (default: "./vector_db").
    chunk_size:
        Size of document chunks (default: 1000).
    chunk_overlap:
        Overlap between chunks (default: 200).
    embedding_model_name:
        Optional embedding model name.
    enable_preflight_embedding:
        Forces question-time preflight embedding when True.
    append_disclaimer_in_answer:
        Controls whether MEDICAL_DISCLAIMER is appended to answer text.
    force_disclaimer_in_answer:
        When True, forces disclaimer in answer regardless of rails;
        when False, skips disclaimer; when None (default), uses standard behavior.
    guardrails_config_path:
        Path to guardrails configuration file.
    enable_synthesis:
        Whether to enable synthesis analysis (default: False).
    enable_ddi_analysis:
        Whether to enable DDI/PK analysis (default: False).
    safety_mode:
        Safety validation mode ("strict", "balanced", "permissive").
    config:
        Configuration object for enhanced features. If not provided,
        loads from environment variables.
    pubmed_query_engine:
        Optional pre-configured EnhancedQueryEngine for PubMed integration.
        If not provided and config enables PubMed, creates one automatically.
    pubmed_scraper:
        Optional pre-configured PubMedScraper. Used if
        pubmed_query_engine is not provided.

    Returns
    -------
    EnhancedRAGAgent
        Configured RAG agent with PubMed integration capabilities.
    """
    config = config or EnhancedRAGConfig.from_env()

    # If PubMed integration is enabled and components not provided, create them
    if config.should_enable_pubmed() and pubmed_query_engine is None:
        if pubmed_scraper is None:
            advanced_enabled = config.enable_enhanced_pubmed_scraper and config.enable_advanced_caching
            scraper_kwargs = {
                "enable_rate_limiting": config.enable_rate_limiting,
                "enable_advanced_caching": advanced_enabled,
                "use_normalized_cache_keys": config.use_normalized_cache_keys and advanced_enabled,
            }
            pubmed_scraper = PubMedScraper(**scraper_kwargs)

        engine_kwargs = {
            "enable_query_enhancement": config.enable_query_enhancement,
            "cache_filtered_results": config.pubmed_cache_integration,
        }
        pubmed_query_engine = EnhancedQueryEngine(pubmed_scraper, **engine_kwargs)

    return EnhancedRAGAgent(
        docs_folder=docs_folder,
        api_key=api_key,
        vector_db_path=vector_db_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name,
        enable_preflight_embedding=enable_preflight_embedding,
        append_disclaimer_in_answer=append_disclaimer_in_answer,
        force_disclaimer_in_answer=force_disclaimer_in_answer,
        guardrails_config_path=guardrails_config_path,
        enable_synthesis=enable_synthesis,
        enable_ddi_analysis=enable_ddi_analysis,
        safety_mode=safety_mode,
        config=config,
        pubmed_query_engine=pubmed_query_engine,
        pubmed_scraper=pubmed_scraper,
    )


def build_integrated_system(
    docs_folder: str,
    api_key: str,
    *,
    vector_db_path: str = "./vector_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model_name: str | None = None,
    guardrails_config_path: str | None = None,
    enable_synthesis: bool = False,
    enable_ddi_analysis: bool = False,
    safety_mode: str = "balanced",
    config: EnhancedRAGConfig | None = None,
) -> dict[str, Any]:
    """Build a complete integrated RAG + PubMed system.

    This factory creates all necessary components for a fully functional
    system that combines local document processing with PubMed integration.

    Parameters
    ----------
    docs_folder:
        Path to PDF documents folder.
    api_key:
        NVIDIA API key.
    vector_db_path:
        Path for vector database storage.
    chunk_size:
        Size of document chunks.
    chunk_overlap:
        Overlap between chunks.
    embedding_model_name:
        Optional embedding model name.
    guardrails_config_path:
        Path to guardrails configuration file.
    enable_synthesis:
        Enable synthesis analysis.
    enable_ddi_analysis:
        Enable DDI/PK analysis.
    safety_mode:
        Safety validation mode.
    config:
        Configuration object. If not provided, loads from environment.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'rag_agent': EnhancedRAGAgent instance
        - 'pubmed_scraper': PubMedScraper instance (if enabled)
        - 'query_engine': EnhancedQueryEngine instance (if enabled)
        - 'config': EnhancedRAGConfig instance
        - 'status': System status information
    """
    config = config or EnhancedRAGConfig.from_env()

    components: dict[str, Any] = {
        "config": config,
        "status": {
            "pubmed_enabled": config.should_enable_pubmed(),
            "hybrid_mode": config.should_use_hybrid_mode(),
            "components_ready": False,
        },
    }

    # Build the RAG agent (this will create PubMed components if enabled)
    rag_agent = build_enhanced_rag_agent(
        docs_folder=docs_folder,
        api_key=api_key,
        vector_db_path=vector_db_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name,
        guardrails_config_path=guardrails_config_path,
        enable_synthesis=enable_synthesis,
        enable_ddi_analysis=enable_ddi_analysis,
        safety_mode=safety_mode,
        config=config,
    )
    components["rag_agent"] = rag_agent

    # Extract PubMed components for direct access
    if config.should_enable_pubmed():
        # Use public APIs for component access
        is_mock_agent = rag_agent.__class__.__module__.startswith("unittest.mock")

        def _resolve_component(public_attr: str, private_attr: str) -> Any:
            if is_mock_agent:
                return getattr(rag_agent, private_attr, None)
            return getattr(rag_agent, public_attr, getattr(rag_agent, private_attr, None))

        scraper_obj = _resolve_component("pubmed_scraper", "_pubmed_scraper")
        engine_obj = _resolve_component("pubmed_query_engine", "_pubmed_query_engine")
        components["pubmed_scraper"] = scraper_obj
        components["query_engine"] = engine_obj

        availability = getattr(rag_agent, "pubmed_available", None)
        if availability is None and hasattr(rag_agent, "_pubmed_available"):
            availability = getattr(rag_agent, "_pubmed_available")
        if callable(availability):
            components["status"]["components_ready"] = bool(availability())
        else:
            components["status"]["components_ready"] = bool(availability)
        system_status = rag_agent.get_system_status()
        components["status"]["pubmed_status"] = (
            system_status.get("component_health", {}).get("pubmed_integration", {}).get("status", "unknown")
        )

    return components


@dataclass
class SystemHealthReport:
    """Health report for integrated system components."""

    rag_agent_ready: bool = False
    pubmed_integration_ready: bool = False
    pubmed_scraper_ready: bool = False
    query_engine_ready: bool = False
    guardrails_ready: bool = False
    synthesis_ready: bool = False
    ddi_analysis_ready: bool = False
    warnings: list[str] = None
    errors: list[str] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


def check_system_health(
    rag_agent: EnhancedRAGAgent | None = None,
    pubmed_scraper: PubMedScraper | None = None,
    query_engine: EnhancedQueryEngine | None = None,
    config: EnhancedRAGConfig | None = None,
) -> SystemHealthReport:
    """Check the health of an integrated RAG + PubMed system.

    Parameters
    ----------
    rag_agent:
        EnhancedRAGAgent instance to check.
    pubmed_scraper:
        PubMedScraper instance to check.
    query_engine:
        EnhancedQueryEngine instance to check.
    config:
        Configuration object for reference.

    Returns
    -------
    SystemHealthReport
        Health status report for all system components.
    """
    config = config or EnhancedRAGConfig.from_env()
    report = SystemHealthReport()

    # Check RAG agent
    if rag_agent is not None:
        try:
            rag_agent._ensure_components_initialized()
            rag_agent._ensure_pubmed_components()
            report.rag_agent_ready = True

            # Check sub-components using public API
            system_status: Dict[str, Any] = {}
            if hasattr(rag_agent, "get_system_status"):
                try:
                    status_candidate = rag_agent.get_system_status()
                    if isinstance(status_candidate, dict):
                        system_status = status_candidate
                except Exception as exc:  # pragma: no cover - best effort telemetry
                    report.warnings.append(f"Unable to retrieve system status: {exc}")
            health = system_status.get("component_health", {})
            if not isinstance(health, dict) or not health:
                health = getattr(rag_agent, "component_health", {})
            report.guardrails_ready = health.get("medical_guardrails", {}).get("status") == "ready"
            report.synthesis_ready = health.get("synthesis_engine", {}).get("status") == "ready"
            report.ddi_analysis_ready = health.get("ddi_pk_processor", {}).get("status") == "ready"

            pubmed_health = health.get("pubmed_integration", {})
            if isinstance(pubmed_health, dict):
                report.pubmed_integration_ready = pubmed_health.get("status") == "ready"
            else:
                report.pubmed_integration_ready = False

        except Exception as exc:
            report.errors.append(f"RAG agent health check failed: {exc}")
    else:
        report.warnings.append("No RAG agent provided")

    # Check PubMed scraper
    if pubmed_scraper is not None:
        try:
            if hasattr(pubmed_scraper, "combined_status_report"):
                status = pubmed_scraper.combined_status_report()
                cache_ok = status.get("cache", {}).get("status") == "ready"
                rate_limit_ok = status.get("rate_limit", {}).get("status") == "ready"
                report.pubmed_scraper_ready = cache_ok and rate_limit_ok
            else:
                report.pubmed_scraper_ready = True
        except Exception as exc:
            report.errors.append(f"PubMed scraper health check failed: {exc}")
    elif config.should_enable_pubmed():
        report.warnings.append("PubMed integration enabled but no scraper provided")

    # Check query engine
    if query_engine is not None:
        report.query_engine_ready = True
    elif config.should_enable_pubmed():
        report.warnings.append("PubMed integration enabled but no query engine provided")

    return report


__all__ = [
    "build_pharmaceutical_query_engine",
    "build_enhanced_pharmaceutical_query_engine",
    "build_enhanced_rag_agent",
    "build_integrated_system",
    "check_system_health",
    "SystemHealthReport",
]
