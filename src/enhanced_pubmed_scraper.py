"""
Compatibility shim for the legacy EnhancedPubMedScraper.

The advanced caching and rate limiting features now live directly inside
``src.pubmed_scraper.PubMedScraper``.  This wrapper simply forwards all
parameters to the unified implementation while emitting a deprecation warning.
"""
from __future__ import annotations

import warnings
from typing import Any

from .pubmed_scraper import PubMedScraper

try:
    from .cache_management import NCBICacheManager
except Exception:  # pragma: no cover - optional dependency
    NCBICacheManager = None  # type: ignore


class EnhancedPubMedScraper(PubMedScraper):
    """Deprecated alias for :class:`PubMedScraper`.

    The constructor mirrors the old signature but forces advanced caching on so
    existing integrations continue to benefit from the richer cache manager.
    """

    def __init__(
        self,
        *args: Any,
        cache_manager: NCBICacheManager | None = None,
        enable_rate_limiting: bool | None = None,
        enable_advanced_caching: bool | None = None,
        use_normalized_cache_keys: bool | None = None,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "EnhancedPubMedScraper is deprecated; use PubMedScraper with " "enable_advanced_caching=True instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if enable_rate_limiting is not None:
            kwargs.setdefault("enable_rate_limiting", enable_rate_limiting)
        kwargs.setdefault(
            "enable_advanced_caching",
            enable_advanced_caching if enable_advanced_caching is not None else True,
        )
        if cache_manager is not None:
            kwargs.setdefault("cache_manager", cache_manager)
        kwargs.setdefault(
            "use_normalized_cache_keys",
            use_normalized_cache_keys if use_normalized_cache_keys is not None else True,
        )

        super().__init__(*args, **kwargs)
