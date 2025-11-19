"""
NVIDIA Build API LLM Client Wrapper for Pharmaceutical RAG Chatbot

This module provides a focused wrapper for NVIDIA Build API LLM chat completions,
specifically designed for pharmaceutical and medical applications requiring high accuracy.

Supported Models:
    - meta/llama-3.1-8b-instruct (8B): Fast, efficient for simple queries
    - meta/llama-3.3-70b-instruct (70B): Advanced reasoning for complex medical queries

API Documentation: https://build.nvidia.com/
Rate Limits: 10,000 requests/month (free tier)

Pharmaceutical Optimizations:
    - Lower default temperature (0.3) for medical accuracy and consistency
    - Conservative max_tokens (800) for focused responses
    - Comprehensive logging for audit trails and debugging
    - Retry logic for transient API failures

Warning:
    This client is designed for pharmaceutical information retrieval and should not
    be used as a substitute for professional medical advice, diagnosis, or treatment.
    All outputs should be reviewed by qualified healthcare professionals.

Example:
    >>> from src.nvidia_llm_client import NVIDIALLMClient
    >>> client = NVIDIALLMClient()
    >>> response = client.generate_simple("What is aspirin?", model="8b")
    >>> print(response)
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional


# Logger setup
logger = logging.getLogger(__name__)


# Custom Exceptions
class NVIDIALLMError(Exception):
    """Base exception for NVIDIA LLM client errors."""
    pass


class NVIDIALLMAPIError(NVIDIALLMError):
    """
    API-specific errors from NVIDIA Build API.

    Raised for:
        - Rate limit exceeded (429)
        - Authentication failures (401)
        - Model unavailable (404)
        - Server errors (5xx)
        - Timeout errors
    """
    def __init__(self, message: str, status_code: Optional[int] = None,
                 model: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.model = model
        self.details = details or {}
        super().__init__(message)


class NVIDIALLMConfigError(NVIDIALLMError):
    """
    Configuration errors for NVIDIA LLM client.

    Raised for:
        - Missing API key
        - Invalid model selection
        - Invalid base URL
        - Missing OpenAI SDK dependency
    """
    pass


# Deferred OpenAI import to avoid dependency errors at module load time
def _get_openai_client(api_key: str, base_url: str):
    """
    Deferred import and initialization of OpenAI client.

    This follows the pattern from openai_wrapper.py to avoid import errors
    when the OpenAI SDK is not installed.

    Args:
        api_key: NVIDIA Build API key
        base_url: NVIDIA Build API base URL

    Returns:
        Initialized OpenAI client instance

    Raises:
        NVIDIALLMConfigError: If OpenAI SDK is not installed
    """
    try:
        from openai import OpenAI
        logger.debug("OpenAI SDK imported successfully")
        return OpenAI(api_key=api_key, base_url=base_url)
    except ImportError as e:
        error_msg = (
            "OpenAI SDK not installed. Install with: pip install openai\n"
            "The NVIDIA Build API uses OpenAI-compatible endpoints."
        )
        logger.error(f"Failed to import OpenAI SDK: {e}")
        raise NVIDIALLMConfigError(error_msg) from e


class NVIDIALLMClient:
    """
    Client for NVIDIA Build API LLM chat completions.

    This client provides a focused wrapper for chat completions using NVIDIA's
    free LLM API, with pharmaceutical optimizations and robust error handling.

    Attributes:
        MODELS: Mapping of model shortcuts to full model IDs
        BASE_URL: Default NVIDIA Build API base URL
        DEFAULT_TEMPERATURE: Conservative temperature for medical accuracy (0.3)
        DEFAULT_MAX_TOKENS: Default token limit for responses (800)
        MAX_RETRIES: Maximum retry attempts for transient errors (3)
        RETRY_DELAY_SECONDS: Initial retry delay in seconds (1.0)

    Example:
        >>> client = NVIDIALLMClient()
        >>> messages = [{"role": "user", "content": "What is aspirin?"}]
        >>> response = client.generate(messages, model="8b")
        >>> print(response)
    """

    # Model configuration
    MODELS = {
        "8b": "meta/llama-3.1-8b-instruct",
        "70b": "meta/llama-3.3-70b-instruct"
    }

    # API configuration
    BASE_URL = "https://integrate.api.nvidia.com/v1"

    # Pharmaceutical defaults - optimized for medical accuracy
    DEFAULT_TEMPERATURE = 0.3  # Lower than typical 0.7 for consistency
    DEFAULT_MAX_TOKENS = 800  # Sufficient for patient answers

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_logging: bool = True
    ):
        """
        Initialize NVIDIA LLM client.

        Args:
            api_key: NVIDIA Build API key. If not provided, reads from NVIDIA_API_KEY
                    environment variable. Get your key at https://build.nvidia.com/
            base_url: Custom base URL for NVIDIA Build API. Defaults to official endpoint.
            enable_logging: Enable detailed logging for debugging and audit trails

        Raises:
            NVIDIALLMConfigError: If API key is missing or OpenAI SDK not installed
        """
        # Setup logging
        self.enable_logging = enable_logging
        if not self.enable_logging:
            logger.setLevel(logging.WARNING)

        # Retrieve API key
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            error_msg = (
                "NVIDIA API key not found. Please provide api_key parameter or set "
                "NVIDIA_API_KEY environment variable.\n"
                "Get your free API key at: https://build.nvidia.com/"
            )
            logger.error("NVIDIA API key missing")
            raise NVIDIALLMConfigError(error_msg)

        # Set base URL
        self.base_url = base_url or self.BASE_URL

        # Initialize OpenAI client with deferred import
        try:
            self.client = _get_openai_client(self.api_key, self.base_url)
            logger.info(f"NVIDIA LLM client initialized: base_url={self.base_url}")
        except NVIDIALLMConfigError:
            raise
        except Exception as e:
            error_msg = f"Failed to initialize NVIDIA LLM client: {str(e)}"
            logger.error(error_msg)
            raise NVIDIALLMConfigError(error_msg) from e

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "8b",
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion using NVIDIA Build API.

        This method handles model routing, retry logic, and comprehensive error handling
        for robust pharmaceutical applications.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [{"role": "user", "content": "What is aspirin?"}]
            model: Model shorthand ("8b" or "70b"). Defaults to "8b" for efficiency.
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.
                        Defaults to 0.3 for medical accuracy.
            max_tokens: Maximum tokens to generate. Defaults to 800.
            **kwargs: Additional parameters passed to OpenAI chat completion API

        Returns:
            Generated text string from the model

        Raises:
            NVIDIALLMAPIError: API errors (rate limits, auth, model unavailable)
            NVIDIALLMError: Other generation errors
        """
        # Validate and map model
        if model not in self.MODELS:
            logger.warning(f"Invalid model '{model}', defaulting to '8b'")
            model = "8b"
        full_model_id = self.MODELS[model]

        # Set defaults
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS

        # Log request
        logger.info(
            f"Generating with model={full_model_id}, temperature={temperature}, "
            f"max_tokens={max_tokens}, messages={len(messages)}"
        )

        # Generate with retry logic
        try:
            start_time = time.time()

            response = self._retry_with_backoff(
                self.client.chat.completions.create,
                model=full_model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Extract generated text
            if not response.choices:
                raise NVIDIALLMAPIError(
                    "No choices returned in API response",
                    model=full_model_id
                )

            generated_text = response.choices[0].message.content
            if generated_text is None:
                raise NVIDIALLMAPIError(
                    "Empty content in API response",
                    model=full_model_id
                )

            # Log success
            elapsed_ms = (time.time() - start_time) * 1000
            token_usage = getattr(response, 'usage', None)
            tokens_used = token_usage.total_tokens if token_usage else "unknown"

            logger.info(
                f"Generated {len(generated_text)} characters in {elapsed_ms:.0f}ms "
                f"using {full_model_id} (tokens: {tokens_used})"
            )

            return generated_text

        except NVIDIALLMAPIError:
            raise
        except Exception as e:
            error_msg = f"Generation failed: {str(e)} (model={full_model_id})"
            logger.error(error_msg, exc_info=True)
            raise NVIDIALLMError(error_msg) from e

    def generate_simple(
        self,
        prompt: str,
        model: str = "8b",
        temperature: float = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Convenience method for single-turn generation.

        This method simplifies common use cases where you just need to send a prompt
        and get a response, without manually constructing message lists.

        Args:
            prompt: User prompt/question
            model: Model shorthand ("8b" or "70b")
            temperature: Sampling temperature. Defaults to 0.3.
            system_prompt: Optional system prompt to set context/behavior
            **kwargs: Additional parameters passed to generate()

        Returns:
            Generated text string

        Example:
            >>> client = NVIDIALLMClient()
            >>> answer = client.generate_simple(
            ...     "What is aspirin?",
            ...     model="8b",
            ...     system_prompt="You are a pharmaceutical expert."
            ... )
        """
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call main generate method
        return self.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs
        )

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry logic.

        Retries on transient errors:
            - Rate limits (429)
            - Timeouts
            - Server errors (5xx)

        Does not retry on:
            - Authentication errors (401)
            - Not found errors (404)
            - Bad request errors (400)

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            NVIDIALLMAPIError: After max retries exceeded or on permanent errors
        """
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if error is retryable
                is_retryable = False
                status_code = None

                # Extract status code from OpenAI SDK exceptions
                if hasattr(e, 'status_code'):
                    status_code = e.status_code
                    # Retry on rate limits (429) and server errors (5xx)
                    is_retryable = status_code == 429 or status_code >= 500
                elif "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    is_retryable = True

                # Don't retry on permanent errors
                if status_code in [401, 403, 404, 400]:
                    error_msg = f"Permanent error ({status_code}): {str(e)}"
                    logger.error(error_msg)
                    raise NVIDIALLMAPIError(
                        error_msg,
                        status_code=status_code,
                        details={"attempt": attempt + 1}
                    ) from e

                # If not retryable or last attempt, raise
                if not is_retryable or attempt == self.MAX_RETRIES - 1:
                    error_msg = f"Max retries exceeded: {str(e)}"
                    logger.error(error_msg)
                    raise NVIDIALLMAPIError(
                        error_msg,
                        status_code=status_code,
                        details={"attempts": attempt + 1}
                    ) from e

                # Calculate exponential backoff delay
                delay = self.RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{self.MAX_RETRIES} after {delay}s: {str(e)}"
                )
                time.sleep(delay)

        # Should never reach here, but just in case
        raise NVIDIALLMAPIError(
            f"Max retries exceeded: {str(last_exception)}",
            details={"attempts": self.MAX_RETRIES}
        ) from last_exception

    def test_connection(self, model: str = "8b") -> Dict[str, Any]:
        """
        Test API connectivity with a simple generation.

        Useful for validating API key, checking model availability, and measuring
        response times.

        Args:
            model: Model shorthand to test ("8b" or "70b")

        Returns:
            Dict with test results:
                - success (bool): Whether test succeeded
                - model_tested (str): Full model ID tested
                - response_time_ms (float): Response time in milliseconds
                - response_text (str): Generated text (if successful)
                - error (str): Error message (if failed)

        Example:
            >>> client = NVIDIALLMClient()
            >>> result = client.test_connection(model="8b")
            >>> if result["success"]:
            ...     print(f"API working! Response time: {result['response_time_ms']}ms")
        """
        test_prompt = "What is aspirin? Answer in one sentence."
        model_id = self.MODELS.get(model, self.MODELS["8b"])

        result = {
            "success": False,
            "model_tested": model_id,
            "response_time_ms": 0.0,
            "response_text": None,
            "error": None
        }

        try:
            start_time = time.time()
            response_text = self.generate_simple(test_prompt, model=model)
            elapsed_ms = (time.time() - start_time) * 1000

            result["success"] = True
            result["response_time_ms"] = elapsed_ms
            result["response_text"] = response_text
            logger.info(f"Connection test successful: {model_id} ({elapsed_ms:.0f}ms)")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Connection test failed for {model_id}: {str(e)}")

        return result


# Module-level convenience functions

def generate_with_nvidia(
    prompt: str,
    model: str = "8b",
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Quick generation wrapper for one-off requests.

    Creates a client instance and generates a response without requiring
    manual client management. Useful for simple scripts and notebooks.

    Args:
        prompt: User prompt/question
        model: Model shorthand ("8b" or "70b")
        api_key: Optional API key (defaults to NVIDIA_API_KEY env var)
        **kwargs: Additional parameters passed to generate_simple()

    Returns:
        Generated text string

    Raises:
        NVIDIALLMConfigError: If API key is missing
        NVIDIALLMAPIError: If generation fails

    Example:
        >>> from src.nvidia_llm_client import generate_with_nvidia
        >>> answer = generate_with_nvidia("What is aspirin?", model="8b")
        >>> print(answer)
    """
    client = NVIDIALLMClient(api_key=api_key)
    return client.generate_simple(prompt, model=model, **kwargs)


def test_nvidia_llm_access(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick test function for API access validation.

    Tests connectivity to both 8B and 70B models, measuring response times
    and checking for common configuration issues.

    Args:
        api_key: Optional API key (defaults to NVIDIA_API_KEY env var)

    Returns:
        Dict with comprehensive test results:
            - overall_success (bool): Whether all tests passed
            - 8b_result (dict): Test result for 8B model
            - 70b_result (dict): Test result for 70B model
            - api_key_configured (bool): Whether API key is present
            - errors (list): List of error messages

    Example:
        >>> from src.nvidia_llm_client import test_nvidia_llm_access
        >>> results = test_nvidia_llm_access()
        >>> print(f"API Status: {'✓' if results['overall_success'] else '✗'}")
        >>> print(f"8B Model: {results['8b_result']['response_time_ms']}ms")
        >>> print(f"70B Model: {results['70b_result']['response_time_ms']}ms")
    """
    results = {
        "overall_success": False,
        "api_key_configured": bool(api_key or os.getenv("NVIDIA_API_KEY")),
        "8b_result": None,
        "70b_result": None,
        "errors": []
    }

    if not results["api_key_configured"]:
        results["errors"].append(
            "API key not configured. Set NVIDIA_API_KEY environment variable or "
            "pass api_key parameter. Get your key at: https://build.nvidia.com/"
        )
        return results

    try:
        client = NVIDIALLMClient(api_key=api_key)

        # Test 8B model
        logger.info("Testing 8B model...")
        results["8b_result"] = client.test_connection(model="8b")
        if not results["8b_result"]["success"]:
            results["errors"].append(f"8B model test failed: {results['8b_result']['error']}")

        # Test 70B model
        logger.info("Testing 70B model...")
        results["70b_result"] = client.test_connection(model="70b")
        if not results["70b_result"]["success"]:
            results["errors"].append(f"70B model test failed: {results['70b_result']['error']}")

        # Overall success if both models work
        results["overall_success"] = (
            results["8b_result"]["success"] and
            results["70b_result"]["success"]
        )

    except Exception as e:
        error_msg = f"Client initialization failed: {str(e)}"
        results["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)

    return results


# Module exports
__all__ = [
    "NVIDIALLMClient",
    "NVIDIALLMError",
    "NVIDIALLMAPIError",
    "NVIDIALLMConfigError",
    "generate_with_nvidia",
    "test_nvidia_llm_access"
]


# Main block for quick testing
if __name__ == "__main__":
    import json

    print("=" * 70)
    print("NVIDIA Build API LLM Client - Connection Test")
    print("=" * 70)
    print()

    # Run comprehensive test
    results = test_nvidia_llm_access()

    # Pretty print results
    print(json.dumps(results, indent=2, default=str))
    print()

    if results["overall_success"]:
        print("✓ All tests passed! NVIDIA Build API is accessible.")
        print(f"  - 8B Model: {results['8b_result']['response_time_ms']:.0f}ms")
        print(f"  - 70B Model: {results['70b_result']['response_time_ms']:.0f}ms")
    else:
        print("✗ Some tests failed. Check errors above.")
        for error in results["errors"]:
            print(f"  - {error}")

    print()
    print("=" * 70)
