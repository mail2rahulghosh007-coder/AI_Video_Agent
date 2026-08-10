# utils/retry_utils.py
# A simple, dependency-free retry decorator for LLM/API calls.
# WHY: Mistral or Sarvam API calls can fail transiently (rate limits,
# network timeouts, temporary server errors). Without retry logic, ONE
# failed call crashes the entire pipeline -- wasting all the work already
# done (download, transcription) up to that point. This wraps risky calls
# so they retry a few times with increasing delay before giving up.

import time
import functools


def retry_on_failure(max_attempts=3, initial_delay=2, backoff_factor=2, exceptions=(Exception,)):
    """
    Decorator that retries a function if it raises an exception.

    max_attempts: how many total tries before giving up
    initial_delay: seconds to wait before the first retry
    backoff_factor: multiplies the delay after each failed attempt
                     (e.g. 2s, then 4s, then 8s -- "exponential backoff",
                     standard practice for API rate-limit errors)
    exceptions: which exception types should trigger a retry
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"[retry] {func.__name__} failed (attempt {attempt}/{max_attempts}): {e}")
                        print(f"[retry] Waiting {delay}s before next attempt...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        print(f"[retry] {func.__name__} failed after {max_attempts} attempts. Giving up.")

            # All attempts failed -- re-raise the last error so the caller
            # (e.g. the Streamlit UI) can show a clear message, instead of
            # silently returning None or crashing with a confusing traceback
            raise last_exception

        return wrapper
    return decorator