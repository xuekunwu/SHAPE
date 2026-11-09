"""The fatal class of LLM errors: the basic problems that make every further call fail, so a run stops on them.

Fatal (`is_fatal_llm_error`): no engine configured (`NoEngineError`), authentication / permission (an invalid key),
the model not found, the endpoint unreachable (a connection error after the client's retries). Classified by
exception type, never by message text; an exception counts when it, or any exception it was raised from or during
(``__cause__`` / ``__context__``; a tenacity RetryError's last attempt), is of a fatal type. A single timeout
(``APITimeoutError``, a subclass of the connection error) is not fatal. Every other failure of one call (a malformed
or unparseable answer, a refusal, a timeout, a rate limit) is that call's failure: the caller records it and the run
goes on.
"""
from __future__ import annotations


class NoEngineError(RuntimeError):
    """A question or a vision call with no LLM engine configured (a system error: the run stops)."""


def _fatal_types() -> tuple[tuple, tuple]:
    """(fatal types, the non-fatal subtypes excluded from them), from the provider SDKs that are installed."""
    fatal, excluded = [NoEngineError], []
    try:
        import openai

        fatal += [openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError,
                  openai.APIConnectionError]
        excluded.append(openai.APITimeoutError)
    except ImportError:
        pass
    try:
        import anthropic

        fatal += [anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                  anthropic.APIConnectionError]
        excluded.append(anthropic.APITimeoutError)
    except ImportError:
        pass
    return tuple(fatal), tuple(excluded)


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        last = getattr(exc, "last_attempt", None)         # tenacity RetryError
        inner = last.exception() if last is not None and callable(getattr(last, "exception", None)) else None
        exc = inner or exc.__cause__ or exc.__context__


def is_fatal_llm_error(exc: BaseException) -> bool:
    """Whether `exc` (or an exception in its chain) is of the fatal class (the module docstring)."""
    fatal, excluded = _fatal_types()
    return any(isinstance(e, fatal) and not isinstance(e, excluded) for e in _chain(exc))
