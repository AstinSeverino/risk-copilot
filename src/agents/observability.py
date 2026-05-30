"""Langfuse observability for agent pipeline. Uses @observe decorator (Langfuse v4)."""
import os
import functools

from dotenv import load_dotenv

load_dotenv()


def _is_enabled():
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def traced(name: str = None, as_type: str = None):
    def decorator(func):
        if not _is_enabled():
            return func

        try:
            from langfuse import observe
            kwargs = {}
            if name:
                kwargs["name"] = name
            if as_type:
                kwargs["as_type"] = as_type
            return observe(**kwargs)(func)
        except Exception:
            return func

    return decorator


def flush():
    if not _is_enabled():
        return
    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass
