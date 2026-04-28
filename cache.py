import functools
import hashlib
import os
import pickle
import time
from pathlib import Path


def _normalize(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _normalize(v)) for k, v in value.items()))
    return repr(value)


def disk_cache(ttl_hours: int = 24):
    ttl_seconds = ttl_hours * 3600
    cache_dir = Path.home() / ".td_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key_obj = (func.__module__, func.__qualname__, _normalize(args[1:]), _normalize(kwargs))
            key = hashlib.sha256(repr(key_obj).encode("utf-8")).hexdigest()
            cache_path = cache_dir / f"{key}.pkl"

            if cache_path.exists():
                age = time.time() - cache_path.stat().st_mtime
                if age <= ttl_seconds:
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper

    return decorator
