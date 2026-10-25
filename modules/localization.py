import json
import os
from functools import lru_cache

from modules import errors, scripts

localizations = {}


def list_localizations(dirname):
    localizations.clear()

    for file in os.listdir(dirname):
        fn, ext = os.path.splitext(file)
        if ext.lower() != ".json":
            continue

        localizations[fn] = [os.path.join(dirname, file)]

    for file in scripts.list_scripts("localizations", ".json"):
        fn, ext = os.path.splitext(file.filename)
        if fn not in localizations:
            localizations[fn] = []
        localizations[fn].append(file.path)


def localization_js(current_localization_name: str) -> str:
    fns = localizations.get(current_localization_name, None)
    data = {}
    if fns is not None:
        for fn in fns:
            try:
                with open(fn, "r", encoding="utf8") as file:
                    data.update(json.load(file))
            except Exception:
                errors.report(f"Error loading localization from {fn}", exc_info=True)

    return f"window.localization = {json.dumps(data)}"


@lru_cache(maxsize=32)
def _read_translation_file(filename, mtime_ns, size):
    # Stat fields invalidate edited language files without caching an old locale.
    with open(filename, "r", encoding="utf8") as file:
        return json.load(file)


def translate(text, *args, **kwargs):
    """Translate callback messages using the selected UI dictionary, then format data.

    Source strings stay English. Format values (paths, prompts, names) are never
    looked up as translation keys. English/None and missing keys use the source.
    """
    from modules import shared

    locale = getattr(shared.opts, "localization", "None")
    translated = text
    for filename in localizations.get(locale, ()):
        try:
            stat = os.stat(filename)
            candidate = _read_translation_file(filename, stat.st_mtime_ns, stat.st_size).get(text)
            if isinstance(candidate, str):
                translated = candidate
        except Exception:
            errors.report(f"Error loading localization from {filename}", exc_info=True)
    if not args and not kwargs:
        return translated
    try:
        return translated.format(*args, **kwargs)
    except (KeyError, IndexError, ValueError):
        # A malformed translation must not interrupt a download or generation.
        return text.format(*args, **kwargs)
