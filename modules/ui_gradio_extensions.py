import os
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from modules import extensions, localization, scripts, shared, util
from modules.paths import data_path, script_path


def webpath(fn):
    path = util.truncate_path(fn).replace("\\", "/")
    return f"file={quote(path, safe='/:')}?{os.path.getmtime(fn)}"


def ui_asset_paths():
    """Allow shipped frontend files even when --data-dir is outside the source.

    Gradio only allows data_path by default. Do not allow the whole source or
    extension root: they can contain API credentials and private settings.
    Recompute on UI reload so only active extensions contribute assets.
    """
    assets = {Path(script_path) / "script.js"}
    assets.update(Path(p) for p in scripts.list_files_with_name("style.css"))
    user_css = Path(data_path) / "user.css"
    if user_css.is_file():
        assets.add(user_css)
    public_suffixes = {".js", ".mjs", ".css", ".png", ".jpg", ".jpeg",
                       ".webp", ".avif", ".svg", ".gif", ".ico", ".woff", ".woff2", ".ttf"}
    for base in [Path(script_path), *(Path(ext.path) for ext in extensions.active())]:
        for name in ("javascript", "html", "styles"):
            folder = base / name
            if folder.is_dir():
                assets.update(p for p in folder.rglob("*")
                              if p.is_file() and p.suffix.lower() in public_suffixes)
    # Gradio preserves symlink spellings; Windows profile junctions and linked
    # source checkouts must work through either spelling of the same file.
    return sorted({str(name) for p in assets if p.is_file()
                   for name in (p.absolute(), p.resolve())})


def javascript_html():
    # Ensure localization is in `window` before scripts
    head = f'<script type="text/javascript">{localization.localization_js(shared.opts.localization)}</script>\n'

    script_js = os.path.join(script_path, "script.js")
    head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

    for script in scripts.list_scripts("javascript", ".js"):
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'

    for script in scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'

    if shared.cmd_opts.theme:
        head += f'<script type="text/javascript">set_theme("{shared.cmd_opts.theme}");</script>\n'

    return head


def css_html():
    head = ""

    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    for cssfile in scripts.list_files_with_name("style.css"):
        head += stylesheet(cssfile)

    user_css = os.path.join(data_path, "user.css")
    if os.path.exists(user_css):
        head += stylesheet(user_css)

    from modules.shared_gradio_themes import resolve_var

    light = resolve_var("background_fill_primary")
    dark = resolve_var("background_fill_primary_dark")
    head += f"<style>html {{ background-color: {light}; }} @media (prefers-color-scheme: dark) {{ html {{background-color:  {dark}; }} }}</style>"

    return head


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b"</head>", f'{js}<meta name="referrer" content="no-referrer"/></head>'.encode("utf8"))
        res.body = res.body.replace(b"</body>", f"{css}</body>".encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


if not hasattr(shared, "GradioTemplateResponseOriginal"):
    shared.GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
