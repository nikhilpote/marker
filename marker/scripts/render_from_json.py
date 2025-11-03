import json
from collections.abc import Iterable
from pathlib import Path
from typing import List

import click
from bs4 import BeautifulSoup

from marker.renderers.json import JSONBlockOutput
from marker.settings import settings


def _load_blocks(json_path: Path) -> List[JSONBlockOutput]:
    raw_text = json_path.read_text(encoding=settings.OUTPUT_ENCODING)
    data = json.loads(raw_text)
    if isinstance(data, dict):
        children = data.get("children", [])
    elif isinstance(data, list):
        children = data
    else:
        raise ValueError("JSON root must be a list or an object with a 'children' key.")
    return [JSONBlockOutput.model_validate(child) for child in children]


def _guess_mime_type() -> str:
    fmt = settings.OUTPUT_IMAGE_FORMAT.lower()
    if fmt in {"jpg", "jpeg"}:
        return "image/jpeg"
    if fmt == "png":
        return "image/png"
    if fmt == "webp":
        return "image/webp"
    return f"image/{fmt}"


def _image_html(block_id: str, payload: str) -> str:
    alt_text = block_id.split("/")[-1]
    mime_type = _guess_mime_type()
    return (
        f'<figure data-block-id="{block_id}">'
        f'<img src="data:{mime_type};base64,{payload}" alt="{alt_text}"/>'
        "</figure>"
    )


def _render_block(block: JSONBlockOutput) -> str:
    if block.children:
        soup = BeautifulSoup(block.html or "", "html.parser")
        child_html = {child.id: _render_block(child) for child in block.children}
        for ref in soup.find_all("content-ref"):
            src_id = ref.get("src")
            rendered = child_html.get(src_id, "")
            replacement = BeautifulSoup(rendered, "html.parser")
            ref.replace_with(replacement)
        html = str(soup)
    else:
        html = block.html or ""

    if block.images:
        image_tags = "".join(
            _image_html(img_id, payload) for img_id, payload in block.images.items()
        )
        if html.strip():
            html += image_tags
        else:
            html = image_tags

    return _convert_math_to_latex(html)


def _convert_math_to_latex(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for math_tag in soup.find_all("math"):
        tex = (math_tag.get_text() or "").strip()
        display = math_tag.get("display", "inline")
        if not tex:
            math_tag.decompose()
            continue

        latex = f"\\[{tex}\\]" if display == "block" else f"\\({tex}\\)"
        span = soup.new_tag("span")
        span["class"] = ["latex-block"] if display == "block" else ["latex-inline"]
        span.string = latex
        math_tag.replace_with(span)

    return str(soup)


def _render_html(blocks: Iterable[JSONBlockOutput]) -> str:
    page_sections = []
    for index, block in enumerate(blocks, start=1):
        body_html = _render_block(block)
        section = (
            f'<section class="page" data-page="{index}">'
            f"<header>Page {index}</header>"
            f"{body_html}"
            "</section>"
        )
        page_sections.append(section)
    return "\n".join(page_sections)


def _wrap_html(body: str) -> str:
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '    <meta charset="UTF-8">',
            "    <title>Marker JSON Render</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }",
            "        .page { background: white; padding: 24px 32px; margin-bottom: 48px; box-shadow: 0 2px 6px rgba(0,0,0,0.12); }",
            "        .page header { font-size: 0.9rem; color: #666; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.08em; }",
            "        figure { margin: 24px auto; text-align: center; }",
            "        figure img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; background: #fafafa; }",
            "        .latex-inline, .latex-block { font-family: 'Fira Code', 'Courier New', monospace; background: #f0f0f0; padding: 2px 4px; border-radius: 3px; }",
            "        .latex-block { display: block; text-align: center; margin: 18px 0; }",
            "    </style>",
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
        ]
    )


def render_json_document(
    json_path: Path, output_path: Path | None = None
) -> Path:
    """Render a Marker JSON export to an HTML file."""
    blocks = _load_blocks(json_path)
    html_body = _render_html(blocks)
    full_html = _wrap_html(html_body)

    target_path = output_path or json_path.with_suffix(".rerendered.html")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(full_html, encoding=settings.OUTPUT_ENCODING)
    return target_path


@click.command()
@click.argument("json_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional path for rendered HTML output.",
)
def render_from_json_cli(json_path: Path, output_path: Path | None):
    """Read a Marker JSON export and render an HTML document."""
    target_path = render_json_document(json_path, output_path)
    click.echo(f"Rendered HTML written to {target_path}")


if __name__ == "__main__":
    render_from_json_cli()
