#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Reject missing local pages, assets and fragment targets in the built site."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a":
            if attrs.get("name"):
                self.ids.add(attrs["name"])
            if attrs.get("href"):
                self.links.append(attrs["href"])


def check(root):
    base = "https://ambiqai.github.io/ns-cmsis-nn/"
    pages = {file: Page(file.read_text()) for file in root.rglob("*.html") if file.is_file()}
    if not pages or not (root / "index.html").exists():
        raise SystemExit("Built site is missing; run npm run build first.")
    failures = []
    links = 0
    fragments = 0
    for file, page in pages.items():
        relative = file.relative_to(root).as_posix()
        route = relative.removesuffix("index.html")
        for href in page.links:
            url = urlsplit(urljoin(base + route, href))
            if url.netloc != "ambiqai.github.io" or not url.path.startswith("/ns-cmsis-nn/"):
                continue
            links += 1
            target = root / unquote(url.path.removeprefix("/ns-cmsis-nn/"))
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                failures.append(f"{relative}: missing target {href}")
            elif url.fragment and target in pages:
                fragments += 1
                if unquote(url.fragment) not in pages[target].ids:
                    failures.append(f"{relative}: missing fragment {href}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Checked {len(pages)} HTML files, {links} internal links, {fragments} fragments: no broken links.")


if __name__ == "__main__":
    check(Path(__file__).resolve().parents[1] / "dist")
