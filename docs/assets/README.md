# heliaCORE logo & favicon assets

Branding artwork used by the site.

| File                          | Used for                                                   |
|-------------------------------|------------------------------------------------------------|
| `helia-core-icon-white.png`   | Header mark in the MkDocs Material nav bar.                |
| `helia-core-icon-color.svg`   | Browser favicon (via `mkdocs.yml`).                        |
| `helia-core-logo-light.png`   | Wordmark on the landing-page hero, **light** scheme.       |
| `helia-core-logo-dark.png`    | Wordmark on the landing-page hero, **dark** scheme.        |

The light/dark wordmark swap on the landing page is handled by
mkdocs-material's `#only-light` / `#only-dark` URL-fragment convention
(see `docs/index.md`).

## Future additions (optional)

| File              | Recommended format          | Used for                         |
|-------------------|-----------------------------|----------------------------------|
| `favicon.ico`     | ICO, multi-size             | Legacy browser fallback.         |
| `og-card.png`     | PNG, 1200×630               | Open Graph / social preview.     |
