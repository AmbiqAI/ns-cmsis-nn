# heliaCORE logo & favicon assets

Drop the following files in this directory once branding artwork is
ready. They are referenced by `mkdocs.yml` (`theme.logo` and
`theme.favicon`) and by the landing page hero.

| File              | Used for                              | Recommended format          |
|-------------------|---------------------------------------|-----------------------------|
| `logo.svg`        | Header logo (light + dark schemes).   | SVG, transparent background |
| `logo-mark.svg`   | Small honeycomb-only mark (optional). | SVG, square                 |
| `favicon.png`     | Browser tab favicon.                  | PNG, 32×32 or 64×64         |
| `favicon.ico`     | Legacy browser fallback (optional).   | ICO, multi-size             |
| `og-card.png`     | Open Graph / social preview.          | PNG, 1200×630               |

Until real assets are in place, the site will fall back to the
mkdocs-material default mark — this is intentional so the rest of the
skeleton can be reviewed without blocking on design.
