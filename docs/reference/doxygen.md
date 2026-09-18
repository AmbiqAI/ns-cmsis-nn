# API documentation generation

The customer site generates its API reference from C headers using Doxygen and
renders the extracted model through helia-ui in Astro/Starlight. Product guides
and generated API pages share navigation, styling and search.

See [documentation development and publishing](../../astro-site/README.md) for
prerequisites, preview commands, validation and deployment instructions. From
the repository root:

```sh
cd astro-site
npm ci
npm run dev
```

Use `npm run build` for the complete API and site build, followed by
`npm run check`, `npm run test:coverage` and `npm run test:smoke` for validation.
Run `npm run reference` to refresh generated API content after changing headers.

## Sources and outputs

| Item | Location |
| --- | --- |
| Public integer kernels | `Include/arm_nnfunctions.h` |
| Public floating-point kernels | `Include/arm_nnfunctions_flt.h` |
| Shared types and support contracts | `Include/arm_nn_types.h` and support headers under `Include/` |
| Doxygen template | `Documentation/Doxygen/nn.dxy.in` |
| API extraction and grouping configuration | `astro-site/reference.config.json` |
| Site API generation | `astro-site/scripts/build-reference.mjs` |
| Generated API model | `astro-site/public/reference/api/reference.json` |
| Built customer site | `astro-site/dist/` |

Generated API content is rebuilt from the headers rather than edited by hand.
The public coverage check compares extracted names with header declarations;
missing parameter descriptions must be corrected in the source comments.

CMSIS-Pack retains its separate Doxygen generation under
`Documentation/Doxygen/`. Its standalone HTML output is independent of the
Astro site and does not replace the customer documentation deployment.

## Attribution

Inherited CMSIS-NN comments and documentation remain under their original Arm
license terms. Preserve their attribution when rendering or transforming
Doxygen output.
