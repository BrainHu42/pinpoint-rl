# Codebase layout

`geo_search_env` is organized by responsibility, with dependencies flowing from
experiment orchestration toward stable core abstractions:

```text
geo_search_env/
├── core/        contracts, backend protocol, state machine, geography
├── data/        OSV-5M loading and searchable corpus abstractions
├── models/      frozen matcher and Pinpoint retrieval adapters
├── backends/    live Mapillary, OSV simulator, and fixture implementations
└── experiment/  policies, private scoring, and run orchestration
```

Use imports from `geo_search_env` for the supported convenience API. Import from
a subpackage when ownership matters, such as `geo_search_env.data.OSV5MDataset`
or `geo_search_env.backends.LiveMapillaryTools`.

The small top-level `contracts`, `scoring`, and `runner` modules preserve the
existing contract/scoring imports and the `python -m geo_search_env.runner`
command. Their implementations live in the responsibility-based subpackages.

Dependency rules:

- `core` has no dependencies on the other package layers.
- `data` and `models` build on `core`; models may consume data interfaces.
- `backends` implement the core backend protocol using core, data, and models.
- `experiment` composes all layers but contains no provider-specific logic.
- private labels and scoring remain in `experiment`; backends and policies only
  consume public episode state.
