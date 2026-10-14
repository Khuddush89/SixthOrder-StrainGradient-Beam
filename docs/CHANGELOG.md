# Changes in version 2.1.0

- Align the manuscript title and single-author citation metadata.
- Replace the old numerical entry point with stable small-parameter quadrature and overflow-free evaluation.
- Make the core `sgbeam.green.G0` stable too; retain the algebraic coefficient expression separately as `G0_raw`.
- Add regressions for the wrong-sign result at `alpha=0.001` and nonfinite result at `alpha=800`.
- Propagate failed tests and reject nonfinite entry-point inputs.
- Replace the failed stored Maple verification with runnable exact symbolic identities.
- Include all corrected solvers, numerical results, analytical cone bounds, conservative existence annuli and the independent audit.
- Describe the endpoint constraints as factorized constraints and the shear benchmarks by their coefficient choices.
- Supply an original animated six-tile header and six scientific GIFs.
- Add colourful SVG navigation, local badges, emoji summaries and expandable sections.
- Export high-resolution PNGs and vector SVGs without creating manuscript folders.
- Export tables as Markdown, with no LaTeX, PDF or CSV files.
