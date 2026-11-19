# CS230 milestone report

This folder contains the milestone paper source and its compiled PDF.

## Files

- `main.pdf`: compiled milestone report.
- `main.tex`: report source.
- `references.bib`: citations for datasets, prior work, methods, and libraries.
- `neurips_2026.sty`: supplied course style file.
- `figures/composite_losses.pdf`: error-comparison figure used in the paper.
- `report_data_provenance.json`: hashes identifying local artifacts used to
  generate reported results. The underlying data and model artifacts are not
  included in this portfolio repository.

The report documents the four-output design, current two-layer development
results, salary-related future work, and limitations of the previously examined
test split.

## Rebuild

The checked-in PDF is ready to view. Rebuilding requires the local prepared data,
models, and results referenced by `build_report.py`; those generated artifacts
are not included here. With them available, run:

```powershell
python milestone/build_report.py
```

## Overleaf

Upload `main.tex`, `references.bib`, `neurips_2026.sty`, and the `figures/`
directory. Set `main.tex` as the main document and use pdfLaTeX with BibTeX.
