# Revision package verification — 13 October 2026

Status: technically checked author-review package; scientific provenance and author-information release conditions remain in `../README.md`.

## Numerical and document checks

`verify_package.py` passed 611 assertions. The machine-readable record is `package_verification.json`. Checks cover numerical reproduction, input hashes, split assignments, uncertainty decomposition, clean/red text identity, anonymous manuscript content, reviewer-comment coverage, figure/table counts, document XML, and rendered-page text bounds. These checks do not establish source-data validity or journal acceptance.

All four holdout models and all spatial folds were independently reproduced. Repeated-CV summaries were reused and labeled accordingly. Bootstrap, negative-subset, coarse-predictor, classification-break and SHAP analyses are archived under `../analysis/`.

## Rendering and visual review

The supplied LibreOffice-based renderer could not run because LibreOffice is unavailable. Installed Microsoft Word exported the six documents to PDF without saving changes to their DOCX sources. PyMuPDF rendered the PDFs to PNG. Current page counts are: clean manuscript 11; red manuscript 11; reviewer response 7; supplement 7; title page 1; cover letter 1 (38 total).

All current pages were reviewed through contact sheets. Full-resolution inspection additionally covered manuscript text pages, key figures/tables, supplement sensitivity tables, title page and cover letter. No clipping, overlapping content, unintended blank pages or illegible text was observed in this review. Automated checks found text within page bounds. The red manuscript has the same text and tables as the clean copy and uses red font rather than Word tracked changes.

QA PDFs, page PNGs and render reports remain locally under `../qa/` and are excluded from version control and journal attachments. Earlier rendering intermediates have been retained to respect the no-deletion instruction; `qa/render_report.json` identifies current page counts.

## Preservation and release status

The revision was prepared in a separate worktree on `revision/journal-resubmission-20261013`. Existing main-workspace edits and source documents were preserved. No deletion command was used. New journal files are under `../submission/`.

The package is deliberately marked AUTHOR_REVIEW_REQUIRED. Missing categorical/raster provenance, inventory and rainfall metadata, author declarations and revision-deadline status cannot be supplied by numerical checks or editorial rewriting. The original maps and manuscripts remain available; the revised paper explicitly limits its spatial displays to observed locations.
