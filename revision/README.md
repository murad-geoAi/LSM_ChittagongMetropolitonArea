# Journal of Mountain Science revision package

Prepared 13 October 2026 on branch `revision/journal-resubmission-20261013`.

**Status: evidence-corrected revision for author review. Do not upload the package as-is.**
The manuscript, response and supplemental analyses are completed to the extent supported by the supplied records. The audit found material source-data problems that cannot be solved by writing a stronger rebuttal. Required author declarations are also missing. None has been invented.

## Files to use after resolving the release conditions

| File in `submission/` | Intended journal role |
|---|---|
| `Manuscript_Revised_RedMarked.docx` | Revised Main Document, anonymous, changes in red |
| `Manuscript_Revised_Clean.docx` | Clean alternative manuscript, anonymous |
| `Response_to_Reviewers.docx` | Supplementary File for Review / response to reviewers |
| `Supplementary_Methods_Results.docx` | Supplementary File for Review |
| `Figures/` and `figure_manifest.csv` | Separate main and supplementary figure assets |
| `Reproducibility_Supplement.zip` | Analysis scripts, input table and machine-readable results |
| `Title_Page_AUTHOR_COMPLETION_REQUIRED.docx` | Separate title page; complete declarations first |
| `Cover_Letter_AUTHOR_COMPLETION_REQUIRED.docx` | Unsigned cover-letter draft; complete and approve first |

The two manuscript copies contain the same text and results. Red marks indicate additions and substantial rewrites relative to the supplied old manuscript. Black headings remain readable. No Word tracked changes are used. Author names, order and affiliations are preserved on the separate title-page draft, and have not been silently corrected.

## Release conditions that require the authors or original data

1. **Categorical predictor provenance.** The table contains 31 fractional LULC values and 19 fractional soil-texture values. The original code treats all predictors as numeric and does not perform the one-hot encoding claimed in the draft. Supply the category codebook and original rasters/extraction procedure. Re-extract valid classes, choose appropriate categorical and circular-aspect encoding, rerun affected analyses and update the documents if the source is corrected. Arbitrary rounding would invent a correction.
2. **Continuous mapping inputs.** Supply the 14 aligned predictor rasters and study-area polygon/CRS/no-data mask. The legacy code interpolates 369 point predictions to a 500 × 500 geographic grid; this is not wall-to-wall inference from 30 m predictor cells. The revised paper therefore shows corrected uncertainty at observed locations and withdraws operational raster-area/SCAI validation claims. If continuous susceptibility mapping remains the intended submission scope, regenerate it using the source layers and independent validation. The original maps remain intact.
3. **Inventory and negative-sampling records.** Supply event dates, source identifiers, failure type/size/trigger, positional accuracy, deduplication decisions, field/imagery checks, inventory completeness information and the negative-sampling candidate mask/seed if available. The archive supports exact-row deduplication, not the invented 10 m quality-control rule. It contradicts the draft's universal 500 m exclusion and slope-below-10-degree rules (71 nearby negatives; 27 slopes at least 10 degrees). The revised text discloses these issues. Three negative-subset sensitivity runs do not establish validity of the negative labels or replace new candidate-domain sampling.
4. **Rainfall and raster metadata.** Confirm CHIRPS product version and exact aggregation years, source acquisition dates, per-layer resampling methods and the soil-variable definition. These are not recoverable from the numeric samples. The revision does not substitute an invented date range.
5. **Author and administrative information.** Supply manuscript ID, funding/acknowledgements, author contributions, competing interests, ethics/data permissions, all-author approval and exclusive-submission/prior-publication declarations. Confirm the last author's name and malformed email exactly as preserved in the source: “Hideaki Yasuharara” and “yasuhara.hideaki.7pkyoto-u.ac.jp”. Complete the required author information, including ORCID/research interests where requested. The cover letter is not signed and does not assert missing approvals.
6. **Deadline.** The decision letter states 28 August 2026; preparation occurred on 13 October 2026. Confirm an extension or reopening with the journal. No message was sent to the journal and no upload was attempted.

The revised manuscript is candid about these gaps; the authors must decide whether the reduced point-analysis scope is acceptable or provide the sources needed to restore continuous mapping.

## Completed work

- Preserved the original workspace on `main`, including its uncommitted work; created a separate worktree and branch. No file deletion commands were used.
- Located and copied the parent-folder source table: 376 rows, seven exact duplicates, 369 retained locations (155 positive, 214 negative).
- Independently reproduced all four holdout models and every spatial validation fold. MCC, AUC and average precision match the stored holdout results; other checked holdout differences are below 2 × 10^-14. Spatial MCC mean differences are below 10^-15.
- Reconstructed the **actual nine-cluster training validation** and its fold counts, and provided a separately labeled **ten-cluster all-369-point illustration**. The random holdout is explicitly not spatially independent.
- Computed **2,000 paired bootstrap resamples**, ten-bin reliability diagrams with counts, five/ten/fifteen-bin ECE sensitivity and a paired ECE difference interval.
- Ran three repeated negative-training-subset experiments (seeds 7, 21, 84), holding positive training records and the holdout fixed. Reported changed MCC rankings rather than asserting full stability.
- Ran 12 coarse-predictor ablation fits: all four models without rainfall, without soil, and without both.
- Identified that the old uncertainty export equals `1 - 2*abs(p - 0.5)`; replaced its interpretation and computed actual GPC latent moments, probability-scale variance and expected Bernoulli variance.
- Verified the variance decomposition numerically and compared tertiles, Jenks and fixed-thirds display classes.
- Regenerated GPC Kernel SHAP using an explicit 30-row training background, 74 holdout explanations and log-odds output; archived the attributions.
- Rewrote the manuscript and all nine Reviewer 1 and eight Reviewer 2 responses, plus editor and executive-editor responses. Removed unsupported calibration superiority, sampling QC, causal SHAP and area-validation claims.
- Verified selected replacement references against primary sources, removed irrelevant citations and corrected publication-year/validation descriptions.
- Applied A4, 10-point Times New Roman and single-spaced manuscript formatting; main figures and editable tables are placed after the references. Anonymous review files are separated from the title-page draft. New plots use Times New Roman and 600 dpi PNG plus vector PDF.

## Reproduction

From the revision worktree root:

```powershell
python revision/run_revision_analysis.py
python revision/run_revision_checks.py
python revision/style_cached_figures.py
python revision/build_submission.py
python revision/render_with_word.py
python revision/verify_package.py
python revision/package_revision.py
```

`run_revision_analysis.py` reads only the BLR class from the legacy source using Python AST. It never invokes the legacy pipeline's `main()` or file-cleanup functions. All new outputs go under `revision/`. The original repeated-CV summary is reused, not represented as newly recomputed. New sensitivity experiments use the archived predictor encoding to retain a transparent link to the original results.

Core environment: Python 3.12.3, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.8.0, SciPy 1.13.1 and SHAP 0.46.0. See `analysis/environment.json`, `analysis/verification.json`, `requirements-revision.txt`, and the logs for further details. Serialized models are local intermediates, not required journal attachments.

## Document QA

The packaged `render_docx.py` was attempted and diagnosed: this Windows host lacks LibreOffice. The documents were instead exported using installed Microsoft Word in an invisible separate instance, without saving changes to the input DOCX files, and rendered to PNG using PyMuPDF. QA PDFs/PNGs are retained locally, not included as journal deliverables. Visual review and package validation are recorded in `audit/FINAL_QA.md` and `audit/package_verification.json`.

## Journal requirements source

[Journal submission guidelines](https://link.springer.com/journal/11629/submission-guidelines) and the linked September 2024 Instructions for Authors were checked on 13 October 2026. They require an anonymous Main Document, separate Title Page and Cover Letter, and author declarations; the supplied decision letter specifically asks for red revisions and a point-by-point response. Portal-specific choices should follow the current revision invitation.
