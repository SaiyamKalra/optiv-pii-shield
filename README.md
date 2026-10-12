# PII Sentinel

PII Sentinel detects and removes personal and confidential identifiers from business documents before the content is shared with an AI tool. You upload a document, the system reads every word including text inside images, finds sensitive values, blacks them out in a copy of the PDF, re-reads that copy to prove nothing sensitive is left, and gives you three outputs: a sanitized PDF, a findings CSV, and an LLM-ready Markdown file.

The project is a Streamlit web app backed by a Python pipeline. It uses Microsoft Presidio for PII detection, spaCy for named entity recognition, Tesseract for OCR, PyMuPDF for PDF reading and redaction, and SQLAlchemy for storing audit metrics.

---

## Table of contents

1. What the project does
2. Repository layout
3. Technology stack
4. End-to-end flow (what happens when you click "Analyze Document")
5. Stage by stage: how each part works
6. The user interface
7. Data model
8. Storage and analytics
9. Verdict logic
10. Configuration and tuning
11. Setup and running
12. Command-line tools
13. Testing
14. Known limitations and bugs
15. Security notes
16. Glossary

---

## 1. What the project does

The core problem is that documents shared with AI tools often carry names, emails, phone numbers, employee IDs, card numbers and similar data, and some of it hides in places a reader never sees: image text, speaker notes, comments, document metadata. PII Sentinel is built to catch these.

For every uploaded file the system produces the following results.

| Output | What it is | Where it comes from |
|---|---|---|
| Sanitized PDF | A copy of the original PDF with each sensitive value covered by a solid black box and the underlying text removed | `redaction/pdf_redactor.py` |
| Findings CSV | One row per detected item: page, source, type, detector, confidence, token, context | Built in `app.py` from the detection results |
| LLM Markdown | Page-by-page Markdown in which every sensitive value is replaced by a stable token such as `[PERSON_001]` | `redaction/token_redaction/token_optimizer.py` |
| Verdict | SAFE, UNKNOWN, or UNSAFE, based on the verification rescan and the confidence of the findings | `app.py` (`verdict()`) |
| Dashboard | Totals, pass rates and charts across every document processed so far | `analytics/metrics.py` and the Analytics screen |

Only PDF files are actually processed end to end. The uploader also accepts DOCX and PPTX, and the extraction and detection layers handle them, but the app refuses them at the redaction step (see Known limitations).

---

## 2. Repository layout

```
PII_Sentinel_Project/
├── app.py                      Streamlit UI: three screens and the orchestration in _run_real_pipeline()
├── config/
│   └── settings.py             Custom recognizers, regex patterns, entity list, ignore list, thresholds
├── extraction/                 Turns files into per-page ExtractionInput chunks
│   ├── validation.py           Checks a file really is a PDF, DOCX or PPTX and is safe to open
│   ├── pdf_extractor.py        PDF: native text, embedded-image OCR, hidden content, unread ink, overlays
│   ├── docx_extractor.py       DOCX: text, tables, text boxes, headers/footers, notes, charts, images
│   ├── pptx_extractor.py       PPTX: one page per slide, shapes, tables, charts, notes, comments, images
│   ├── normalizer.py           Shared helpers: PageTextBuilder, image OCR cache, OOXML metadata, JSON export
│   ├── decorative_images.json  SHA-256 hashes of logo images that are skipped during OCR
│   ├── evaluate.py             Measures extraction quality against ground truth (counts only, no text)
│   └── __main__.py             CLI: python -m extraction FILE
├── ocr/
│   └── ocr_engine.py           Tesseract wrapper: preprocessing, retries, orientation, ink detection
├── pii/                        PII detection (the live detection path)
│   ├── pipeline.py             PIIPipeline: Presidio analysis, filtering, tokenization, verification, DB save
│   ├── token_manager.py        Stable [TYPE_NNN] tokens with fuzzy person-name matching
│   ├── verifier.py             VerificationGate: re-analyses sanitized text, expects zero high-risk hits
│   ├── regressor_calibrator.py PIIRiskCalibrator: adjusts raw scores with context rules (or a trained model)
│   ├── gliner_recognizer.py    GLiNER zero-shot recognizer: written but NOT wired in (see Known limitations)
│   └── pdf_redactor.py         Empty placeholder file
├── redaction/                  Produces and verifies the sanitized PDF (the live redaction path)
│   ├── pipeline.py             RedactionPipeline: coverage checks, mapping, redaction, rescan, final status
│   ├── pdf_redactor.py         PDFRedactor: draws redaction boxes and saves the PDF
│   ├── verifier.py             RedactionVerifier: passes only if every page rescans clean
│   ├── token_manager.py        Empty placeholder file
│   └── token_redaction/
│       └── token_optimizer.py  TokenOptimizer: Markdown export, boilerplate removal, cleanup
├── models/
│   └── schemas.py              Pydantic models shared across all layers
├── database/
│   ├── db.py                   SQLAlchemy engine, session factory, two tables
│   ├── repository.py           Writes one summary row per page and one row per detection
│   └── schema.sql              Empty placeholder
├── analytics/
│   └── metrics.py              Queries the database for the dashboard
├── utils/
│   └── metrics.py              Precision and word-coverage helpers (not used by the app)
├── tests/                      19 test modules, 78 test functions, plus a ground-truth fixture
├── docs/                       architecture.md, development.md, data-contract.md (all currently empty)
├── .streamlit/config.toml      Dark theme (background #0d0d14, accent #4ade80)
├── requirements.txt            Python dependencies
├── .env                        Environment variables (contains DATABASE_URL; must not be shared)
├── optiv_shield.db             Bundled SQLite database with earlier run records
└── README.md                   This file
```

Two folders have near-identical names, `pii/` and `redaction/`. Both contain a `pipeline.py`, a `verifier.py` and a `pdf_redactor.py`, but they are different files with different jobs. `pii/` does detection and `redaction/` does the PDF work. The `pii/pipeline.py` file imports the token optimizer from `redaction/`, and `redaction/pipeline.py` imports `PIIPipeline` from `pii/`.

---

## 3. Technology stack

| Layer | Library or tool | Version or note |
|---|---|---|
| Web UI | Streamlit | unpinned |
| Charts | Plotly Express | unpinned |
| PII engine | Presidio Analyzer and Anonymizer | 2.2.35 (pinned) |
| NLP | spaCy, model `en_core_web_lg` | spaCy 3.8 or later; the model must be downloaded separately |
| PDF read and write | PyMuPDF (`pymupdf`, imported as `fitz` and `pymupdf`) | 1.24 or later |
| OCR | Tesseract binary, via `pytesseract` | Binary installed separately |
| Image handling | Pillow, NumPy | Pillow 10 or later, NumPy 1.26 or later |
| Word files | python-docx, lxml (dependency) | 1.1 or later |
| PowerPoint files | python-pptx | 1.0 or later |
| Data models | Pydantic | 2.9 or later |
| Database | SQLAlchemy, psycopg2-binary | SQLite by default; PostgreSQL via `DATABASE_URL` |
| Data handling | pandas | unpinned |
| Model loading | joblib | optional trained calibrator |
| Config | python-dotenv | reads `.env` |
| Testing | pytest; ruff for linting | |
| Declared but unused | torchvision, torch (used by `pii/pipeline.py`) | `torch` is imported but not listed; it arrives with torchvision |

---

## 4. End-to-end flow

This is what happens from the moment a PDF is uploaded.

1. **Upload.** The user drops a PDF into the Upload screen and clicks Analyze Document. The button is disabled until a file is chosen.
2. **Temporary copy.** `_run_real_pipeline()` writes the upload to a temporary folder as `original.pdf`. Every later step works on this copy, and the folder is deleted when the step finishes.
3. **Validation.** `extract_pdf()` calls `validate_document()`, which checks the file's content, not its extension (see section 5.1).
4. **Extraction.** The file is read page by page. Each page becomes an `ExtractionInput` holding plain text, word boxes with coordinates, OCR results for images, hidden content, and unread regions (section 5.2).
5. **Detection (first pass).** `PIIPipeline()` starts up. It loads the spaCy model, builds the Presidio analyzer with custom recognizers, and creates the verification gate. `process_document_chunks()` then analyses every page's text, filters the candidates, resolves overlaps, assigns tokens, builds the sanitized text, and verifies it (section 5.3).
6. **Mapping to the PDF.** `RedactionPipeline.process()` checks that the extraction covers every page of the PDF. It then links each finding to the word boxes it overlaps, using character offsets (section 5.4). If any finding cannot be linked to a box, the run is blocked.
7. **Redaction.** `PDFRedactor.redact()` adds a solid black redaction annotation over each box, applies the redactions page by page, and saves the result as `sanitized.pdf`.
8. **Rescan.** The sanitized PDF is extracted again from scratch, and the full detection pipeline runs on its text. This is the proof step. If OCR is weak or the rescan finds anything, the output is not verified.
9. **Final status.** `RedactionVerifier` requires every page to be rescanned, each page to be marked SUCCESS with verification passed, and zero detected entities on every page. The result is one of three statuses (section 5.4).
10. **Results.** Back in `app.py`, the findings are turned into a DataFrame, the word counts and KPIs are computed, and the session state is updated. The UI moves to the Findings screen.
11. **Export.** On the Findings screen the user can download the findings CSV, the sanitized PDF, and the LLM Markdown. The Markdown is built by `TokenOptimizer` from the sanitized page text.
12. **Dashboard.** Every page and every detection, including the rescan's, is written to the database, and the Analytics screen reads from there.

Timing is printed in the UI. A short document takes a few seconds on a modern laptop. Large scanned documents take much longer because of OCR.

---

## 5. Stage by stage

### 5.1 Validation (`extraction/validation.py`)

The file type is decided from the file's bytes:

| Signature | Result |
|---|---|
| Starts with `%PDF-` | PDF |
| OLE header (`D0 CF 11 E0 …`) | Rejected as password-protected Office or legacy `.doc`/`.ppt` |
| ZIP header with `word/document.xml` inside | DOCX |
| ZIP header with `ppt/presentation.xml` inside | PPTX |
| Anything else | Rejected as unsupported |

Other checks are made before extraction starts. The file must not be empty, and it must be under 200 MB. For ZIP-based Office files the code also checks the total uncompressed size (under 1 GB) and the compression ratio (under 200:1), to guard against zip bombs, and it runs `testzip()` for damaged parts. For PDFs it rejects anything that needs a password or has no pages, and it logs a warning if PyMuPDF had to repair the file. If the extension does not match the detected type, a warning is logged. The app itself only accepts PDF in the redaction step.

Each rejection has a machine-readable reason: `not_found`, `empty`, `too_large`, `unsupported_type`, `encrypted`, `encrypted_or_legacy`, `corrupt` or `no_pages`.

### 5.2 Extraction

The extractor's job is to produce, for each page, the visible text in reading order and a box for every word, along with everything else that could carry personal data.

**PDF (`extraction/pdf_extractor.py`).** Each page is handled in four parts.

*Native text.* PyMuPDF returns words with coordinates. They are grouped into lines and blocks in PyMuPDF's reading order. Each word is recorded as a box in points with a top-left origin, and its character offsets in the page text are stored too, so the rest of the system can link a detected value back to a box.

*Embedded images.* Each image placement on the page is rendered at 300 DPI, and images smaller than 40 points on either side are skipped as logos or icons. Images listed in `decorative_images.json` by SHA-256 hash are skipped and counted separately. Each remaining image goes to Tesseract (section 5.2.1). OCR words that duplicate native text already on the page are removed when more than half of their area overlaps a matching native word. That prevents double-counting when a PDF has both a text layer and a scanned copy of the same line.

*Hidden content.* Annotation text and comments, form field values, link URLs, document metadata (title, author, subject, keywords, creator, producer), XMP metadata, embedded files, and bookmark titles are all recorded as hidden content. Document-level items are attached to page 1. This text is stored with the page but is not part of the page's visible text.

*Unread ink.* After OCR, the engine looks for dark marks no word covers. A large leftover area, such as a barcode, is recorded as an unread region in orange and treated as possibly containing personal data.

If a page throws an error, it is reported as not inspected rather than as empty, so a broken page cannot pass silently.

Pages are processed in order while their images are OCR'd on several threads, and only a few pages are held in memory at a time.

**DOCX (`extraction/docx_extractor.py`).** Word files have no fixed pages, so the code splits pages on the places Word itself recorded: explicit page breaks, the `lastRenderedPageBreak` markers Word writes when it saves, and paragraphs flagged "page break before". Page numbers are therefore an approximation of what Word showed when the file was last saved. The walker reads paragraphs, tables (cells joined with `|`), text boxes, headers and footers (added once at the start of page 1), chart labels and SmartArt text. Comments, footnotes and endnotes go into hidden content, as do tracked deletions, alt text, document properties, custom properties, embedded objects, and orphaned parts. Images are OCR'd, and their word boxes are in image pixels because a DOCX image has no fixed page position.

**PPTX (`extraction/pptx_extractor.py`).** One page per slide, with shapes ordered by top and then left position. Text frames, tables (rows as `|`-separated cells), groups, charts, SmartArt, OLE objects, speaker notes and slide comments are read. Pictures are OCR'd and their boxes are converted to slide points, accounting for cropping. Hidden slides are flagged in hidden content.

#### 5.2.1 OCR engine (`ocr/ocr_engine.py`)

Tesseract is called through `pytesseract`. For each image the engine does the following.

1. Converts to grayscale, inverts it if the image is dark (mean below 110), and stretches the contrast.
2. Upscales the image 2× if it is shorter than 1000 pixels.
3. Runs Tesseract with page segmentation mode 3 and a character blacklist of `|`, because Tesseract confuses a lowercase `l` with `|` and that breaks email addresses.
4. If the mean confidence is below 60, retries at 2× scale and keeps the better result.
5. Detects whether the text is sideways or upside down by trying 90, 180 and 270 degrees, and keeps the rotation that reads best.
6. Re-reads low-confidence lines with a tighter crop at 3× scale, up to 30 lines per image.
7. Re-reads uncovered ink at 2× scale, up to 15 regions per image.

Words are kept only if they contain at least one letter or digit. Each Tesseract call has a 180-second timeout. The number of images OCR'd at once is 75% of available CPUs, capped at 16, or set with `OCR_WORKERS`. If Tesseract is not installed, images are counted as skipped and the run continues with a warning.

#### 5.2.2 Output: the `ExtractionInput` chunk

Each page becomes one `ExtractionInput` with these fields: `file_id` (a SHA-256 prefix of the file's bytes), `file_name`, `page_number`, `raw_text`, `bounding_boxes`, `ocr_used`, `extraction_method` (`native`, `ocr` or `hybrid`), `ocr_mean_confidence`, `low_confidence_word_count`, image counts (OCR'd, skipped, decorative), `hidden_content`, `unread_regions`, and `extraction_errors`.

### 5.3 PII detection (`pii/pipeline.py`)

This is where sensitive values are found.

**The analyzer.** `PIIPipeline.__init__` builds a Presidio `AnalyzerEngine` on top of spaCy `en_core_web_lg`. The built-in recognizers are loaded, and the built-in phone recognizer is removed and replaced with the project's own. The project then adds these recognizers from `config/settings.py`:

| Recognizer | Entity | What it matches | Score |
|---|---|---|---|
| CorpEmailRecognizer | EMAIL_ADDRESS | Standard email pattern | 1.0 |
| RobustPhoneRecognizer | PHONE_NUMBER | Optional `+` country code, bracketed or plain area code, groups separated by spaces, dots, hyphens or newlines, optional extension. Built to cope with phone numbers split across table lines | 0.85 |
| StrictSsnRecognizer | US_SSN | `NNN-NN-NNNN` | 1.0 |
| CorpCreditCardRecognizer | CREDIT_CARD | 16 digits, grouped or not | 1.0 |
| EmployeeIdRecognizer | EMPLOYEE_ID | `EMP-` followed by 4 to 8 digits | 0.99 |
| DirectorIdRecognizer | DIRECTOR_ID | `DIR-` followed by 4 to 8 digits | 0.99 |
| ExceptionIdRecognizer | EXCEPTION_ID | `EXC-` followed by 3 to 8 digits | 0.99 |
| DocumentReferenceRecognizer | DOCUMENT_REFERENCE | `GRP-POL-` followed by 2 to 8 digits | 0.99 |
| ContextualNameRecognizer | PERSON | Two or three capitalised words, or an initial plus a surname, scored higher when a role word such as "owner", "director" or "approved by" appears nearby | 0.70 to 0.75 |

SpaCy also contributes PERSON, LOCATION and ORGANIZATION. SpaCy labels that are not useful for this task (cardinal numbers, money, percentages, laws, products, events) are switched off.

**Table harvesting.** When a page's text is a table cell or the column has a header, `PIIRiskCalibrator.harvest_table_cell_candidates()` scans the text for capitalised name pairs and for phone-like digit runs, which are often missed by the standard recognizers in tables. Names score 0.55 in a person-type column and 0.45 elsewhere. Phones score 0.65.

**Calibration.** Each candidate's base score is adjusted by `PIIRiskCalibrator`. If a trained model file `models/pii_regressor.joblib` exists, the calibrator uses it. That file is not in the project, so the rule-based adjustment is what actually runs. The rules are as follows: a title-case name with person-type context in the surrounding 40 characters, or inside a table, gets +0.35. An international phone prefix, or an ID context with more than 35% digits, gets +0.25. A purely numeric span with no context at all and more than 80% digits gets −0.20.

**Filtering.** A candidate survives only if it meets its minimum score (0.60 for spaCy NER, 0.85 for pattern and rule recognizers) and passes these checks.

- It is not a regulatory acronym or standard reference (ISO, NIST, GDPR, Basel, SLA, KPI and similar), and it does not look like a standard number such as `27001:2013`.
- A PERSON that is entirely lowercase, or contains digits or symbols such as `:`, `/` or `_`, is discarded.
- An ORGANIZATION whose text contains words such as ACT, POLICY, FRAMEWORK, MODEL, STANDARD, GUIDELINES or RESILIENCE is discarded.
- Its text is not in the ignore list in `settings.py` (for example RACI, SWOT, KPI, GDPR, DORA, "Compliance Sign-off").

**Overlap resolution.** When spans overlap, the one with the highest priority wins: IDs first, then credit card, SSN, IP address and email, then phone, then person, then organisation and location. Ties go to the higher score and then the longer span.

**Tokenization.** `TokenManager` gives each distinct value a token of the form `[TYPE_NNN]`, numbered per type across the whole document. The same value always gets the same token. For names, the key is the first initial plus the last name, so "Marguerite Vossberg" and "M. Vossberg" produce the same key and share one token. Matching depends only on the initial and the surname, so two different people with the same initial and surname would be merged.

**Building the sanitized text.** Detected spans are replaced with their tokens, starting from the end of the text so earlier offsets stay correct.

**Verification.** `VerificationGate` re-runs the analyzer on the sanitized text, looking only for high-risk entity types (email, phone, SSN, card, the four ID types, IP address) at a score of 0.70 or more. The page passes only if none are found. The page status is SUCCESS when it passes, and FLAGGED_FOR_REVIEW when it does not.

**Persistence.** Each page's result is saved to the database as one `document_summary` row and one `pii_detection_logs` row per detection. Only tokens and scores are stored, never original values.

Empty pages are handled as a special case. They are saved as FLAGGED_FOR_REVIEW with an empty sanitized text.

### 5.4 Redaction and verification (`redaction/`)

**Coverage check.** `RedactionPipeline.process()` first confirms the chunks cover every page of the PDF, with no page missing and none repeated. If not, the run is BLOCKED.

**Inspection.** Each page is checked for OCR without confidence data, low-confidence OCR words, and skipped images. OCR problems are recorded as inspection issues and skipped images as warnings. These only affect the final status; they do not stop redaction.

**Mapping.** For each finding, `PDFRedactor._boxes_for_finding()` returns every word box whose character range overlaps the finding's range. If a finding has no box, it is listed as unmapped and the run is BLOCKED. This is a deliberate fail-safe: a value that cannot be placed on the page cannot be blacked out.

**Redaction.** For each box, a redaction annotation with black fill is added and the page is redacted. The file is saved with garbage collection level 3 and compression on. Every finding is redacted, regardless of its confidence score. The confidence score only controls the review flag.

**Rescan.** The saved file is extracted again from scratch with `extract_pdf()`, and the full PII pipeline runs on the result. The coverage check is repeated on the new extraction. Any OCR issues found on the rescan are also recorded as inspection issues.

**Final status.** `RedactionVerifier` requires that every page appears exactly once, that every page's status is SUCCESS, that every page passed verification, and that no page has any detected entity. The status is then set as follows.

| Status | Condition |
|---|---|
| `BLOCKED` | Coverage failed, a finding is unmapped, the rescan failed, any inspection issue exists, or verification failed |
| `REVIEW_REQUIRED` | Verification passed but some images were skipped (warnings only) |
| `VERIFIED_NO_PII_FOUND` | Verification passed and there are no issues or warnings |

**What "verified" means.** A verified status means the same detectors, run on the output, found nothing above threshold. It does not prove that nothing sensitive remains. Anything the detectors cannot see, such as a value they miss, a value outside the text layer, or hidden metadata, is not covered. Section 14 lists these gaps.

### 5.5 Markdown export (`redaction/token_redaction/token_optimizer.py`)

`TokenOptimizer` converts the sanitized page text into Markdown for an LLM. The output has a YAML-style front matter block with the file name, file ID, page count and a status line. Each page is introduced with an HTML comment, `<!-- Page N -->`, and pages are separated by horizontal rules.

Before writing, each page's text goes through two cleanups. Boilerplate phrases are removed: "Page X of Y", "Confidential - Internal Use Only", and "All Rights Reserved." Then the text is tidied: spacing around tokens is normalised, runs of spaces and tabs are collapsed, and lines are merged into paragraphs at blank lines. Known issues with this step are listed in section 14.

The class also has two methods that are not used by the app: `redact_pdf_artifacts()` and `extract_sanitized_markdown()`. They belong to an earlier design that did its own detection and did not use the main pipeline.

---

## 6. The user interface (`app.py`)

The app has a single page with three screens. Navigation is three buttons at the top, and the Findings and Analytics buttons are disabled until a document has been processed.

**Upload screen.** A file picker accepts PDF, PPTX and DOCX. The Analyze button runs the pipeline inside a status box. The step ticks shown in the box are cosmetic: they are displayed on a timer and do not reflect real progress. A static sample preview on the right shows how "Rahul Sharma" and an email address would be tokenized. The info panel describes the six pipeline steps.

**Findings screen.** The screen shows the following, in order.

1. A conversion summary: total words processed, PII identified, and percentage of data converted (characters in detected values divided by total characters).
2. A status banner with the file name and processing time.
3. A row of KPI cards: total pages, OCR pages, PII findings, coverage, and processing time.
4. Three filters: PII type, source, and page range.
5. A findings table with a confidence bar for each row. A row is labelled "Redacted" when its confidence is at or above the threshold, and "Needs review" when it is below. As noted in section 5.4, every finding is redacted either way, so this label is a review flag and not a redaction decision.
6. Download buttons for the filtered findings CSV, the sanitized PDF, and the LLM Markdown.

The threshold is 0.70 and is stored in session state. There is no control in the UI to change it.

**Analytics screen.** This screen reads the database, not the current upload. It shows four KPI cards (total documents, verification pass rate, total PII masked, average confidence) and four charts: entity distribution (pie), findings by source (pie), average confidence by entity type (bar), and recent pass rates (bar). If a database query fails, that KPI falls back to zero. The charts fall back to the current session's findings when the database has no data.

**Styling.** The dark theme uses cyan headings, green accents and a monospace font for tokens. Layouts are responsive and charts are styled with the Plotly dark template.

---

## 7. Data model (`models/schemas.py`)

| Model | Purpose |
|---|---|
| `BoundingBox` | One word: page, text, position and size, character offsets, source (`native` or `ocr`), confidence, coordinate space (`page_pt` or `image_px`), and image reference |
| `HiddenContent` | Text stored in the file but not visible: kind (metadata, comment, alt_text, annotation, etc.), text, and location |
| `UnreadRegion` | An area of ink that OCR could not read, with a reason code |
| `ExtractionInput` | One page as extracted, with all of the fields listed in section 5.2.2 |
| `DetectedEntityDetail` | One finding: token, entity type, original value, start and end, score, page |
| `PIIOutput` | One page's detection result: status, sanitized text, token-to-original mapping (in memory only), detected entities, verification result, and counts |

The mapping from token to original value lives only in memory on `PIIOutput.mapping_ledger`. It is not written to the database.

---

## 8. Storage and analytics

**Database (`database/db.py`).** SQLAlchemy creates two tables.

| Table | One row per | Columns |
|---|---|---|
| `document_summary` | Page processed | `file_id`, `file_name`, `page_number`, `status`, `verification_passed`, `total_entities_detected`, `processed_at` |
| `pii_detection_logs` | Detection | `file_id`, `token`, `entity_type`, `confidence_score`, `page_number`, `processed_at` |

The database URL comes from the `DATABASE_URL` environment variable and defaults to `sqlite:///optiv_shield.db` in the project folder. The `init_db()` call creates the tables when they do not exist. There is no migration system, and `schema.sql` is empty.

**Analytics (`analytics/metrics.py`).** Each query runs through pandas. `fetch_executive_kpis()` returns total documents, pass rate and total PII masked. `fetch_entity_distribution()` counts detections per type. `fetch_confidence_score()` averages confidence per type. `fetch_recent_pass_rates()` returns the last 10 pages' pass results. `fetch_risk_exposure_matrix()` pivots detections by file and type, though no screen uses it. `get_conversion_matrix()` builds the conversion summary on the Findings screen.

The counts in the dashboard do not reflect documents as the label suggests. Section 14 explains why.

**Bundled database.** `optiv_shield.db` contains 2,189 page records from 176 distinct file IDs and 60,059 detection rows. The file names suggest it was produced during development and testing, using sample and case-study documents.

---

## 9. Verdict logic (`verdict()` in `app.py`)

The verdict is decided in this order, and the first match wins.

| Order | Condition | Verdict |
|---|---|---|
| 1 | Backend status is `BLOCKED` | UNSAFE |
| 2 | Backend status is anything other than `VERIFIED_NO_PII_FOUND` | UNKNOWN |
| 3 | Residual count is above zero | UNSAFE |
| 4 | Any finding has confidence below 0.70, or coverage is below 95% | UNKNOWN |
| 5 | None of the above | SAFE |

Coverage is computed as reliable words over total words. In the current code the two counts are always set to the same value, so coverage is always 100% (see section 14).

---

## 10. Configuration and tuning

Every tunable value is in one of a few places.

| Setting | Where | Default | Effect |
|---|---|---|---|
| Recognizer patterns | `config/settings.py` | see section 5.3 | What counts as an email, phone, SSN, card, ID or name |
| Target entities | `TARGET_ENTITIES` in `settings.py` | 12 types | What the analyzer looks for |
| High-risk entities | `HIGH_RISK_VERIFICATION_ENTITIES` | 9 types | What the verification rescan fails on |
| Ignore list | `IGNORED_WORDS` | about 40 terms | Standards and acronyms never redacted |
| NER minimum score | `MIN_NER_SCORE` | 0.60 | Threshold for spaCy-based findings |
| Pattern minimum score | `MIN_PATTERN_SCORE` | 0.85 | Threshold for pattern-based findings |
| Review threshold | `thr` in session state | 0.70 | Findings below this are flagged "Needs review" and make the verdict UNKNOWN |
| Coverage threshold | `COVERAGE_REQUIRED_PCT` in `app.py` | 95% | Minimum share of words that must be reliably read |
| Minimum image size | `MIN_IMAGE_SIDE_PT` | 40 pt | Smaller images are not OCR'd |
| Decorative logos | `extraction/decorative_images.json` | one logo (Cadence) | Images with these SHA-256 hashes are skipped |
| OCR language | `OcrEngine(lang="eng")` | English | Tesseract language |
| OCR workers | `OCR_WORKERS` environment variable | 75% of CPUs, max 16 | Parallel OCR jobs |
| Tesseract path | `TESSERACT_CMD` environment variable | searches PATH, then the default Windows install | Location of the binary |
| Database | `DATABASE_URL` environment variable | `sqlite:///optiv_shield.db` | Where metrics are stored |
| Torch threads | `pii/pipeline.py` | 4 | CPU threads for PyTorch |

To add a logo to the decorative list, run `python -m extraction FILE --list-images`, check that the image holds no personal data, and add its hash to `decorative_images.json`.

---

## 11. Setup and running

**Prerequisites.** Python 3.11 or later (the code imports `datetime.UTC`), Tesseract OCR installed on the system (the binary, not just the Python wrapper), and several gigabytes of disk space for the Python dependencies and the spaCy model.

**Steps.**

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download the spaCy model (not in requirements.txt)
python -m spacy download en_core_web_lg

# 4. Install Tesseract
#    Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
#    macOS:   brew install tesseract
#    Linux:   sudo apt install tesseract-ocr

# 5. (Optional) create a .env file in the project root
#    DATABASE_URL=sqlite:///optiv_shield.db
#    TESSERACT_CMD=/path/to/tesseract        (only if it is not on PATH)
#    OCR_WORKERS=4                           (optional)

# 6. Run the app from the project root
streamlit run app.py
```

Run the app from the project root. The calibrator looks for `models/pii_regressor.joblib` using a path relative to the working folder, and the test and evaluation tools also depend on the current folder.

---

## 12. Command-line tools

| Command | What it does |
|---|---|
| `python -m extraction FILE` | Prints a per-page table: method, characters, boxes, mean confidence, low-confidence count, OCR and skipped images, decorative images, unread regions, hidden items, errors |
| `python -m extraction FILE --show-text` | Also prints each page's extracted text (contains PII) |
| `python -m extraction FILE --show-hidden` | Also prints hidden content and unread regions (contains PII) |
| `python -m extraction FILE --save-chunks out.json` | Saves all pages as JSON for the detection CLI (contains PII) |
| `python -m extraction FILE --list-images` | Lists each embedded image with its SHA-256 hash, for the decorative list |
| `python -m extraction.pdf_extractor FILE --overlay DIR` | Saves a PNG per page with every extracted word box drawn on it. Blue is native text, green is OCR, red is low-confidence OCR, orange is unread ink |
| `python -m extraction.evaluate [DIR]` | Scores extraction against `tests/fixtures/cadence_ground_truth.json`: identifier recall, row integrity, OCR counts. Needs the case-study files, which are not in the zip |
| `python -m pii.pipeline out.json` | Runs detection on saved chunks and prints a summary. Add `--show-sanitized`, `--json FILE`, or `--export-md FILE` |

---

## 13. Testing

The suite has 19 test modules (plus a fixtures folder) and 78 test functions.

| Area | Files | What is checked |
|---|---|---|
| Extraction | `test_extraction.py`, `test_extraction_hardening.py`, `test_office_extraction.py` | Page order, character offsets, blank pages, tiny images, OCR box placement, rotation, hidden content in all three formats, validation rejects, decorative logos, OCR failure reporting |
| OCR | `test_ocr.py` | Missing binary, preprocessing, light-on-dark text, worker count, crop re-reads |
| Detection | `test_pii.py` | A stress test with dense, deliberately tricky text across three pages: every email must be masked and every page must pass verification |
| Redaction | `test_redaction.py`, `test_redaction_e2e.py`, `test_redaction_multipage.py`, `test_redaction_scanned_pdf.py`, `test_redaction_rescan.py`, `test_redaction_page_coverage.py`, `test_redaction_unmapped.py`, `test_redaction_verifier.py`, `test_redaction_pipeline.py`, `test_redaction_quality.py` | Redaction boxes, multi-page output, scanned-PDF redaction, rescan removes original text, blocking on coverage gaps and unmapped findings, verifier edge cases, quality gates |
| Token optimizer | `test_token_optimizer.py` | Markdown generation across pages |
| Analytics | `test_analytics.py` | Each analytics query against a test database |
| Real documents | `test_real_documents.py` | Regression against the case-study files. Skipped unless `RUN_REAL_DOCS=1` is set |
| Database | `test_database.py` | Empty file, so no database tests run |

Run everything except the slow real-document checks:

```bash
pytest -q
```

Run the real-document regression (needs the case-study files, which are not in the zip):

```bash
RUN_REAL_DOCS=1 pytest tests/test_real_documents.py -v
```

Several tests need the `en_core_web_lg` model and skip themselves if it is missing. Tests that need Tesseract also depend on it being installed. The `.pytest_cache` folder shows an empty list of last failures, but I did not run the suite here because its dependencies and the Tesseract binary are not available in this environment, so treat the test results as unverified.

---

## 14. Known limitations and bugs

These were found by reading the code and, where stated, by checking the bundled database. Each one is worth fixing before the tool is relied on.

**Privacy gaps**

1. **Hidden content is never scanned.** Document metadata, comments, annotations, form values, alt text, tracked deletions, embedded files and orphaned parts are all extracted and stored in `hidden_content`. Nothing in the pipeline runs PII detection on that text, and nothing removes it. The schema's own docstring says downstream PII scanning must check it, and no downstream code does.
2. **The sanitized PDF keeps its metadata.** The redactor saves the PDF with `garbage=3` and compression, but it does not clear the document information fields, XMP metadata, annotations, or embedded files. An author name or a comment in the original is still present in the "sanitized" output.
3. **The verification is only as good as the detectors.** The rescan uses the same detectors as the first pass, so it confirms only that they find nothing more. A value they miss in the first pass will also be missed in the rescan. The phrase "0% residual PII leakage" in the UI overstates what is checked.
4. **The sanitized PDF can be downloaded when the verdict is UNSAFE or UNKNOWN.** The download button is enabled whenever the output file has bytes (`disabled=not has_bytes`). The variable `verified` is computed and never used, so the verdict does not gate the download.
5. **Every finding is redacted, but the UI implies otherwise.** The "Needs review" label is a confidence flag. Redaction itself does not depend on it.

**Functional bugs**

6. **Coverage is always 100%.** In `_run_real_pipeline()`, `reliable_words` is set to `total_words`, so the coverage check in `verdict()` can never trigger. The verdict's coverage rule is therefore inactive.
7. **DOCX and PPTX uploads fail at the end.** The uploader accepts them, and the extraction layer handles them, but `_run_real_pipeline()` raises "Verified redaction currently supports PDF files only." The Upload screen's text says all three are supported.
8. **Each upload is recorded twice.** The rescan calls `process_document_chunks()` again, and that saves to the database a second time. The bundled database contains both kinds of rows: records saved under the first pass's temporary name `original.pdf` and records saved under the rescan's name `sanitized.pdf` (725 rows across 46 file IDs). One upload therefore produces two sets of rows, and the totals are inflated.
9. **The real file name is lost in the database.** The app writes the upload to `original.pdf` in a temporary folder, so every record says `original.pdf` (1,115 rows across only 4 distinct file IDs in the bundled database), not the uploaded file's name.
10. **"Total Documents" counts pages.** `fetch_executive_kpis()` takes `len(document_summary)`, which is one row per page. The bundled database shows 2,189 "documents" from 176 real file IDs.
11. **The Markdown header always says VERIFIED.** `optimize_output()` hard-codes `status: VERIFIED_PII_REDACTED` whatever the verdict was.
12. **The Markdown cleanup removes real hyphens.** `_clean_text()` uses `(\w+)-\s*(\w+)`, which joins any two words separated by a hyphen. Checked directly: "Dubois-Kane" becomes "DuboisKane" and "risk-based" becomes "riskbased". The intent was only to rejoin words split across line breaks.
13. **The `--export-md` option of the detection CLI crashes.** `pii/pipeline.py` reads `metrics['tokens_saved']` and `metrics['savings_percentage']`, but `calculate_metrics()` returns `words_saved` and `savings_pct`. This raises a `KeyError`.
14. **The Source filter lists values that are never produced.** "table" and "narrative" appear in the filter, but the code only ever assigns "body" or "image/OCR".
15. **The progress steps are cosmetic.** The steps in the status box are written on a fixed timer, not as work completes.
16. **The database URL in `.env` is not loaded in time.** `database/db.py` reads `DATABASE_URL` when it is imported. `app.py` imports the analytics module, which imports the database module, before anything calls `load_dotenv()`. The `.env` value (a PostgreSQL URL) is therefore ignored unless `DATABASE_URL` is also set in the shell. Otherwise the app uses the bundled SQLite file, which is the mismatch the bundled data reflects.

**Detection gaps**

17. **The trained calibrator is missing.** `pii/pipeline.py` loads `models/pii_regressor.joblib`, which is not in the project. The rule-based adjustment runs instead, silently.
18. **The spaCy model is not in the requirements.** `en_core_web_lg` must be downloaded separately. Without it the analyzer fails to start.
19. **The GLiNER recognizer is written but not used.** `pii/gliner_recognizer.py` is never imported, and `gliner` is not in the requirements. It is a zero-shot alternative to spaCy that was never connected.
20. **Organisations and locations are redacted.** `ORGANIZATION` and `LOCATION` are in the target list, and spaCy's guesses at these are noisy. The ignore list and filters catch some of them, but not all.
21. **The name rules depend on capitalisation.** Names are found by title-case patterns and context words. Lowercase names, names in all caps, and names without a role word nearby can be missed.
22. **The phone pattern is broad.** Long digit runs with separators can match non-phone numbers, such as reference numbers, which is why the calibrator penalises purely numeric spans.
23. **The UI claims more than the calibrator does.** The info panel says the calibrator boosts "non-Western names". It boosts capitalised names near context words, which is not the same thing.
24. **Unused recognizers are registered.** `TokenOptimizer` adds Indian PAN, US SSN and Polish PESEL recognizers. The PESEL pattern (any 11-digit number) is likely to create false positives, but it only affects the unused methods.

**Dead and duplicate code**

25. `_demo_pipeline()`, `ENTITY_TYPES`, `WEIGHTS` and `SOURCES` in `app.py` produce fake data for a preview that no screen calls. `_demo_pipeline()` is never referenced.
26. `pii/pdf_redactor.py` and `redaction/token_manager.py` are empty.
27. `database/schema.sql`, `docs/*.md`, `README.md` and `tests/test_database.py` were empty in the zip.
28. `utils/metrics.py` is not used by the app.
29. `TokenOptimizer.redact_pdf_artifacts()` and `extract_sanitized_markdown()` are unused.
30. The duplicated `pii/` and `redaction/` module names make the code harder to follow.

**Operational**

31. **The analyzers are rebuilt on every run.** `PIIPipeline()` and `TokenOptimizer()` each load spaCy from scratch. `TokenOptimizer()` is constructed inside the Findings screen, so it reloads each time Streamlit reruns the page.
32. **Relative paths.** The calibrator model path and the evaluation paths are relative to the working folder.
33. **Hard-coded thread count.** `torch.set_num_threads(4)` is set regardless of the machine.
34. **Blocked output is kept on disk.** The code that removes a blocked output file is commented out. The file sits in a temporary folder, so this is mostly harmless, but it is worth knowing.
35. **Commented-out blocking.** A block on inspection issues in the first phase of `RedactionPipeline.process()` is commented out. Issues still block at the end, so the outcome is correct, but the code is confusing.
36. **Tests depend on external files.** The evaluation tool and the real-document test expect a sibling folder called "Case Study 2 Supporting Material", which is not in the zip.

---

## 15. Security notes

- **`.env` contains credentials.** It includes a `DATABASE_URL` in PostgreSQL format, which usually holds a username and password. Do not commit it or share the zip. Rotate the password if it has been shared, and add `.env` to `.gitignore`.
- **`optiv_shield.db` holds metadata.** It contains file names, token names, entity types, scores and timestamps, but not original values. Even so, file names can be sensitive. Keep it out of public repositories.
- **The verbose CLI options print personal data.** `--show-text`, `--show-hidden`, `--save-chunks` and `--show-sanitized` all print or save original text.
- **Uploads are held in memory and in temporary files.** The temporary folder is removed after processing. The sanitized PDF is kept in memory until the session ends.
- **The app has no authentication.** Anyone who can reach the Streamlit server can upload documents and see results from the current session. Run it on a trusted network only.
- **Analytics are shared.** The dashboard reads the whole database, so every user of the app sees the same history.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| PII | Personally identifiable information: any data that identifies a person |
| Presidio | Microsoft's open-source framework for detecting and anonymising PII |
| NER | Named entity recognition: a model that labels spans of text as people, places and so on |
| OCR | Optical character recognition: reading text from images |
| Tesseract | The open-source OCR engine used here |
| Bounding box | Coordinates of a word on a page, used to place a redaction |
| Redaction | Covering text in a PDF so it cannot be read, and removing the text beneath |
| Rescan | Running extraction and detection again on the sanitized output to prove it is clean |
| Token | A placeholder such as `[PERSON_001]` that replaces a sensitive value consistently |
| Chunk | The extracted content of one page, as an `ExtractionInput` |
| Ground truth | Known correct answers used to measure extraction quality |
| Unread region | An area of ink that OCR could not read, reported so it is not silently ignored |
