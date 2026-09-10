# Journal of Cheminformatics submission build.
#
# article.md, bibliography.bib and the supplementary Markdown are GENERATED
# from the frozen manuscript sources. Regenerate them, never edit them:
#
#     .\backend\.venv\Scripts\python.exe -m molfusion_backend.benchmark.submission_cli
#     .\backend\.venv\Scripts\python.exe -m molfusion_backend.benchmark.supplementary_cli
#
# Verified toolchain (Phase 6C.7): pandoc 3.11, typst 0.15.1.
# Pandoc cannot produce a PDF on its own; typst is the PDF engine because it
# is a single small binary and pandoc 3.11 drives it natively, where a full
# LaTeX distribution would be a far larger install for the same output.

PANDOC ?= pandoc
FILTERS = --lua-filter=filters/extract-cito.lua --citeproc \
          --lua-filter=filters/insert-cito-in-ref.lua
CSL     = --csl=journal-of-cheminformatics.csl
PDFENG  = --pdf-engine=typst

MANUSCRIPT = MolFusion_JCheminformatics_Manuscript_FINAL
SI         = supplementary/MolFusion_Supplementary_Information_FINAL

.PHONY: all final si rendered check clean
all: final si rendered

final: $(MANUSCRIPT).docx $(MANUSCRIPT).pdf

$(MANUSCRIPT).docx: article.md bibliography.bib
	$(PANDOC) --from=markdown --to=docx $(FILTERS) $(CSL) --output=$@ article.md

$(MANUSCRIPT).pdf: article.md bibliography.bib
	$(PANDOC) --from=markdown --to=pdf $(PDFENG) $(FILTERS) $(CSL) --output=$@ article.md

# CSL-processed plain text: what the PDF typesets, in an auditable form.
rendered: validation/article_rendered.txt
validation/article_rendered.txt: article.md bibliography.bib
	$(PANDOC) --from=markdown --to=plain --wrap=none $(FILTERS) $(CSL) --output=$@ article.md

si: $(SI).docx $(SI).pdf
$(SI).docx: supplementary/supplementary_information.md
	$(PANDOC) --from=markdown --to=docx --output=$@ $<
$(SI).pdf: supplementary/supplementary_information.md
	$(PANDOC) --from=markdown --to=pdf $(PDFENG) --output=$@ $<

check:
	cd ../../../../backend && ./.venv/Scripts/python.exe -m pytest tests/test_submission_package.py -q

clean:
	rm -f $(MANUSCRIPT).docx $(MANUSCRIPT).pdf $(SI).docx $(SI).pdf \
	      validation/article_rendered.txt
