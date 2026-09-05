# Journal of Cheminformatics submission build.
#
# article.md, bibliography.bib and the supplementary Markdown are GENERATED
# from the frozen manuscript sources. Regenerate them, never edit them:
#
#     .\backend\.venv\Scripts\python.exe -m molfusion_backend.benchmark.submission_cli
#
# That command also writes a .docx and a review .pdf using dependency-free
# writers, so the package is complete without pandoc. The targets below use
# the journal's own pandoc route and produce submission-quality typesetting;
# they require pandoc >= 2.12, which is not installed in this environment.

PANDOC ?= pandoc
FILTERS = --lua-filter=filters/extract-cito.lua --citeproc \
          --lua-filter=filters/insert-cito-in-ref.lua
CSL     = --csl=journal-of-cheminformatics.csl

.PHONY: all clean check
all: pandoc-docx pandoc-pdf

pandoc-docx: article.md bibliography.bib
	$(PANDOC) --from=markdown --to=docx $(FILTERS) $(CSL) \
	    --output=MolFusion_JCheminformatics_Manuscript_pandoc.docx article.md

pandoc-pdf: article.md bibliography.bib
	$(PANDOC) --from=markdown --to=pdf $(FILTERS) $(CSL) \
	    --output=MolFusion_JCheminformatics_Manuscript_pandoc.pdf article.md

pandoc-si: supplementary/supplementary_information.md
	$(PANDOC) --from=markdown --to=docx \
	    --output=supplementary/MolFusion_Supplementary_Information_pandoc.docx \
	    supplementary/supplementary_information.md

# Re-run the submission guards against whatever artifacts exist.
check:
	cd ../../../../backend && ./.venv/Scripts/python.exe -m pytest tests/test_submission_package.py -q

clean:
	rm -f *_pandoc.docx *_pandoc.pdf supplementary/*_pandoc.docx
