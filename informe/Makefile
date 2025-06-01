TEXFILE=main
PDF=$(TEXFILE).pdf

LATEX=pdflatex
LATEX_FLAGS=-interaction=nonstopmode -shell-escape
BIBTEX=bibtex

AUXFILES=$(TEXFILE).aux $(TEXFILE).log $(TEXFILE).out $(TEXFILE).toc \
         $(TEXFILE).bbl $(TEXFILE).blg $(TEXFILE).fdb_latexmk \
         $(TEXFILE).fls $(TEXFILE).lof $(TEXFILE).lot $(TEXFILE).nav \
         $(TEXFILE).snm $(TEXFILE).synctex.gz

all: force
	$(LATEX) $(LATEX_FLAGS) $(TEXFILE)
	$(BIBTEX) $(TEXFILE) || true
	$(LATEX) $(LATEX_FLAGS) $(TEXFILE)
	$(LATEX) $(LATEX_FLAGS) $(TEXFILE)


clean:
	rm -f $(AUXFILES)

distclean: clean
	rm -f $(PDF)

watch:
	latexmk -pdf -pvc -shell-escape $(TEXFILE)
force:

.PHONY: all clean distclean watch force
