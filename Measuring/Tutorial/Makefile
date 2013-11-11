REPORT_TEX = tutorial
MAIN_TEX = content
BIBLIO = bibliography

all: tutorial clean

tutorial: tutorial-pdf
tutorial-proof-read: tutorial-pdf
	evince $(REPORT_TEX).pdf

tutorial-pdf: tutorial-dvi
	dvipdf $(REPORT_TEX)

tutorial-dvi: tutorial.tex content.tex
	latex  $(REPORT_TEX)
	bibtex $(REPORT_TEX)
	latex  $(REPORT_TEX)
	latex  $(REPORT_TEX)

clean:
	$(RM) *.bbl *.lot *.log *.toc *.blg *.lof *.aux *.nav *.snm *.dvi *.out
