cd paper
pandoc --citeproc --bibliography=refs.bib --variable papersize=a4paper --variable geometry="margin=2.5cm" --pdf-engine=xelatex -s paper.md -o paper.pdf
