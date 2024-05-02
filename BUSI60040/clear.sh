# Remove LaTeX auxiliary files
rm -f *.aux
rm -f *.log
rm -f *.out
rm -f *.toc
rm -f *.fls
rm -f *.fdb_latexmk
rm -f *.synctex.gz
rm -f *.bbl
rm -f *.blg
rm -f *.lof
rm -f *.lot

# Remove minted cache directory
rm -rf _minted-*

# Remove any .aux files in subdirectories (like titlepage/)
find . -name "*.aux" -type f -delete

echo "Cache and unneeded files cleared."