# ST444-Group3
Statistical Computing (ST444) group project - Parallelizing Optimisation Algorithms

## GitHub workflow :construction_worker:

From the command line, make sure you have git installed by running

`git --version`

Navigate to where you want to repository to live on your computer and copy it here by running

`git clone https://github.com/vviers/ST444-Group3.git`

Before starting to work, always use `git pull origin master` so that you are up-to-date with all the code. 

Open a new _branch_ so that your work does not erase everyone else's work when we collaborate (we can merge your branch onto the master branch later)

`git checkout -b <name of your branch>`

Open a jupyter notebook by typing `jupyter notebook` at the command line. Get work done :smile:

Push your work onto your branch:
```
git add .
git commit -m "my commit message"
git push origin <name of your branch>
```

Navigate to https://github.com/vviers/ST444-Group3, you will see a `Submit PULL REQUEST` button asociated with your branch. Click on it and describe what you did.

You're done!


## Generating the report

To convert to pdf with bilbiography and table of contents, mae sure you have latest Pandoc and execute via command line:

```bash
cd path/to/ST444-Group3/report

jupyter nbconvert --to markdown Report.ipynb

pandoc --metadata title="Particle Swarm Optimisation" --metadata author="Group 3" -N --toc -V fontsize=12pt -V geometry:margin=1in --listings -H listings-setup.tex --filter pandoc-citeproc --bibliography=references.bib --csl=apa.csl Report.md --pdf-engine=xelatex -o Group3report.pdf
```

(The `--pdf-engine=xelatex` argument might be optional (in Ubuntu 18.04))