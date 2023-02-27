# rsa_reliability

This repository hosts scripts to generate some of the figures in the manuscript as well as a notebook to understand the lower noise ceiling analysis.

Live Notebook: https://blog.smazurchuk.com/rsa_reliability/ 

Thy jupyter notebooks are designed to run right in the browser with no need to download the data. To enable this, only a small Excel file with 10 resamples is included in this repository. The neural RDMs and Excel files having 1,000 resamples used to generate the figures in the manuscript can be found at:

https://osf.io/wsrfh/?view_only=4c7295fa574e474689b983dfceadd6b6 

## Cloning the repository

The live jupyter notebook is built using jupyter-lite and the command:

> jupyter lite build --contents content --force --output-dir docs