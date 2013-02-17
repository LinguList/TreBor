Tree-Based Detection of Borrowings in Lexicostatistic Datasets
==============================================================

This code uses different gain-loss mapping techniques to detect borrowings in lexicostatistic datasets.
The code will be later included as part of LingPy (https://github.com/lingpy/lingpy).


# Requirements

## Software

In order to be fully functionable, the code requires 
* NumPy (http://numpy.org), 
* SciPy (http://scipy.org), 
* Matplotlib (http://matplotlib.org), 
* Basemap (http://matplotlib.org/basemap/), and 
* the most recent version of LingPy (http://github.com/lingpy/lingpy).
 
## Input Data

Currently, the input data for a full analysis, six files, with specific file-extensions, are required.
This is surely not the most economic solution, and we will try to reduce the number of input files in the future.

* file.coords: A file that gives the coordinates of the data in csv-format, with the first column indicating the name of the taxon, the second column indicating the longitude, and the third column indicating the latitude.
* file.csv: A file containing the core data (cognate sets, words, taxa) in Wordlist-format (see: https://github.com/lingpy/lingpy/blob/master/doc/source/tutorial/lingpy.basic.wordlist.rst).
* file.gml: A file containing a gml-representation of the reference tree, including the x/y-coordinates for the placement of nodes. Currently, these coordinates are not created automatically, and have to be provided by the user. We recommend to use Cytoscape (http://www.cytoscape.org) for this purpose.
* file.groups: A file containing the name of the taxa in the first column and the name of the groups (dialects, subgroups) in the second column.
* file.json: A file containing specific format-parameters for the output of the plots. For all details of this format, check the example files provided in the test directory.
* file.tre: A file containing the reference tree in Newick format.




