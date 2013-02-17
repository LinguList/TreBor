# author   : Johann-Mattis List
# email    : mattis.list@gmail.com
# created  : 2013-02-17 12:47
# modified : 2013-02-17 12:47
"""
Test the method for tree-based borrowing detection.
"""

__author__="Johann-Mattis List"
__date__="2013-02-17"

# append library path to sys.path
import sys
sys.path.append('../../')

# import colormap from mpl
import matplotlib as mpl

# import trebor
from TreBor.trebor import TreBor

# load a trebor-object
tr = TreBor('test')

# define the runs
runs = [
    ('weighted',(3,1)),
    ('weighted',(2,1)),
    ('weighted',(1,1)),
    ('weighted',(5,2)),
    ('weighted',(3,2)),
    ('restriction',2),
    ('restriction',3),
    ('restriction',4)
    ]

# carry out the analysis
tr.analyze(
        runs = runs,
        full_analysis = True,
        plot_dists = True,
        threshold = 1,
        fileformat = 'pdf',
        colormap = mpl.cm.jet,
        usetex = False
        )

