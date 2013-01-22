from trebor import *
from sys import argv

if len(argv) == 1:
    data = 'dogon'
else:
    data = argv[1]

tr = TreBor(data,verbose=True)

tr.analyze(plot_dists=True)

tr.get_MLN('w-2-1')

