from trebor import *

tr = TreBor('sindial',verbose=True)

stats = 0
#for cog in tr.cogs:
#    test = tr.get_restricted_gls(tr.paps[cog],restriction=10)
#    if len(test) > 1:
#        sA = sum([t[1] for t in test[0]])
#        sB = sum([t[1] for t in test[1]])
#        if sA < sB:
#            stats += 1
#            print(sA,sB)
#    elif len(test) == 1:
#        stats += 1



#print(stats / len(tr.cogs))


tr.get_gls(
        mode = 'weighted',
        ratio = (3,1),
        verbose = True,
        output_gml = True,
        tar = True
        )

