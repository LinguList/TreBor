# created: 2013-01-21
# modified: 2013-01-25

__author__ = "Johann-Mattis List"
__date__ = "2013-01-21"

"""
Tree-based detection of borrowings in lexicostatistical wordlists.
"""

# basic imports
import os
import json

# thirdparty imports
import numpy as np
import networkx as nx
import scipy.stats as sps
import numpy.linalg as linalg

# import error classes
from lingpy.check.exceptions import *
from lingpy.check.messages import *

# mpl is only used for specific plots, we can therefor make a safe import
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    ThirdPartyModuleError('matplotlib').warning()

# import the geoplot module
try:
    import mpl_toolkits.basemap as bmp
except ImportError:
    ThirdPartyModuleError('basemap').warning()

# import polygon
#try:
from .polygon import getConvexHull
#except:
#    ThirdPartyModuleError('polygon').warning()

# lingpy imports
from lingpy.thirdparty import cogent as cg
from lingpy.convert.gml import *
from lingpy.basic import Wordlist
from lingpy.read.csv import csv2dict,csv2list

class TreBor(object):
    """
    Basic class for calculations using the TreBor method.
    """

    # XXX generally: find a way to check whether a dataset was already loaded,
    # XXX otherwise it takes too long a time to recalculate everything
    
    def __init__(
            self,
            dataset,
            tree = None,
            paps = 'pap',
            verbose = False,
            ):
        """
        
        Parameters
        ----------
        dataset : string
            Name of the dataset that shall be analyzed.
        tree : {None, string}
            Name of the tree file.
        paps : string (default="pap")
            Name of the column that stores the specific cognate IDs consisting
            of an arbitrary integer key and a key for the concept.

        """
        # store the name of the dataset and the identifier for paps
        self.dataset = dataset
        self._pap_string = paps

        # open csv-file of the data and store it as a word list attribute
        self.Wordlist = Wordlist(dataset+'.csv')
        self.wl = self.Wordlist

        if verbose: print("[i] Loaded the wordlist file.")

        # check for glossid
        if 'glid' not in self.wl.entries:
            self._gl2id = dict(
                    zip(
                        self.wl.rows,
                        [i+1 for i in range(len(self.wl.rows))]
                        )
                    )
            self._id2gl = dict([(b,a) for a,b in self._gl2id.items()])

            f = lambda x: self._gl2id[x]

            self.wl.add_entries(
                    'glid',
                    'concept',
                    f
                    )

        # check for paps as attribute in the wordlist
        if paps not in self.wl.entries:
            
            # define the function for conversion
            f = lambda x,y: "{0}:{1}".format(x[y[0]],x[y[1]])
            self.wl.add_entries(
                    paps,
                    'cogid,glid',
                    f
                    )

            if verbose: print("[i] Created entry PAP.")
        
        # get the paps and the etymological dictionary
        self.paps = self.wl.get_paps(ref=paps)

        if verbose: print("[i] Created the PAP matrix.")

        # get a list of concepts corresponding to the cogs and get the
        # singletons to be excluded from the calculation
        tmp = self.wl.get_etymdict(ref=paps,entry='concept')
        
        # a dictionary with pap-key as key and concept as value
        self.concepts = {}

        # list stores the singletons
        self.singletons = []

        for key in self.paps:
            
            # get the names of the concepts
            concept_list = [k for k in tmp[key] if k != 0]
            concept = concept_list[0][0]
            self.concepts[key] = concept

            # check for singletons
            if sum(self.paps[key]) == 1:
                self.singletons.append(key)
        
        # create a list of keys for faster access when iterating
        self.cogs = [k for k in self.concepts if k not in self.singletons]

        if verbose: print("[i] Excluded singletons.")

        # Load the tree, if it is not defined, assume that the treefile has the
        # same name as the dataset
        if not tree:
            # try to load the tree first
            try:
                self.tree = cg.LoadTree(dataset+'.tre')
            except:
                # create it otherwise
                print("[i] Tree-file was not found, creating it now...")
                pass
            # XXX TODO
        
        # if it is explicitly defined, try to load that file
        else:
            self.tree = LoadTree(tree)

        if verbose: print("[i] Loaded the tree.")

        # get the taxa
        self.taxa = self.wl.cols

        if verbose: print("[i] Assigned the taxa.")

        # create a stats-dictionary
        self.stats = {}

        # create gls-dictionary
        self.gls = {}

        # create dictionary for distributions
        self.dists = {}

        # create dictionary for graph attributes
        self.graph = {}
    
    def get_weighted_GLS(
            self,
            pap,
            ratio = (1,1),
            verbose = False
            ):
        """
        Calculate a weighted gain-loss-scenario (WGLS) for a given PAP.
        """
        
        # make a dictionary that stores the scenario
        d = {}

        # get the subtree containing all taxa that have positive paps
        tree = self.tree.lowestCommonAncestor(
                [
                    self.taxa[i] for i in range(len(self.taxa)) if pap[i] >= 1
                    ]
                )
        if verbose: print("[i] Subtree is {0}.".format(str(tree)))

        # assign the basic (starting) values to the dictionary
        nodes = [t.Name for t in tree.tips()]

        if verbose: print("[i] Nodes are {0}.".format(','.join(nodes)))

        # get the first state of all nodes and store the state in the
        # dictionary. note that we start from two distinct scenarios: one
        # assuming single origin at the root where all present states in the
        # leave are treated as retentions, and one assuming multiple origins,
        # where all prsent states in the leaves are treated as origins
        for node in nodes:
            idx = self.taxa.index(node)
            if pap[idx] >= 1:
                state = 1
            else:
                state = 0
            d[node] = [(state,[])]

        # return simple scenario, if the group is single-origin
        if sum([d[node][0][0] for node in nodes]) == len(nodes):
            return [(tree.Name,1)]

        # order the internal nodes according to the number of their leaves
        ordered_nodes = sorted(
                tree.nontips()+[tree],key=lambda x:len(x.tips())
                )

        # calculate the general restriction value (maximal weight). This is roughly
        # spoken simply the minimal value of either all events being counted as
        # origins (case 1) or assuming origin of the character at the root and
        # counting all leaves that lost the character as single loss events (case
        # 2). In case two, the first gain of the character has to be added
        # additionally
        maxWeight = min(pap.count(1) * ratio[0], pap.count(0) * ratio[1] + ratio[0])

        # join the nodes successively
        for i,node in enumerate(ordered_nodes):
            if verbose: print(node.Name)
            
            # get the name of the children of the nodes
            nameA,nameB = [x.Name for x in node.Children]

            # get the nodes with their states from the dictionary
            nodesA,nodesB = [d[x.Name] for x in node.Children]

            # there might be alternative states, therefore it is important to
            # iterate over all possible paths
            newNodes = []
            for nodeA in nodesA:
                for nodeB in nodesB:
                    # if the nodes have the same state, the state is assigned to
                    # the most recent common ancestor node
                    if nodeA[0] == nodeB[0]:

                        # combine the rest of the histories of the items
                        tmp = nodeA[1] + nodeB[1]

                        # add them to the queue only if their weight is less or
                        # equal to the maxWeight
                        gl = [k[1] for k in tmp]+[x for x in [nodeA[0]] if x == 1]
                        weight = gl.count(1) * ratio[0] + gl.count(0) * ratio[1]

                        if weight <= maxWeight:
                            newNodes.append((nodeA[0],tmp))

                    # if the nodes have different states, we go on with two
                    # distinct scenarios
                    else:
                        
                        # first scenario assumes retention of nodeB
                        tmpA = nodeA[1] + nodeB[1] + [(nameA,nodeA[0])]

                        # second scenario assumes retention of nodeA
                        tmpB = nodeA[1] + nodeB[1] + [(nameB,nodeB[0])]

                        # get the vectors in order to make it easier to retrieve
                        # the number of losses and gains
                        glA = [k[1] for k in tmpA] + [x for x in [nodeB[0]] if x == 1]
                        glB = [k[1] for k in tmpB] + [x for x in [nodeA[0]] if x == 1]

                        # check the gain-loss scores 
                        weightA = glA.count(1) * ratio[0] + glA.count(0) * ratio[1]
                        weightB = glB.count(1) * ratio[0] + glB.count(0) * ratio[1]
                        
                        # check whether an additional gain is inferred on either of
                        # the two possible paths. 
                        # XXX reduce later, this is not efficient XXX
                        if nodeB[0] == 1 and 1 in [k[1] for k in tmpA]:
                            noA = True
                        else:
                            noA = False
                        if nodeA[0] == 1 and 1 in [k[1] for k in tmpB]:
                            noB = True
                        else:
                            noB = False

                        newNodeA = nodeB[0],tmpA
                        newNodeB = nodeA[0],tmpB
                                            
                        # if the weights are above the theoretical maximum, discard
                        # the solution
                        if weightA <= maxWeight and not noA:
                            newNodes += [newNodeA]
                        if weightB <= maxWeight and not noB:
                            newNodes += [newNodeB]

                d[node.Name] = newNodes
                if verbose: print("nodelen",len(d[node.Name]))
        
        # try to find the best scenario by counting the ratio of gains and losses.
        # the key idea here is to reduce the number of possible scenarios according
        # to a given criterion. We choose the criterion of minimal changes as a
        # first criterion to reduce the possibilities, i.e. we weight both gains
        # and losses by 1 and select only those scenarios where gains and losses
        # sum up to a minimal number of gains and losses. This pre-selection of
        # scenarios can be further reduced by weighting gains and losses
        # differently. So in a second stage we choose only those scenarios where
        # there is a minimal amount of gains. 
        
        if verbose: print(len(d[tree.Name]))

        # convert the specific format of the d[tree.Name] to simple format
        gls_list = []
        for first,last in d[tree.Name]:
            if first == 1:
                gls_list.append([(tree.Name,first)]+last)
            else:
                gls_list.append(last)

        # the tracer stores all scores
        tracer = []

        for i,line in enumerate(gls_list):
            
            # calculate gains and losses
            gains = sum([1 for x in line if x[1] == 1])
            losses = sum([1 for x in line if x[1] == 0])

            # calculate the score
            score = ratio[0] * gains + ratio[1] * losses

            # append it to the tracer
            tracer.append(score)
        
        # get the minimum score
        minScore = min(tracer)

        # return the minimal indices, sort them according to the number of
        # gains inferred, thereby pushing gains to the root, similar to
        # Mirkin's (2003) suggestion
        return sorted(
                [gls_list[i] for i in range(len(tracer)) if tracer[i] == minScore],
                key = lambda x:sum([i[1] for i in x])
                )[0]

    def get_restricted_GLS(
            self,
            pap,
            restriction = 4,
            verbose = False
            ):
        """
        Calculate a restricted gain-loss-scenario (RGLS) for a given PAP.
    
        """

        # make a dictionary that stores the scenario
        d = {}

        # get the subtree containing all taxa that have positive paps
        tree = self.tree.lowestCommonAncestor(
                [self.taxa[i] for i in range(len(self.taxa)) if pap[i] >= 1]
                )

        # assign the basic (starting) values to the dictionary
        nodes = [x.Name for x in tree.tips()]

        # get the first state of all nodes and store the state in the dictionary.
        # note that we start from two distinct scenarios: one assuming single
        # origin at the root, where all present states in the leave are treated as
        # retentions, and one assuming multiple origins, where all present states
        # in the leaves are treated as origins
        for node in nodes:
            idx = self.taxa.index(node)
            if pap[idx] >= 1:
                state = 1
            else:
                state = 0
            d[node] = [(state,[])]

        # return simple scenario if the group is single-origin
        if sum([d[node][0][0] for node in nodes]) == len(nodes):
            return [(tree.Name,1)]

        # order the internal nodes according to the number of their leaves
        ordered_nodes = sorted(
                tree.nontips()+[tree],key=lambda x:len(x.tips())
                )

        # join the nodes successively
        for i,node in enumerate(ordered_nodes):
            if verbose: print(node)
            
            # get the name of the children of the nodes
            nameA,nameB = [x.Name for x in node.Children]

            # get the nodes with their states from the dictionary
            nodesA,nodesB = [d[x.Name] for x in node.Children]

            # there might be alternative states, therefore it is important to
            # iterate over all possible paths
            newNodes = []
            for nodeA in nodesA:
                for nodeB in nodesB:
                    # if the nodes have the same state, the state is assigned to
                    # the most recent common ancestor node
                    if nodeA[0] == nodeB[0]:

                        # combine the rest of the histories of the items
                        tmp = nodeA[1] + nodeB[1]

                        # append only if tmp is in concordance with maximal_gains
                        if len([k for k in tmp if k[1] == 1]) + nodeA[0] <= restriction:
                            newNodes.append((nodeA[0],tmp))

                    # if the nodes have different states, we go on with two
                    # distinct scenarios
                    else:
                        
                        # first scenario assumes retention of nodeA
                        tmpA = nodeA[1] + nodeB[1] + [(nameA,nodeA[0])]

                        # second scenario assumes retention of nodeB
                        tmpB = nodeA[1] + nodeB[1] + [(nameB,nodeB[0])]

                        newNodeA = nodeB[0],tmpA
                        newNodeB = nodeA[0],tmpB
                        
                        # check whether one of the solutions is already above the
                        # maximum / best scenario with respect to the gain-loss
                        # weights as a criterion

                        # first, calculate gains and losses
                        gainsA = nodeB[0]+sum([k[1] for k in tmpA])
                        gainsB = nodeA[0]+sum([k[1] for k in tmpB])

                        # check whether an additional gain is inferred on either of
                        # the two possible paths. 
                        # XXX reduce later, this is not efficient XXX
                        if nodeB[0] == 1 and 1 in [k[1] for k in tmpA]:
                            noA = True
                        else:
                            noA = False
                        if nodeA[0] == 1 and 1 in [k[1] for k in tmpB]:
                            noB = True
                        else:
                            noB = False
                                            
                        # if the gains are about the theoretical maximum, discard
                        # the solution
                        if gainsA <= restriction and not noA:
                            newNodes += [newNodeA]
                        if gainsB <= restriction and not noB:
                            newNodes += [newNodeB]

                d[node.Name] = newNodes
        
        # try to find the best scenario by counting the ratio of gains and losses.
        # the key idea here is to reduce the number of possible scenarios according
        # to a given criterion. We choose the criterion of minimal changes as a
        # first criterion to reduce the possibilities, i.e. we weight both gains
        # and losses by 1 and select only those scenarios where gains and losses
        # sum up to a minimal number of gains and losses. This pre-selection of
        # scenarios can be further reduced by weighting gains and losses
        # differently. So in a second stage we choose only those scenarios where
        # there is a minimal amount of gains. 
        
        if verbose: print(len(d[tree.Name]))

        # convert the specific format of the d[tree.Name] to simple format
        gls_list = []
        for first,last in d[tree.Name]:
            if first == 1:
                gls_list.append([(tree.Name,first)]+last)
            else:
                gls_list.append(last)

        # the tracer stores all scores
        tracer = []

        for i,line in enumerate(gls_list):
            
            # calculate gains and losses
            gains = sum([1 for x in line if x[1] == 1])
            losses = sum([1 for x in line if x[1] == 0])

            # calculate the score
            score = gains + losses

            # append it to the tracer
            tracer.append(score)
        
        # get the minimum score
        minScore = min(tracer)

        # return minimal indices
        return sorted(
                [gls_list[i] for i in range(len(tracer)) if tracer[i] == minScore],
                key = lambda x:len(x)
                )[0]

    def get_GLS(
            self,
            mode = 'weighted',
            ratio = (1,1),
            restriction = 3,
            output_gml = False,
            verbose = False,
            tar = False
            ):
        """
        Create gain-loss-scenarios for all non-singleton paps in the data.

        Parameters
        ----------
        mode : string (default="weighted")
            Select between "weighted" and "restriction".
        ratio : tuple (default=(1,1))
            If "weighted" mode is selected, define the ratio between the
            weights for gains and losses.
        restriction : int (default=3)
            If "restriction" is selected as mode, define the maximal number of
            gains.
        output_gml : bool (default=False)
            If set to c{True}, the decisions for each GLS are stored in a
            separate file in GML-format.
        tar : bool (default=False)
            If set to c{True}, the GML-files will be added to a compressed tar-file.

        """
        if mode not in ['weighted','w','r','restriction']:
            raise ValueError("[!] The mode {0} is not available".format(mode))

        # define alias for mode
        if mode in ['w','weighted']:
            mode = 'weighted'
        else:
            mode = 'restriction'

        # create a named string for the mode
        if mode == 'weighted':
            mode_string = 'w-{0[0]}-{0[1]}'.format(ratio)
        elif mode == 'restriction':
            mode_string = 'r-{0}'.format(restriction)
        
        # create statistics for this run
        self.stats[mode_string] = {}

        # store the statistics
        self.stats[mode_string]['mode'] = mode
        self.stats[mode_string]['dataset'] = self.dataset

        # attribute stores all gls for each cog
        
        self.gls[mode_string] = {}
        #self.stats[mode_string]['gls'] = {}

        for cog in self.cogs:
            if verbose: print("[i] Calculating GLS for COG {0}...".format(cog),end="")
            if mode == 'weighted':
                gls = self.get_weighted_GLS(
                        self.paps[cog],
                        ratio = ratio
                        )

            if mode == 'restriction':
                gls = self.get_restricted_GLS(
                        self.paps[cog],
                        restriction = restriction
                        )

            noo = sum([t[1] for t in gls])
            #self.stats[mode_string]['gls'][cog] = sum([t[1] for t in gls])
            
            self.gls[mode_string][cog] = (gls,noo)


            # attend scenario to gls
            if verbose: print(" done.")
        if verbose: print("[i] Successfully calculated Gain-Loss-Scenarios.")
 
        # write the results to file
        # make the folder for the data to store the stats
        folder = self.dataset+'_trebor'
        try:
            os.mkdir(folder)
        except:
            pass       
        
        # if output of gls is chosen, load the gml-graph
        if output_gml:

            # make the directory for the files
            try:
                os.mkdir(folder+'/gml')
            except:
                pass

            # make next directory
            try:
                os.mkdir(
                        folder+'/gml/'+'{0}-{1}'.format(
                            self.dataset,
                            mode_string
                            )
                        )
            except:
                pass

            # load the graph
            self.graph = nx.read_gml(self.dataset+'.gml')

            # store the graph
            for cog in self.cogs:
                gls = self.gls[mode_string][cog][0]
                gls2gml(
                        gls,
                        self.graph,
                        self.tree,
                        filename = folder+'/gml/{0}-{1}/{2}'.format(
                            self.dataset,
                            mode_string,
                            cog
                            ),
                        )

            # if tar is chosen, put it into a tarfile
            if tar:
                os.system(
                        'cd {0}_trebor/gml/ ; tar -pczf {0}-{1}.tar.gz {0}-{1}; cd ..; cd ..'.format(
                            self.dataset,
                            mode_string
                            )
                        )
                os.system('rm {0}_trebor/gml/{0}-{1}/*.gml'.format(self.dataset,mode_string))
                os.system('rmdir {0}_trebor/gml/{0}-{1}'.format(self.dataset,mode_string))


        # store some statistics as attributes
        self.stats[mode_string]['ano'] = sum(
                [v[1] for v in self.gls[mode_string].values()]
                ) / len(self.gls[mode_string])
        self.stats[mode_string]['mno'] = max([v[1] for v in self.gls[mode_string].values()])
        self.stats[mode_string]['ratio'] = ratio 
        self.stats[mode_string]['restriction'] = restriction

        # store statistics and gain-loss-scenarios in textfiles
        # create folder for gls-data
        try:
            os.mkdir(folder+'/gls')
        except:
            pass
        
        if verbose: print("[i] Writing GLS data to file... ",end="")
        
        # write gls-data to folder
        f = open(folder+'/gls/{0}-{1}.gls'.format(self.dataset,mode_string),'w')
        f.write('PAP\tGainLossScenario\tNumberOfOrigins\n')
        for cog in sorted(self.gls[mode_string]):
            gls,noo = self.gls[mode_string][cog]
            f.write(
                    "{0}\t".format(cog)+','.join(
                        ["{0}:{1}".format(a,b) for a,b in gls]
                        ) + '\t'+str(noo)+'\n'
                    )
        f.close()
        if verbose: print("done.")

        
        # print out average number of origins
        if verbose: print("[i] Average Number of Origins: {0:.2f}".format(self.stats[mode_string]['ano']))

        # write statistics to stats file
        try:
            os.mkdir(folder+'/stats')
        except:
            pass

        f = open(folder+'/stats/{0}-{1}'.format(self.dataset,mode_string),'w')
        f.write('Number of PAPs (total): {0}\n'.format(len(self.paps)))
        f.write('Number of PAPs (non-singletons): {0}\n'.format(len(self.gls[mode_string])))
        f.write('Number of Singletons: {0}\n'.format(len(self.singletons)))
        f.write('Average Number of Origins: {0:.2f}\n'.format(self.stats[mode_string]['ano']))
        f.write('Maximum Number of Origins: {0}\n'.format(self.stats[mode_string]['mno']))
        f.write('Mode: {0}\n'.format(mode))
        if mode == 'weighted':
            f.write('Ratio: {0[0]} / {0[1]}\n'.format(ratio))
        elif mode == 'restriction':
            f.write('Restriction: {0}\n'.format(restriction))

        f.close()

        return

    def get_CVSD(
            self,
            verbose = False
            ):
        """
        Calculate the Contemporary Vocabulary Size Distribution (CVSD).

        """
        # XXX todo: Note that form/meaning distributions are problematic, we have
        # XXX find a much better way to cope with this

        # define taxa and concept as attribute for convenience
        taxa = self.taxa
        concepts = self.wl.concept

        # create list to store the forms and the concepts
        dist = [] 
        
        # calculate vocabulary size
        size = []
        for taxon in taxa:
            s = len([
                    x for x in set(
                        self.wl.get_list(
                            col=taxon,
                            entry=self._pap_string,
                            flat = True
                            )
                        ) if x in self.cogs
                    ])
            size += [s]
        
        # store the stuff as an attribute
        self.dists['contemporary'] = size

        if verbose: print("[i] Calculated the distributions for contemporary taxa.")
        
        return 

    def get_AVSD(
            self,
            mode_string,
            verbose = False
            ):
        """
        Function retrieves all paps for ancestor languages in a given tree.
        """

        # XXX todo: Note that form/meaning distributions are problematic, we have
        # XXX find a much better way to cope with this

        # define concepts for convenience
        concepts = self.wl.concept
        
        # get all internal nodes, i.e. the nontips and also the root
        nodes = ['root'] + sorted(
                [node.Name for node in self.tree.nontips()],
                key=lambda x: len(self.tree.getNodeMatchingName(x).tips()),
                reverse = True
                )

        # retrieve scenarios
        tmp = sorted([(a,b,c) for a,(b,c) in self.gls[mode_string].items()])
        cog_list = [t[0] for t in tmp]
        gls_list = [t[1] for t in tmp]
        noo_list = [t[2] for t in tmp]

        # create a list that stores the paps
        paps = [[0 for i in range(len(nodes))] for j in range(len(cog_list))]

        # iterate and assign values
        for i,cog in enumerate(cog_list):
            
            # sort the respective gls
            gls = sorted(
                    gls_list[i],
                    key = lambda x: len(self.tree.getNodeMatchingName(x[0]).tips()),
                    reverse = True
                    )

            # retrieve the state of the root
            if gls[0][1] == 1 and gls[0][0] == 'root':
                state = 1
            else:
                state = 0

            # assign the state of the root to all nodes
            paps[i] = [state for node in nodes]

            # iterate over the gls and assign the respective values to all
            # children
            for name,event in gls:
                if event == 1:
                    this_state = 1
                else:
                    this_state = 0

                # get the subtree nodes
                sub_tree_nodes = [node.Name for node in
                        self.tree.getNodeMatchingName(name).nontips()]

                # assign this state to all subtree nodes
                for node in sub_tree_nodes:
                    paps[i][nodes.index(node)] = this_state
        
        # get the vocabulary size
        size = [sum([p[i] for p in paps]) for i in range(len(nodes))]

        # store the stuff as an attribute
        self.dists[mode_string] = size

        if verbose: print("[i] Calculated the distributions for ancestral taxa.")

        return

    def get_MLN(
            self,
            mode_string,
            threshold = 1,
            verbose = False
            ):
        """
        Compute an evolutionary network for a given model.
        """
        
        # create the primary graph
        gPrm = nx.Graph()

        # create the template graph XXX add fallback procedure
        gTpl = nx.read_gml(self.dataset+'.gml')

        # make alias for tree and taxa for convenience
        taxa = self.taxa
        tree = self.tree

        # make alias for the current gls for convenience
        scenarios = self.gls[mode_string]

        # create dictionary for inferred lateral events
        ile = {}

        # create mst graph
        gMST = nx.Graph()

        # create out graph
        gOut = nx.Graph()

        # load data for nodes into new graph
        for node,data in gTpl.nodes(data=True):
            if data['label'] in taxa:
                data['graphics']['fill'] = '#ff0000'
                data['graphics']['type'] = 'rectangle'
                data['graphics']['w'] = 80.0
                data['graphics']['h'] = 20.0
            else:
                data['graphics']['type'] = 'ellipse'
                data['graphics']['w'] = 30.0
                data['graphics']['h'] = 30.0
                data['graphics']['fill'] = '#ff0000'
            gPrm.add_node(data['label'],**data)

        # load edge data into new graph
        for nodeA,nodeB,data in gTpl.edges(data=True):
            data['graphics']['width'] = 10.0
            data['graphics']['fill'] = '#000000'
            data['label'] = 'vertical'

            gPrm.add_edge(
                    gTpl.node[nodeA]['label'],
                    gTpl.node[nodeB]['label'],
                    **data
                    )

        # start to assign the edge weights
        for cog,(gls,noo) in scenarios.items():
            
            # get the origins
            oris = [x[0] for x in gls if x[1] == 1]

            # connect origins by edges
            for i,oriA in enumerate(oris):
                for j,oriB in enumerate(oris):
                    if i < j:
                        try:
                            gPrm.edge[oriA][oriB]['weight'] += 1
                        except:
                            gPrm.add_edge(
                                    oriA,
                                    oriB,
                                    weight=1
                                    )

        # verbose output
        if verbose: print("[i] Calculated primary graph.")
        
        # verbose output
        if verbose: print("[i] Inferring lateral edges...")
            
        # create MST graph
        gMST = nx.Graph()

        for cog,(gls,noo) in scenarios.items():
            
            ile[cog] = []

            # get the origins
            oris = [x[0] for x in gls if x[1] == 1]

            # create a graph of weights
            gWeights = nx.Graph()
            
            # iterate over nodes
            for i,nodeA in enumerate(oris):
                for j,nodeB in enumerate(oris):
                    if i < j:
                        w = 1000000 / int(gPrm.edge[nodeA][nodeB]['weight'])
                        gWeights.add_edge(
                                nodeA,
                                nodeB,
                                weight=w
                                )
            
            # if the graph is not empty
            if gWeights:
                
                # calculate the MST
                mst = nx.minimum_spanning_tree(gWeights)
                
                # assign the MST-weights to gMST
                for nodeA,nodeB in mst.edges():
                    try:
                        gMST.edge[nodeA][nodeB]['weight'] += 1
                        gMST.edge[nodeA][nodeB]['cogs'] += [cog]
                    except:
                        gMST.add_edge(
                                nodeA,
                                nodeB,
                                weight=1,
                                cogs=[cog]
                                )
                    ile[cog]+= [(nodeA,nodeB)]

        # get colormap for edgeweights
        edge_weights = []
        for nodeA,nodeB,data in gMST.edges(data=True):
            edge_weights.append(data['weight'])
        
        # determine a colorfunction
        cfunc = np.array(np.linspace(0,256,len(set(edge_weights))),dtype='int')
        lfunc = np.linspace(0.5,8,len(set(edge_weights)))

        # sort the weights
        weights = sorted(set(edge_weights))

        # get the scale for the weights (needed for the line-width)
        scale = 20.0 / max(edge_weights)

        # load data for nodes into new graph
        for node,data in gTpl.nodes(data=True):
            if data['label'] in taxa:
                data['graphics']['fill'] = '#ff0000'
                data['graphics']['type'] = 'rectangle'
                data['graphics']['w'] = 80.0
                data['graphics']['h'] = 20.0
            else:
                data['graphics']['type'] = 'ellipse'
                data['graphics']['w'] = 30.0
                data['graphics']['h'] = 30.0
                data['graphics']['fill'] = '#ff0000'
            
            gOut.add_node(data['label'],**data)

        # load edge data into new graph
        for nodeA,nodeB,data in gTpl.edges(data=True):
            data['graphics']['width'] = 10.0
            data['graphics']['fill'] = '#000000'
            data['label'] = 'vertical'
            del data['graphics']['Line']

            gOut.add_edge(
                    gTpl.node[nodeA]['label'],
                    gTpl.node[nodeB]['label'],
                    **data
                    )
        
        # assign new edge weights
        for nodeA,nodeB,data in gMST.edges(data=True):
            w = data['weight']

            # get the color for the weight
            color = mpl.colors.rgb2hex(mpl.cm.jet(cfunc[weights.index(w)]))

            data['graphics'] = {}
            data['graphics']['fill'] = color
            data['graphics']['width'] = w * scale
            data['cogs'] = ','.join([str(i) for i in data['cogs']])
            data['label'] = 'horizontal'

            # check for threshold
            if w >= threshold:
                try:
                    gOut.edge[nodeA][nodeB]
                except:
                    # add the data to the out-graph
                    gOut.add_edge(
                        nodeA,
                        nodeB,
                        **data
                        )

        # verbose output
        if verbose: print("[i] Writing graph to file...")

        # write the graph to file
        f = open(self.dataset+'_trebor/mln-'+mode_string+'.gml','w')
        for line in nx.generate_gml(gOut):
            f.write(line+'\n')
        f.close()

        # write the inferred borrowing events (ILS, inferred lateral event) 
        # between all taxa to file
        # verbose output
        if verbose: print("[i] Writing Inferred Lateral Events to file...")

        f = open(self.dataset+'_trebor/ile-'+mode_string+'.csv','w')
        for cog,events in ile.items():
            if events:
                f.write(
                        cog+'\t'+','.join(
                            ['{0}:{1}'.format(a,b) for a,b in events]
                            )+'\n'
                        )
        f.close()

        # create file name for node labels (cytoscape output)
        f = open(self.dataset+'_trebor/node.label.NA','w')
        f.write("node.label (class=java.lang.String)\n")
        for taxon in taxa:
            f.write('{0} = {1}\n'.format(taxon,taxon))
        f.close()

        # add gOut to graphattributes
        self.graph[mode_string] = gOut

        # write stats to file
        f = open(self.dataset+'_trebor/taxa-'+mode_string+'.stats','w')
        
        # get the degree
        nodes = tree.getNodeNames()

        dgr,wdgr = [],[]
        for taxon in nodes:
            
            horizontals = [g for g in gOut[taxon] if 'weight' in gOut[taxon][g]]
            
            dgr.append(len(horizontals))
            wdgr.append(sum([gOut[taxon][g]['weight'] for g in horizontals]))

        sorted_nodes = sorted(
                zip(nodes,dgr,wdgr),
                key=lambda x:x[1],
                reverse=True
                )
        for n,d,w in sorted_nodes:
            f.write(
                    '{0}\t{1}\t{2}\t{3}\n'.format(
                        n,
                        str(tree.getNodeMatchingName(n)),
                        d,
                        w
                        )
                    )
        f.close()

        if verbose: print("[i] Wrote node degree distributions to file.")

        # write edge distributions
        f = open(self.dataset+'_trebor/edge-'+mode_string+'.stats','w')
        edges = []
        edges = [g for g in gOut.edges(data=True) if 'weight' in g[2]]

        for nA,nB,d in sorted(
                edges,
                key=lambda x: x[2]['weight'],
                reverse = True
                ):
            f.write(
                    '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(
                        nA,
                        nB,
                        d['weight'],
                        d['cogs'],
                        tree.getNodeMatchingName(nA),
                        tree.getNodeMatchingName(nB)
                        )
                    )
        f.close()
        if verbose: print("[i] Wrote edge-weight distributions to file.")

        return 

    def get_PDC(
            self,
            mode_string,
            verbose = False,
            **keywords
            ):
        """
        Calculate Patchily Distributed Cognates.
        """
        
        patchy = {}
        paps = []

        for key,(gls,noo) in self.gls[mode_string].items():

            # get the origins
            oris = [x[0] for x in gls if x[1] == 1]
            
            # get the tip-taxa for each origin
            tips = []

            # get the losses 
            tmp_loss = [x[0] for x in gls if x[1] == 0]
            losses = []
            for l in tmp_loss:
                losses += self.tree.getNodeMatchingName(l).getTipNames()

            for i,ori in enumerate(oris):
                tips += [
                        (
                            i+1,
                            [t for t in self.tree.getNodeMatchingName(
                                ori
                                ).getTipNames() if t not in losses]
                            )
                        ]

            # now, all set of origins with their tips are there, we store them
            # in the patchy dictionary, where each taxon is assigned the
            # numerical value of the given patchy dist
            patchy[key] = {}
            if len(tips) > 1:
                for i,tip in tips:
                    for taxon in tip:
                        patchy[key][taxon] = i
            else:
                for i,tip in tips:
                    for taxon in tip:
                        patchy[key][taxon] = 0
            
            paps.append((key,noo))
        
        if verbose: print("[i] Retrieved patchy distributions.")

        # get the index for the paps in the wordlist
        papIdx = self.wl.header['pap']
        taxIdx = self.wl._colIdx

        # create a dictionary as updater for the wordlist
        updater = {}
        for key in self.wl:

            # get the taxon first
            taxon = self.wl[key][taxIdx]

            # get the pap
            pap = self.wl[key][papIdx]
            
            try:
                updater[key] = '{0}:{1}'.format(pap,patchy[pap][taxon])
            except KeyError:
                updater[key] = '{0}:{1}'.format(pap,0)

        # update the wordlist
        self.wl.add_entries(
                'patchy',
                updater,
                lambda x:x
                )

        # write data to file
        self.wl.output('csv',filename=self.dataset+'_trebor/wl-'+mode_string)

        if verbose: print("[i] Updated the wordlist.")

        # write ranking of concepts to file
        f = open(self.dataset + '_trebor/paps-'+mode_string+'.stats','w')
        concepts = {}
        for a,b in sorted(paps,key=lambda x:x[1],reverse=True):
            
            a1,a2 = a.split(':')
            a3 = self._id2gl[int(a2)]
            
            try:
                concepts[a3] += [b]
            except:
                concepts[a3] = [b]

            f.write('{0}\t{1}\t{2}\t{3}\n'.format(a1,a2,a3,b))
        f.close()
        if verbose: print("[i] Wrote stats on paps to file.")

        # write stats on concepts
        f = open(self.dataset+'_trebor/concepts-'+mode_string+'.stats','w')
        for key in concepts:
            concepts[key] = sum(concepts[key])/len(concepts[key])

        for a,b in sorted(concepts.items(),key=lambda x:x[1],reverse=True):
            f.write('{0}\t{1:.2f}\n'.format(a,b))
        f.close()
        if verbose: print("[i] Wrote stats on concepts to file.")

    def analyze(
            self,
            runs = "default",
            verbose = False,
            output_gml = False,
            tar = False,
            plot_dists = False,
            usetex = True
            ):
        """
        Carry out a full analysis using various parameters.

        Parameters
        ----------
        runs : {str list} (default="default")
            Define a couple of different models to be analyzed.
        verbose : bool (default = False)
            If set to c{True}, be verbose when carrying out the analysis.
        usetex : bool (default=True)
            Specify whether you want to use LaTeX to render plots.
        """
        
        # define a default set of runs
        if runs == 'default':
            runs = [
                    #('weighted',(5,2)),
                    #('weighted',(79,40)),
                    #('weighted',(39,20)),
                    #('weighted',(3,1)),
                    #('weighted',(2,1)),
                    #('weighted',(1,1)),
                    #('weighted',(1,2)),
                    #('weighted',(1,3)),
                    #('weighted',(2,5)),
                    #('weighted',(2,3)),
                    #('restriction',1),
                    #('restriction',2),
                    ('restriction',3),
                    #('restriction',4),
                    #('restriction',5)
                    ]
        
        # carry out the various analyses
        for mode,params in runs:
            if mode == 'weighted':
                print(
                        "[i] Analysing dataset with mode {0} ".format(mode)+\
                                "and ratio {0[0]}:{0[1]}...".format(params)
                                )

                self.get_GLS(
                        mode = mode,
                        ratio = params,
                        verbose = verbose,
                        output_gml = output_gml,
                        tar = tar,
                        )
            elif mode == 'restriction':
                print(
                        "[i] Analysing dataset with mode {0} ".format(mode)+\
                                "and restriction {0}...".format(params)
                                )
                
                self.get_GLS(
                        mode = mode,
                        restriction = params,
                        verbose = verbose,
                        output_gml = output_gml,
                        tar = tar,
                        )
    
        # calculate the different distributions
        # start by calculating the contemporary distributions
        print("[i] Calculating the Contemporary Vocabulary Distributions...")
        self.get_CVSD(verbose=verbose)
        
    
        # now calculate the rest of the distributions
        print("[i] Calculating the Ancestral Vocabulary Distributions...")
        modes = list(self.gls.keys())
        for m in modes:
            self.get_AVSD(m,verbose=verbose)

        # compare the distributions using mannwhitneyu
        print("[i] Comparing the distributions...")
        
        zp_vsd = []
        for m in modes:
            vsd = sps.mannwhitneyu(
                    self.dists['contemporary'],
                    self.dists[m]
                    )

            zp_vsd.append(vsd)

        # write results to file
        print("[i] Writing stats to file.")
        f = open(self.dataset+'_trebor/'+self.dataset+'.stats','w')
        f.write("Mode\tANO\tMNO\tVSD_z\tVSD_p\n")
        for i in range(len(zp_vsd)):
            f.write(
                    '{0}\t{1:.2f}\t{2}\t{3}\n'.format(
                        modes[i],
                        self.stats[modes[i]]['ano'],
                        self.stats[modes[i]]['mno'],
                        '{0[0]}\t{0[1]:.4f}'.format(zp_vsd[i])
                        )
                    )
        f.close()

        # plot the stats if this is defined in the settings
        if plot_dists:

            # specify latex
            mpl.rc('text',usetex=usetex)
                        
            # store distributions in lists
            dists_vsd = [self.dists[m] for m in modes]
            
            # store contemporary dists
            dist_vsd = self.dists['contemporary']
            
            # get the average number of origins
            ano = [self.stats[m]['ano'] for m in modes]

            # create a sorter for the distributions
            sorter = [s[0] for s in sorted(
                zip(range(len(modes)),ano),
                key=lambda x:x[1]   
                )]

            # sort the stuff
            dists_vsd = [dists_vsd[i] for i in sorter]
            modes = [modes[i] for i in sorter]

            # sort the zp-values
            zp_vsd = [zp_vsd[i] for i in sorter]

            # format the zp-values
            if usetex:

                p_vsd = []
                for i,(z,p) in enumerate(zp_vsd):
                    if p < 0.001:
                        p_vsd.append('p$<${0:.2f}'.format(p))
                    elif p >= 0.05:
                        p_vsd.append(r'\textbf{{p$=${0:.2f}}}'.format(p))
                        # adjust the modes
                        modes[i] = r'\textbf{'+modes[i]+'}'
                    else:
                        p_vsd.append('p$=${0:.2f}'.format(p))
                
            else:
                p_vsd = []
                for z,p in zp_vsd:
                    if p < 0.001:
                        p_vsd.append('p<{0:.2f}'.format(p))
                    elif p >= 0.05:
                        p_vsd.append(r'{p={0:.2f}'.format(p))
                    else:
                        p_vsd.append('p={0:.2f}'.format(p))
            
            # create the figure
            fig = plt.figure()

            # create the axis
            ax = fig.add_subplot(111)

            # add the boxplots
            ax.boxplot([dist_vsd]+dists_vsd)

            # add the xticks
            plt.xticks(
                    range(1,len(modes)+2),
                    ['']+['{0}\n{1}'.format(
                        m,
                        p
                        ) for m,p in zip(modes,p_vsd)
                        ],
                    size=6
                    )

            # save the figure
            plt.savefig(self.dataset+'_trebor/vsd.pdf')
            plt.clf()
            
            print("[i] Plotted the distributions.")

    def plot_MLN(
            self,
            mode_string,
            verbose = False,
            filename = 'pdf',
            threshold = 1,
            fileformat = 'pdf',
            usetex = True
            ):
        """
        Plot the MLN with help of Matplotlib.
        """
        
        # get the graph
        graph = self.graph[mode_string]

        # store in internal and external nodes
        inodes = []
        enodes = []
        
        # get the nodes
        for n,d in graph.nodes(data=True):
            g = d['graphics']
            x = g['x']
            y = g['y']
            h = g['h']
            w = g['w']

            if d['label'] not in self.taxa:
                inodes += [(x,y)]
            else:
                if usetex:
                    enodes += [(x,y,r'\textbf{'+d['label']+r'}')]
                else:
                    enodes += [(x,y,d['label'])]
        
        # store vertical and lateral edges
        vedges = []
        ledges = []
        weights = []

        # get the edges
        for a,b,d in graph.edges(data=True):
            
            xA = graph.node[a]['graphics']['x']
            yA = graph.node[a]['graphics']['y']
            xB = graph.node[b]['graphics']['x']
            yB = graph.node[b]['graphics']['y']

            if d['label'] == 'vertical':

                vedges += [(xA,xB,yA,yB)]
            else:
                g = d['graphics']
                f = g['fill']
                w = g['width']
                if d['weight'] < threshold:
                    w = 0.0

                ledges += [(xA,xB,yA,yB,f,w)]

                weights.append(d['weight'])
        
        # usetex
        mpl.rc('text',usetex = usetex)

        # create the figure
        fig = plt.figure(facecolor='white')
        figsp = fig.add_subplot(111)
        
        # create the axis
        ax = plt.axes(frameon=False)
        plt.xticks([0],[''])
        plt.yticks([0],[''])
        
        # set equal axis
        plt.axis('equal')

        # draw the horizontal edges
        for xA,xB,yA,yB,f,w in ledges:
            plt.plot(
                    [xA,xB],
                    [yA,yB],
                    '-',
                    color=f,
                    linewidth=float(w) / 3,
                    alpha=0.75
                    )

        # draw the vertical edges
        for xA,xB,yA,yB in vedges:
            plt.plot(
                    [xA,xB],
                    [yA,yB],
                    '-',
                    color='0.0',
                    linewidth=5,
                    )
            plt.plot(
                    [xA,xB],
                    [yA,yB],
                    '-',
                    color='0.2',
                    linewidth=4,
                    )

        # draw the nodes
        for x,y in inodes:
            plt.plot(
                    x,
                    y,
                    'o',
                    markersize=10,
                    color='black',
                    )

        # draw the leaves
        for x,y,t in enodes:
            plt.text(
                    x,
                    y,
                    t,
                    size = '5',
                    verticalalignment='center',
                    backgroundcolor='black',
                    horizontalalignment='center',
                    color='white'
                    )

        # add a colorbar
        cax = figsp.imshow([[1,2],[1,2]],visible=False)
        cbar = fig.colorbar(
                cax,
                ticks = [
                    1,
                    1.25,
                    1.5,
                    1.75,
                    2
                    ],
                orientation='vertical',
                shrink=0.55
                )
        cbar.set_clim(1.0)
        cbar.set_label('Inferred Borrowings')
        cbar.ax.set_yticklabels(
                [
                    str(min(weights)),
                    '',
                    str(int(max(weights) / 2)),
                    '',
                    str(max(weights))
                    ]
                )

        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05)

        # save the figure
        plt.savefig(filename+'.'+fileformat)
        plt.clf()
        if verbose: FileWriteMessage(filename,fileformat).message('written')

        return 

    def geoplot_MLN(
            self,
            mode_string,
            verbose=False,
            filename='pdf',
            fileformat='pdf',
            mode = "basic",
            threshold = 1,
            only = None
            ):
        """
        Carry out a geographical plot of a given MLN.
        """
    
        # redefine taxa and tree for convenience
        taxa,tree = self.taxa,self.tree

        # get the graph
        graph = self.graph[mode_string]

        # XXX check for coordinates of the taxa, otherwise load them from file and
        # add them to the wordlist XXX add later, we first load it from file
        if 'coords' in self.wl.entries:
            pass
        
        else:
            coords = csv2dict(
                    self.dataset,
                    'coords',
                    dtype=[str,float,float]
                    )

        # check for groups, add functionality for groups in qlc-file later XXX
        if 'group' in self.wl.entries:
            pass
        else:
            groups = dict([(k,v) for k,v in csv2list(self.dataset,'groups')])
        # check for color, add functionality for colors later XXX
        if 'colors' in self.wl.entries:
            pass
        else:
            colors = dict([(k,v) for k,v in csv2list(self.dataset,'colors')])

        if verbose: LoadDataMessage('coordinates','groups','colors').message('loaded')
        
        # load the rc-file XXX add internal loading later
        try:
            conf = json.load(open(self.dataset+'.json'))
        except:
            pass # XXX add fallback later
        
        if verbose: LoadDataMessage('configuration')
                
        # calculate all resulting edges, using convex hull as
        # approximation 
        geoGraph = nx.Graph()
        
        for nA,nB,d in graph.edges(data=True):
            
            # get the labels
            lA = graph.node[nA]['label']
            lB = graph.node[nB]['label']
            
            # first check, whether edge is horizontal
            if d['label'] == 'horizontal':
                
                # if both labels occur in taxa, it is simple
                if lA in taxa and lB in taxa:
                    try:
                        geoGraph.edge[lA][lB]['weight'] += d['weight']
                    except:
                        geoGraph.add_edge(lA,lB,weight=d['weight'])
                
                # if only one in taxa, we need the convex hull for that node
                elif lA in taxa or lB in taxa:

                    # check which node is in taxa
                    if lA in taxa:
                        this_label = lA
                        other_nodes = tree.getNodeMatchingName(lB).getTipNames()
                        other_label = lB
                    elif lB in taxa:
                        this_label = lB
                        other_nodes = tree.getNodeMatchingName(lA).getTipNames()
                        other_label = lA

                    # get the convex points of others
                    these_coords = [(round(coords[t][0],5),round(coords[t][1],5)) for t in
                            other_nodes]
                    hulls = getConvexHull(these_coords,polygon=False)
    
                    # get the hull with the minimal euclidean distance
                    distances = []
                    for hull in hulls:
                        distances.append(linalg.norm(np.array(hull) - np.array(coords[this_label])))
                    this_hull = hulls[distances.index(min(distances))]
                    other_label = other_nodes[
                            these_coords.index(
                                (
                                    round(this_hull[0],5),
                                    round(this_hull[1],5)
                                    )
                                )
                            ]
    
                    # append the edge to the graph
                    try:
                        geoGraph.edge[this_label][other_label]['weight'] += d['weight']
                    except:
                        geoGraph.add_edge(this_label,other_label,weight=d['weight'])
                    
                else:
                    # get the taxa of a and b
                    taxA = tree.getNodeMatchingName(lA).getTipNames()
                    taxB = tree.getNodeMatchingName(lB).getTipNames()
    
                    # get the convex points
                    coordsA = [(round(coords[t][0],5),round(coords[t][1],5)) for t in taxA]
                    coordsB = [(round(coords[t][0],5),round(coords[t][1],5)) for t in taxB]
                    hullsA = getConvexHull(coordsA,polygon=False)
                    hullsB = getConvexHull(coordsB,polygon=False)
    
                    # get the closest points
                    distances = []
                    hulls = []
                    for i,hullA in enumerate(hullsA):
                        for j,hullB in enumerate(hullsB):
                            distances.append(linalg.norm(np.array(hullA)-np.array(hullB)))
                            hulls.append((hullA,hullB))
                    minHulls = hulls[distances.index(min(distances))]
                    
                    labelA = taxA[coordsA.index((round(minHulls[0][0],5),round(minHulls[0][1],5)))]
                    labelB = taxB[coordsB.index((round(minHulls[1][0],5),round(minHulls[1][1],5)))]
                    
                    # append the edge to the graph
                    try:
                        geoGraph.edge[labelA][labelB]['weight'] += d['weight']
                    except:
                        geoGraph.add_edge(labelA,labelB,weight=d['weight'])

        # get the weights for the lines
        weights = []
        for a,b,d in geoGraph.edges(data=True):
            weights += [d['weight']]
        max_weight = max(weights)
        scale = 256 / max_weight
        sorted_weights = sorted(set(weights))

        # get a color-function
        color_dict = np.array(
                np.linspace(
                    0,
                    256,
                    len(set(weights))
                    ),
                dtype='int'
                )

        # get a line-function
        line_dict = np.linspace(
                0.5,
                conf['linewidth'],
                len(set(weights))
                )

        # scale the weights for line-widths
        linescale = conf['linescale'] / (max_weight-threshold) #XXX
        # XXX apparently not needed?
        
        # determine the maxima of the coordinates
        latitudes = [i[0] for i in coords.values()]
        longitudes = [i[1] for i in coords.values()]

        min_lat,max_lat = min(latitudes),max(latitudes)
        min_lon,max_lon = min(longitudes),max(longitudes)

        # start to initialize the basemap
        fig = plt.figure()
        figsp = fig.add_subplot(111)
        
        # instantiate the basemap
        m = bmp.Basemap(
            llcrnrlon=min_lon + conf['min_lon'],
            llcrnrlat=min_lat + conf['min_lat'],
            urcrnrlon=max_lon + conf['max_lon'],
            urcrnrlat=max_lat + conf['max_lat'],
            resolution=conf['resolution'],
            projection=conf['projection']
            )
        
        # draw first values
        m.drawmapboundary(fill_color=conf['water_color'])
        m.drawcoastlines(color=conf['continent_color'],linewidth=0.5)
        m.drawcountries(color=conf['coastline_color'],linewidth=0.5)
        m.fillcontinents(color=conf['continent_color'],lake_color=conf['water_color'])

        # plot the lines
        for a,b,d in geoGraph.edges(data=True):
            
            # don't draw lines beyond threshold
            if d['weight'] < threshold:
                pass
            else:
                if a in coords and b in coords and only in [a,b,None]:
                    w = d['weight']

                    # retrieve the coords
                    yA,xA = coords[a]
                    yB,xB = coords[b]
                    
                    # get the points on the map
                    xA,yA = m(xA,yA)
                    xB,yB = m(xB,yB)

                    # plot the points
                    plt.plot(
                            [xA,xB],
                            [yA,yB],
                            '-',
                            color=plt.cm.jet(
                                color_dict[sorted_weights.index(w)]
                                ),
                            alpha = conf['alpha'],
                            linewidth=line_dict[sorted_weights.index(w)],
                            zorder = w + 50
                            )

        # plot the points for the languages
        cell_text = []
        legend_check = []
        for i,(taxon,(lng,lat)) in enumerate(sorted(coords.items(),key=lambda x:x[0])):
            
            # retrieve x and y from the map
            x,y = m(lat,lng)
            
            # get the color of the given taxon
            taxon_color = colors[groups[taxon]]
            
            # check for legend

            if groups[taxon] in legend_check:
                # plot the marker
                plt.plot(
                    x,
                    y,
                    'o',
                    markersize = conf['markersize'],
                    color = taxon_color,
                    zorder = max_weight+52,
                    )
            else:
                # plot the marker
                plt.plot(
                    x,
                    y,
                    'o',
                    markersize = conf['markersize'],
                    color = taxon_color,
                    zorder = max_weight+52,
                    label=groups[taxon]
                    )
                legend_check.append(groups[taxon])
            
            # add number to celltext
            cell_text.append([str(i+1),taxon])

            # plot the text
            plt.text(
                x,
                y,
                str(i+1),
                size = str(int(conf['markersize'] / 2)),
                label=taxon,
                horizontalalignment='center',
                verticalalignment='center',
                zorder=max_weight+55
                )

        # add a colorbar
        cax = figsp.imshow([[1,2],[1,2]],visible=False)
        cbar = fig.colorbar(
                cax,
                ticks = [
                    1,
                    1.25,
                    1.5,
                    1.75,
                    2
                    ],
                orientation='vertical',
                shrink=0.55
                )
        cbar.set_clim(1.0)
        cbar.set_label('Inferred Borrowings')
        cbar.ax.set_yticklabels(
                [
                    str(min(weights)),
                    '',
                    str(int(max(weights) / 2)),
                    '',
                    str(max(weights))
                    ]
                )

        # add the legend
        this_table = plt.table(
                cellText = cell_text,
                colWidths = conf['table.column.width'],
                loc = conf['table.location'],
                )

        # adjust the table
        for line in this_table._cells:
            this_table._cells[line]._text._horizontalalignment = 'left'
            this_table._cells[line]._text._fontproperties.set_weight('bold')
            this_table._cells[line]._text.set_color(conf['table.text.color'])
            this_table._cells[line].set_height(conf['table.cell.height'])
            this_table._cells[line]._text._fontproperties.set_size(conf['table.text.size'])
            this_table._cells[line].set_linewidth(0.0)
            this_table._cells[line].set_color(conf['table.cell.color'])
        
        this_table.set_zorder(100)
        
        plt.legend(
                loc=conf['legend.location'],
                numpoints=1,
                prop={
                    'size':conf['legend.size'],
                    'weight':'bold'
                    }
                )

        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05)

        plt.savefig(filename+'.'+fileformat)
        plt.clf()
        if verbose: FileWriteMessage(filename,fileformat).message('written')
        return geoGraph



