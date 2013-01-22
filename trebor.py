# created: Mo 21 Jan 2013 12:58:47  CET
# modified: Mo 21 Jan 2013 12:58:47  CET

__author__ = "Johann-Mattis List"
__date__ = "2013-01-21"

"""
Tree-based detection of borrowings in lexicostatistical wordlists.
"""

# basic imports
import os

# thirdparty imports
import numpy as np
import networkx as nx
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib as mpl

# lingpy imports
from lingpy.thirdparty import cogent as cg
from lingpy.convert.gml import *
from lingpy.basic import Wordlist


class TreBor(object):
    """
    Basic class for calculations using the TreBor method.
    """
    
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

        # check for paps as attribute in the wordlist
        if paps not in self.wl.entries:
            
            # define the function for conversion
            f = lambda x,y: "{0}:{1}".format(x[y[0]],x[y[1]])
            self.wl.add_entries(
                    paps,
                    'cogid,concept',
                    f
                    )
            if verbose: print("[i] Created entry PAP (CogID:Concept).")
        
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
        #nx.write_gml(gOut,self.dataset+'_trebor/mln-'+mode_string+'.gml')

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

        return 

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
                    ('weighted',(5,1)),
                    ('weighted',(4,1)),
                    ('weighted',(3,1)),
                    ('weighted',(2,1)),
                    ('weighted',(1,1)),
                    #('weighted',(1,2)),
                    #('weighted',(1,3)),
                    #('weighted',(1,4)),
                    #('weighted',(1,5)),
                    #('restriction',1),
                    #('restriction',2),
                    #('restriction',3),
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
        
        zp_fmd,zp_vsd = [],[]
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
        for i in range(len(zp_fmd)):
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

        







