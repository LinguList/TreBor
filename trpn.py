#! /usr/bin/env python
# *-* coding:utf-8 *-*
import cogent as cg
from lingpyd.data.csv import loadCSV,loadDict
from lingpyd import LexStat,patchy_alm2html
import matplotlib as mpl
from sys import argv
import networkx as nx
import os
import numpy as np
from numpy import zeros,array
from scipy.stats import wilcoxon,ttest_ind,mannwhitneyu
from glob import glob
import matplotlib.pyplot as plt

def _make_gls(
        taxa,
        pap,
        backbone,
        ratio=(1,1),
        debug=False,
        ):

    """
    Calculate an evolutionary gain-loss-scenario from a given tree and a list
    of paps.
    """

    # make a dictionary that stores the scenario
    d = {}

    # get the subtree containing all taxa that have positive paps
    tree = backbone.lowestCommonAncestor(
            [taxa[i] for i in range(len(taxa)) if pap[i] >= 1]
            )

    # assign the basic (starting) values to the dictionary
    nodes = [x.Name for x in tree.tips()]

    # get the first state of all nodes and store the state in the dictionary.
    # note that we start from two distinct scenarios: one assuming single
    # origin at the root, where all present states in the leave are treated as
    # retentions, and one assuming multiple origins, where all present states
    # in the leaves are treated as origins
    for node in nodes:
        idx = taxa.index(node)
        if pap[idx] >= 1:
            state = 1
        else:
            state = 0
        d[node] = [(state,[])]

    # return simple scenario if the group is single-origin
    if sum([d[node][0][0] for node in nodes]) == len(nodes):
        return [[(tree.Name,1)]]

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
        if debug: print node.Name
        
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
            if debug: print "nodelen",len(d[node.Name])
    
    # try to find the best scenario by counting the ratio of gains and losses.
    # the key idea here is to reduce the number of possible scenarios according
    # to a given criterion. We choose the criterion of minimal changes as a
    # first criterion to reduce the possibilities, i.e. we weight both gains
    # and losses by 1 and select only those scenarios where gains and losses
    # sum up to a minimal number of gains and losses. This pre-selection of
    # scenarios can be further reduced by weighting gains and losses
    # differently. So in a second stage we choose only those scenarios where
    # there is a minimal amount of gains. 
    
    if debug: print len(d[tree.Name])

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

    # return the minimal indices
    return [gls_list[i] for i in range(len(tracer)) if tracer[i] == minScore]

def _make_gls2(
        taxa,
        pap,
        backbone,
        maximal_gains = 4,
        debug=False,
        ):

    """
    Calculate an evolutionary gain-loss-scenario from a given tree and a list
    of paps. This method follows the idea of Dagan & Martin (2007).
    """

    # make a dictionary that stores the scenario
    d = {}

    # get the subtree containing all taxa that have positive paps
    tree = backbone.lowestCommonAncestor(
            [taxa[i] for i in range(len(taxa)) if pap[i] >= 1]
            )

    # assign the basic (starting) values to the dictionary
    nodes = [x.Name for x in tree.tips()]

    # get the first state of all nodes and store the state in the dictionary.
    # note that we start from two distinct scenarios: one assuming single
    # origin at the root, where all present states in the leave are treated as
    # retentions, and one assuming multiple origins, where all present states
    # in the leaves are treated as origins
    for node in nodes:
        idx = taxa.index(node)
        if pap[idx] >= 1:
            state = 1
            # assign first and second scenario
            #d[node] = [
            #    (1,[]), # first scenario, origin in the leaves
            #    ]
            ## assign second
        else:
            state = 0
        d[node] = [(state,[])]

    # return simple scenario if the group is single-origin
    if sum([d[node][0][0] for node in nodes]) == len(nodes):
        return [[(tree.Name,1)]]

    # order the internal nodes according to the number of their leaves
    ordered_nodes = sorted(
            tree.nontips()+[tree],key=lambda x:len(x.tips())
            )

    # join the nodes successively
    for i,node in enumerate(ordered_nodes):
        if debug: print node
        
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
                    if len([k for k in tmp if k[1] == 1]) + nodeA[0] <= maximal_gains:
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
                    if gainsA <= maximal_gains and not noA:
                        newNodes += [newNodeA]
                    if gainsB <= maximal_gains and not noB:
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
    
    if debug: print len(d[tree.Name])

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
    return [gls_list[i] for i in range(len(tracer)) if tracer[i] == minScore]

def gls2gml(
        gls,
        graph,
        tree
        ):
    """
    Create GML-representation of a given GLS.
    """
    
    # create a mapper for the ids and the string-names
    mapper = {}
    for node,data in graph.nodes(data=True):
        mapper[data['label']] = node

    # create a graph
    g = nx.Graph()

    # sort the gls according to the number of tips
    gls_srt = sorted(
            gls,
            key=lambda x:len(tree.getNodeMatchingName(x[0]).tips()),
            reverse=True
            )

    # set the basic event frame, depending on the state of the root
    if gls_srt[0][1] == 1 and gls_srt[0][0] == 'root':
        this_color = "#ffffff"
    else:
        this_color = "#000000"

    # let all nodes inherit these parameters
    for node,data in graph.nodes(data=True):
        data['graphics']['fill'] = this_color
        data['graphics']['type'] = 'ellipse'
        data['graphics']['w'] = 20.0
        data['graphics']['h'] = 20.0
        g.add_node(node,**data)

    # assign the root as starting point
    data = graph.node[mapper['root']]
    data['graphics']['type'] = 'ellipse'
    data['graphics']['w'] = 50.0
    data['graphics']['h'] = 50.0
    g.add_node(mapper['root'],**data)

    # iterate over the nodes involved in change and assign the values to their
    # children
    for name,event in gls_srt:
        if event == 1:
            this_fill = '#ffffff'
        else:
            this_fill = '#000000'

        # get the names of the descendant nodes in the subtree 
        sub_tree_nodes = backbone.getNodeMatchingName(name).getNodeNames()

        # iterate over all nodes to change
        for node in sub_tree_nodes:
            data = g.node[mapper[node]]
            data['graphics']['fill'] = this_fill
            g.add_node(mapper[node],**data)

        # change the size of the root of the subtree
        g.node[mapper[name]]['graphics']['h'] = 50.0
        g.node[mapper[name]]['graphics']['w'] = 50.0
        g.node[mapper[name]]['graphics']['fill'] = this_fill

    # add the edges to the tree
    for edgeA,edgeB,data in graph.edges(data=True):
        # for computers with new networkx version
        try:
            del data['graphics']['Line']
        except:
            pass
        if 'label' not in data:
            g.add_edge(edgeA,edgeB,**data)

    return g

def tree2graph(tree):
    """
    Function converts a cogent-tree into a networkx graph.
    """
    
    # create an empty graph
    graph = nx.DiGraph()

    # get the node names of the tree
    nodes = tree.getNodeNames()

    # iterate over the nodes and add them and the edges to the graph
    for node in nodes:
        
        # add the node (just as a precaution)
        graph.add_node(node)

        # get the parent of the node
        parent = tree.getNodeMatchingName(node).Parent

        # add the edge if the parent is not None
        if parent:
            graph.add_edge(parent.Name,node)

    return graph

def make_gls(
        taxa,
        paps,
        backbone,
        ratio = (1,1),
        maximal_gains = None,
        filename = "out",
        output_gml = False,
        verbose = False
        ):
    """
    Create GLS from a set of paps for a given set of taxa and a given reference
    tree.
    """
    
    # create a list of gls
    scenarios = []

    # verbose output, count number of multiple scenarios
    if verbose: ms = 0
    
    # create the gls and append it to the list
    for i,pap in enumerate(paps):
        
        # verbose output
        if verbose: print "[i] Calculating GLS for PAP {0}, line {1}...".format(pap[0],i+1)
        
        # check for method
        if ratio:
            scenario = _make_gls(
                            taxa,
                            pap[1:],
                            backbone,
                            ratio
                            )

            # verbose output: number of multiple scenarios
            if verbose: ms += len(scenario)
            
            # append the first of the scenarios to the list
            scenarios.append((pap[0],scenario[0]))

        else:
            scenario = _make_gls2(
                            taxa,
                            pap[1:],
                            backbone,
                            maximal_gains=maximal_gains
                            )

            # verbose output: number of multiple scenarios
            if verbose: ms += len(scenario)
            
            # append the first of the scenarios to the list
            scenarios.append((pap[0],scenario[0]))
        
        # verbose output
        if verbose: print "... done."
    
    # write the results to file
    # verbose output
    if verbose: print "[i] Writing results to file..."

    # check whether there is a project-folder
    folder = filename+'_trpn'
    try:
        os.mkdir(folder)
    except:
        pass

    # make gls-folder inside folder
    try:
        os.mkdir(folder+'/gls')
    except:
        pass
    
    # different file names for the different methods
    if ratio:
        out = open(folder+'/gls/'+filename+'_{0[0]}_{0[1]}.gls'.format(ratio),'w')
    else:
        out = open(folder+'/gls/'+filename+'_{0}.gls'.format(maximal_gains),'w')
    
    # iterate over papId and gls
    for papId,gls in scenarios:
        out.write(papId+'\t'+','.join(['{0}:{1}'.format(
            name,
            event
            ) for name,event in gls])+'\n')
    out.close()
    print "... done."
    
    # write the matchings between nodes and names of nodes to file
    out = open(folder+'/'+filename+'_node_names','w')
    for node in backbone.getNodeNames():
        out.write('{0}\t{1}\n'.format(node,str(backbone.getNodeMatchingName(node))))
    out.close()

    # if output_gml is chosen
    if output_gml:
        try:
            os.mkdir(folder+'/gml')
        except:
            pass

        if ratio:
            new_folder = folder+'/gml/'+filename+'_gml'+'_{0[0]}_{0[1]}'.format(ratio)
        else:
            new_folder = folder+'/gml/'+filename+'_gml'+'_{0}'.format(maximal_gains)

        try:
            os.mkdir(new_folder)
        except:
            pass

        # check whether there is a gml-file, if not, create it
        try:
            graph = nx.read_gml(filename+'.gml')
        except:
            graph = tree2graph(backbone)
            nx.write_gml(graph,filename+'.gml')

        for papId,gls in scenarios:
            
            # verbose output
            if verbose: print "[i] Writing GML graph for PAP {0}...".format(papId)

            # create the graph
            this_graph = gls2gml(
                    gls,
                    graph,
                    backbone
                    )

            ## write the graph to file
            nx.write_gml(this_graph,new_folder+'/'+filename+'_'+papId+'.gml')

            # verbose output
            if verbose: print "... done."
    
    # verbose output, write multiple scenarios
    if verbose: print "[i] Average of Multiple Scenarios: {0:.2f}".format(
            ms / float(len(paps))
            )

def lexical_distribution(
        taxa,
        paps,
        mode="dist"
        ):
    """
    Calculate the vocabulary distribution for a given set of paps.
    """

    # get the pap-ids
    papIds = [line[0] for line in paps]

    # get the concepts
    concepts = sorted(set([papId.split('.')[1] for papId in papIds]))

    # create a numpy-array for the taxa
    dists = zeros((len(taxa),len(concepts)))

    # iterate over all taxa and calculate the number of cogs for each concept
    
    #dists = {}
    for i,taxon in enumerate(taxa):
        
        # get the index
        idx = taxa.index(taxon)+1
        
        # zip the papIds with the specific paps for the taxon
        these_paps = zip(papIds,[line[idx] for line in paps])

        # get the attested paps
        attested_paps = [
                papId.split('.')[1] for papId,pap in these_paps if pap > 0
                ]

        # calculate specific distribution for the given taxon
        for j,c in enumerate(concepts):
            dists[i][j] = attested_paps.count(c)

    # calculate the distribution
    dist = [d.mean() for d in dists]
    
    # calculate vocabulary size
    size = [sum([p[i] for p in paps]) for i in range(1,len(taxa))]
    
    if mode == 'dist':
        return dist
    else:
        return size

def get_paps_for_ancestors(
        taxa,
        backbone,
        scenarios
        ):
    """
    Function retrieves all paps for ancestor languages in a given tree.
    """
    
    # get all internal nodes
    nodes = sorted(
            [node.Name for node in backbone.nontips()],
            key=lambda x: len(backbone.getNodeMatchingName(x).tips()),
            reverse = True
            )
    
    # store scenarios in a list
    events = {}
    papIds = []
    data = loadCSV(scenarios)
    for line in data:
        events[line[0]] = [(a,int(b)) for a,b in [i.split(':') for i in line[1].split(',')]]
        papIds.append(line[0])

    # calculate average number of origins
    origins = []
    for event in events:
        origins.append(len([1 for i in events[event] if i[1] == 1]))

    # create an list storing the paps
    paps = [[0 for i in range(len(nodes) + 1)] for j in range(len(papIds))]

    # add papIds to first column of paps
    for i,papId in enumerate(papIds):
        paps[i][0] = papId

    # iterate over papIds and nodes and assign the values to the paps
    for i,papId in enumerate(papIds):
        
        # sort the gls corresponding to the papId
        gls = sorted(
                events[papId],
                key = lambda x: len(backbone.getNodeMatchingName(x[0]).tips()),
                reverse = True
                )

        # retrieve the state of the root
        if gls[0][1] == 1 and gls[0][0] == 'root':
            state = 1
        else:
            state = 0

        # assign the state of the root to all nodes
        paps[i] = [papId] + [state for node in nodes]

        # iterate over the gls and assign the respective values to all children
        for name,event in gls:
            if event == 1:
                this_state = 1
            else:
                this_state = 0

            # get the subtree nodes
            sub_tree_nodes = [node.Name for node in backbone.getNodeMatchingName(name).nontips()]

            # assign this_state to all subtree nodes
            for node in sub_tree_nodes:
                paps[i][nodes.index(node)+1] = this_state

    return nodes,paps,sum(origins) / float(len(origins)),max(origins)

def compare_distributions(
        distA,
        distB,
        ):
    """
    Compare two distributions using Mann-Whitney test.
    """
    
    m = mannwhitneyu(distA,distB)
    
    return m


def plot_distributions(
        taxa,
        paps,
        backbone,
        filename,
        mode = 'dist',
        verbose = False
        ):
    """
    Function compares all distributions corresponding to a certain filename and
    plots the results to file.
    """
    
    # create name for project folder
    folder = filename+'_trpn'

    # get the first distribution for the contemporary languages
    cnt_dist = lexical_distribution(taxa,paps,mode)

    # get all other distributions
    files = sorted(
            glob(folder+'/gls/'+filename+'_*.gls')
            )
    dists = []
    zp = []
    oris = []
    
    # open outfile
    out = open(folder+'/'+filename+'_'+mode+'.stats','w')

    for f in files:
        
        # get ancestral paps
        anc_taxa,anc_paps,ave_ori,max_ori = get_paps_for_ancestors(
                taxa,
                backbone,
                f
                )

        # get ancestral distribution
        anc_dist = lexical_distribution(anc_taxa,anc_paps,mode)

        # carry out test
        z,p = compare_distributions(cnt_dist,anc_dist)

        # store results in lists
        dists.append(anc_dist)
        zp.append((z,p))
        oris.append(ave_ori)

        # write results to file
        out.write(
                '{0}\t{1:.2f}\t{2}\t{3:.2f}\t{4:.2f}\n'.format(
                    f.strip('.gls'),
                    ave_ori,
                    max_ori,
                    z,
                    p
                    )
                )
        # print result if verbosity is chosen
        if verbose: print '{0}\t{1:.2f}\t{2}\t{3:.2f}\t{4:.2f}'.format(    
                f.strip('.gls'),
                ave_ori,
                max_ori,
                z,
                p
                )

    # plot the distributions
    # create the figure
    fig = plt.figure()

    # create the axis
    ax = fig.add_subplot(111)
    
    # sort the stuff in a meaningful way
    sorter = zip(range(len(dists)),oris)
    sorter = sorted(
            sorter,
            key=lambda x:x[1]
            )
    sorter = [i[0] for i in sorter]

    dists = [dists[i] for i in sorter]
    files = [files[i] for i in sorter]

    # format the zps
    pvals = []
    zps = [zp[i] for i in sorter]
    for z,p in zps:
        if p < 0.001:
            pvals.append('p<{0:.2f}'.format(p))
        else:
            pvals.append('p={0:.2f}'.format(p))

    # add the boxplots
    ax.boxplot([cnt_dist]+dists)

    # add the xticks
    plt.xticks(
            range(1,len(files)+2),
            ['']+['M_{0}\n{1}'.format(
                x.replace(
                    folder+'/gls/'+filename+'_',
                    ''
                    ).replace('.gls',''),
                y
                ) for x,y in
                zip(files,pvals)],
            size=6
            )

    # make x-dates autoformat
    #fig.autofmt_xdate()

    plt.savefig(folder+'/'+filename+'_'+mode+'.pdf')

def load_gls(dataset,model):
    """
    Function returns a dictionary of gain-loss-scenarios.
    """
    
    scenarios = {}
    data = loadCSV(dataset+'_'+model+'.gls')
    
    for line in data:
        scenarios[line[0]] = [(a,int(b)) for a,b in [i.split(':') for i in line[1].split(',')]]

    return scenarios

def load_ile(
        dataset,
        model,
        mode
        ):
    """
    Function returns a dictionary of gain-loss-scenarios.
    """
    
    scenarios = {}
    data = loadCSV(dataset+'_'+model+'_'+mode+'.ile')
    
    for line in data:
        scenarios[line[0]] = [(a,b) for a,b in [i.split(':') for i in line[1].split(',')]]

    return scenarios

def get_evolutionary_network(
        dataset,
        model,
        verbose=False,
        threshold=1,
        mode='mst'
        ):
    """
    Compute a primary network (graph) for a given list of evolutionary
    scenarios. 

    Scenarios are stored in the file filename, as output by the method
    make_gls.
    """

    # make the name for the project folder
    folder = dataset + '_trpn'

    # load the gls
    scenarios = load_gls(folder+'/gls/'+dataset,model)

    # create the primary graph
    gPrm = nx.Graph()

    # load the template graph
    gTpl = nx.read_gml(dataset+'.gml')

    # load the backbone
    backbone = cg.LoadTree(dataset+'.tre')

    # define the taxa
    taxa = [node.Name for node in backbone.tips()]

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
    for papId,gls in scenarios.items():
        
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
    if verbose: print "[i] Calculated primary graph."
    
    # verbose output
    if verbose: print "[i] Inferring lateral edges..."

    # iterate over the scenarios
    ile = {} # stores the inferred lateral events

    if mode == 'mst':
        
        # create MST graph
        gMST = nx.Graph()

        for papId,gls in scenarios.items():
            
            ile[papId] = []

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
                        gMST.edge[nodeA][nodeB]['cogs'] += [papId]
                    except:
                        gMST.add_edge(
                                nodeA,
                                nodeB,
                                weight=1,
                                cogs=[papId]
                                )
                    ile[papId]+= [(nodeA,nodeB)]

    # if directed edges are chosen
    elif mode == 'directed':
        
        # create MST digraph
        gMST = nx.Graph()

        for papId,gls in scenarios.items():
            ile[papId] = []

            # get the origins
            oris = [x[0] for x in gls if x[1] == 1]

            # create dictionary of weighted degrees
            wdeg = {}

            # iterate over nodes
            for i,nodeA in enumerate(oris):
                for j,nodeB in enumerate(oris):
                    if i < j:
                        try:
                            wdeg[nodeA] += gPrm.edge[nodeA][nodeB]['weight']
                        except:
                            wdeg[nodeA] = gPrm.edge[nodeA][nodeB]['weight']
                        try:
                            wdeg[nodeB] += gPrm.edge[nodeA][nodeB]['weight']
                        except:
                            wdeg[nodeB] = gPrm.edge[nodeA][nodeB]['weight']

            # get the central node by sorting the dictionary 
            wdeg = sorted(
                    wdeg.keys(),
                    key = lambda x:wdeg[x]
                    )
            if wdeg:
                central_node = wdeg[-1]

                # draw connections
                for ori in [x for x in oris if x != central_node]:
                    try:
                        gMST.edge[central_node][ori]['weight'] += 1
                        gMST.edge[central_node][ori]['cogs'] += [papId]
                    except:
                        gMST.add_edge(central_node,ori,weight=1)
                        gMST.add_edge(central_node,ori,cogs=[papId])
                    
                    ile[papId]+= [(central_node,ori)]


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

    # append data to output graph
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
    if verbose: print "[i] Writing graph to file..."
    # write the graph to file
    nx.write_gml(gOut,folder+'/'+dataset+'_'+model+'_'+mode+'_'+str(threshold)+'.gml')

    # write the inferred borrowing events (ILS, inferred lateral event) 
    # between all taxa to file
    # verbose output
    if verbose: print "[i] Writing inferred lateral events to file..."
    out = open(folder+'/'+dataset+'_'+model+'_'+mode+'.ile','w')
    for papId,events in ile.items():
        if events:
            out.write(
                    papId+'\t'+','.join(
                        ['{0}:{1}'.format(a,b) for a,b in events]
                        )+'\n'
                    )
    out.close()

    # create file name for node labels (cytoscape output)
    out = open(folder+'/'+dataset+'.node.label.NA','w')
    out.write("node.label (class=java.lang.String)\n")
    for taxon in taxa:
        out.write('{0} = {1}\n'.format(taxon,taxon))
    out.close()


    # return the outgraph
    return gOut

def ile2lxs(
        dataset,
        model,
        mode
        ):
    """
    Write all lateral links for all taxa to the LexStat-csv-file.

    Consider to simply use a file-format in which lateral links are inserted
    into the original csv-files. Given that we "can" somehow retrieve the
    direction, we may assign it simply with help of the algorithm and color
    borrowings differently for the HTML-output of LingPy. 
    """

    # define folder
    folder = dataset+'_trpn'

    # load ile-file
    ile = load_ile(folder+'/'+dataset,model,mode)

    # load the etd-file
    etd = loadDict(dataset+'.pap.ids')

    # change ids to integers
    for key,value in etd.items():
        for i,val in enumerate(value):
            try:
                etd[key][i] = int(val)
            except:
                pass

    # get the backbone
    backbone = cg.LoadTree(dataset+'.tre')
    
    # create the index for the language entries by getting the taxa
    taxa = etd['CogID']
    
    # load the presumed borrowings into the CSV file
    lex = LexStat(dataset+'.csv')

    # iterate over the ile ids
    for key,value in ile.items():

        # iterate over source and target 
        for source,target in value:
            
            # get the values for the target
            if target in taxa:
                target_id = etd[key][taxa.index(target)]
                target_ids = [target_id]
            # if the target is an internal node, get all tips
            else:
                targets = [node.Name for node in
                        backbone.getNodeMatchingName(target).tips()]
                target_ids = []
                for t in targets:
                    tmp = etd[key][taxa.index(t)]
                    if tmp != 0:
                        target_ids.append(tmp)

            # change the cognate ids in lex.dataById
            for idx in target_ids:
                lex.dataById[idx][lex._ifs['cogs'][0]] = \
                        '-'+lex.dataById[idx][lex._ifs['cogs'][0]]
 
    # write data to file
    lex.output('csv',filename=dataset+'_ile')

def gls2alm(
        dataset,
        model,
        verbose = True
        ):
    """
    Convert a given GLS to a file in ALM format where patchy distributions are
    additionally coded.
    """

    # load the etd-file
    etd = loadDict(dataset+'.pap.ids')

    # change ids to integers
    for key,value in etd.items():
        for i,val in enumerate(value):
            try:
                etd[key][i] = int(val)
            except:
                pass

    # load the gls-file
    gls = load_gls(dataset+'_trpn/gls/'+dataset,model)

    # get the backbone
    backbone = cg.LoadTree(dataset+'.tre')
    
    # create the index for the language entries by getting the taxa
    taxa = etd['CogID']
    
    # load the lexstat file
    lex = LexStat(dataset+'.csv',verbose=True)

    # add a new line to lexstat that stores the new cog-ids
    for key in [k for k in lex.dataById.keys() if k != 0]:

        data = lex.dataById[key].tolist()
        data.append('-')
        lex.dataById[key] = array(data)

    # change the identifier for dataById
    lex._ifs['origin'] = (len(lex._ifs),'Origin')

    # change first line
    lex.dataById[0] = array(
            lex.dataById[0].tolist() + ['Origin']
            )

    # iterate over the ile ids
    for key,values in gls.items():
        
        i = 1

        # iterate over source and target 
        for taxon,event in values:
            
            # check for gains
            if event == 1:

                # check for taxon/internal node
                if taxon in taxa:
                    
                    # get the word for the taxon
                    idx = etd[key][taxa.index(taxon)]

                    # add the word to lexstat
                    lex.dataById[idx][-1] = '{0}'.format(i)

                    # increase the counter
                    i += 1
                
                # if internal node, get all children
                else:
                    # get all leaves
                    nodes = [node.Name for node in
                            backbone.getNodeMatchingName(taxon).tips()]
                    
                    # get the indices for the leaves
                    indices = [etd[key][taxa.index(node)] for node in nodes]

                    # add the words to lexstat
                    for idx in indices:
                        if idx != 0:
                            lex.dataById[idx][-1] = '{0}'.format(i)

                    # increase the counter
                    i += 1
    
    if verbose: print "[i] Writing data to CSV file..."
    lex.output('csv',filename=dataset+'_patchy')

    if verbose: print "[i] Writing data to ALM file..."
    lex.output('alm.patchy',filename=dataset)

    return lex

if __name__ == "__main__":

    if len(argv) == 1:
        print "[i] No arguments specified!"

    else:

        # load the dataset 
        try:
            data = loadCSV(argv[1]+'.paps')
        except:
            lex = LexStat(argv[1]+'.csv')
            lex.output('paps')
            data = loadCSV(argv[1]+'.paps')

        # get the taxa in the first line of the paps-file
        taxa = data[0][1:]

        # get the paps
        tmp_paps = [[line[0]] + [int(x) for x in line[1:]] for line in data[1:]]
        paps = []
        for line in tmp_paps:
            if len([x for x in line[1:] if x > 0]) > 1:
                paps.append(line)

        # load the backbone
        check_tree= [k for k in argv if k.startswith('tree=')]
        if not check_tree:
            try:
                backbone = cg.LoadTree(argv[1]+'.tre')
            except:
                lex = LexStat(argv[1]+'.csv')
                lex.output('tre')
                backbone = cg.LoadTree(argv[1]+'.tre')
        else:
            backbone=cg.LoadTree(check_tree[0]+'.tre')

        # make the stuff bifurcating
        #backbone = backbone.bifurcating()
        
        # load the etymological dictionary file
        try:
            etm = loadDict(argv[1]+'.pap.ids')
        except:
            print "[i] loading lexstat"
            lex = LexStat(argv[1]+'.csv')
            lex.output('pap.ids')
            etm = loadDict(argv[1]+'.pap.ids')
            


        # get a mapper for backbone edgenames and taxa contained
        mapper = dict(
            [(i.Name,str(i)) for i in backbone.nontips()]
            )
    
    if 'check' in argv:    
        plot_distributions(
                taxa,
                paps,
                backbone,
                argv[1],
                verbose=True,
                mode = argv[3]
                )

    elif "dagan" in argv:
       make_gls(
               taxa,
               paps,
               backbone,
               maximal_gains=int(argv[2]),
               verbose=True,
               output_gml=False,
               filename=argv[1],
               ratio = None
               )
    
    elif "network" in argv or "nw" in argv:
        get_evolutionary_network(
                argv[1],
                argv[3],
                verbose=True,
                threshold=int(argv[4]),
                mode=argv[5]
                )
    elif "ile" in argv:
        ile2lxs(
                argv[1],
                argv[3],
                argv[4]
                )

    elif "gls" in argv:
        gls2alm(
                argv[1],
                argv[3]
                )
        patchy_alm2html(argv[1])
    
    elif len(argv) > 2:
        make_gls(
                taxa,
                paps,
                backbone,
                ratio = (int(argv[2]),int(argv[3])),
                filename=argv[1],
                verbose=True,
                output_gml=False
                )

