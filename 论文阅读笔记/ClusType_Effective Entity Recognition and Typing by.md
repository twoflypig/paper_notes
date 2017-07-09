# 1.Intruduction
## traditional limits
Traditional named
entity recognition systems [18, 15] are usually designed for
several major types (e.g., person, organization, location) and
general domains (e.g., news), and so require additional steps
for adaptation to a new domain and new types.

## mainly two ways 
- weak supervision
- distant supervision

**Distant supervision** is a more recent trend, aiming to reduce expensive
human labor by utilizing entity information in KBs.

workflows:

1.detect entity mentions from a corpus

2.map candidate mentions to KB entities of target types

3.use those confidently mapped

4.{mention, type} pairs as labeled data to infer the types of
remaining candidate mentions.
## challenges
- domain restriction
- name ambiguity
- context sparsity

## a new solution called Clus Type
1. mines both entity mention candidates and relation phrases by POS-constrained
phrase segmentation; this demonstrates great cross-domain
performance (Sec. 3.1).

2.  constructs a heterogeneous graph to faithfully represent candidate entity mentions, entity surface names, and relation phrases and their
relationship types in a unified form (see Fig. 2). The entity mentions are kept as individual objects to be disambiguated, and linked to surface names and relation phrases
(Sec. 3.2-3.4). With the heterogeneous graph, we formulate
a graph-based semi-supervised learning of two tasks jointly:
(1) type propagation on graph, and (2) relation phrase clustering. By clustering synonymous relation phrases, we can
propagate types among entities bridged via these synonymous relation phrases. 

## two ideas:

 - type propagation with relation phrases 

 - multi-view relation phrase clustering
 - 
# 2. Problem Definition

the surface name could refer to the same entity and could also refer to dufferent entites.Moreover it could also refer to different typies 

# 3.construction of graphs

Three Graphs:
- mention-name
- entity name-relation phrase
- mention-mention 
## 3.1 Candidate Generation
没看懂
## 3.2 Mention-Name Subgraph
