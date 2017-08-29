The input to the framework is a POS-tagged text corpus D

Automatically Labeled Training Data:map the mentions extracted from corpus to KB entities and  heuristically assign type labels to the mapped mentions

# Abstract

1. data-driven text segmentation algorithm to extract entity mentions
2. domain-

# The problem definition

- estimate a relation type in Set R for each test relation mention in unlabeled relation mentions 
- a single type-path for each entity mention in  Set Z.

# 3 The Cotype Framework

there are two unique framework :

- type association in distant supervision between linkable entity mentions and their KB-mapped entities is context-agnostic.May contains "false" types
- there exits dependencies between relation  mentions and their entity arguments 

**Solution** : cast the type prediction task as weakly-supervised learning and use relational learning to capture interactions  

## 3.1 Candidate Generation

**Entity Mention Detection** :Traditional entity recongition systems rely on a set of linguistic features (e.g., dependency parse structures of a sentence) to train sequence labeling models (for a few common entity types).  

**Solution**:By using quality examples from KB as guidance, it partitions sentences into segments of entity mentions and words, by incorporating (1) corpus-level concordance statistics; (2) sentence-level lexical signals; and (3) grammatical constraints (i.e., POS tag patterns). 

**Text Feature Extraction**:

## 3.2 Joint Entity and Relation Embedding

a simple solution is to embed the whole graph into a single low-dimensional space .However, such a solution encounters several problems: (1) False types in candidate type sets (i.e., false mention-type links in the graph) negatively impact the ability of the model to determine mention’s true types; and (2) a single embedding space cannot capture the differences in entity and relation types (i.e., strong link between a relation mention and its entity mention argument does not imply that they have similar types). 

### Modeling Types of Entity Mentions

Relevance between entity mentions and their true type labels should be progressively estimated based on the text features extracted from their local contexts 

### Modeling Entity-Relation Interactions

Entity types of the entity mention arguments pose constraints on the search space for the relation types of the relation mention .For example,force m1 +z $ \approx $ m2