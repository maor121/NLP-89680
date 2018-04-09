# NLP-89680
A repository for NLP course at uni.

Assignment 1
------------


Assignment 2
------------

Assignment 3
------------
**Distributional Semantics**<br/>
Finding similar words by meaning using a combination of algorithms, and making a detailed report comparing them.<br/>
Data: wikipedia<br/>
Algorithms:
- word contexts: (a) sentence (b) window (c) dependency tree (parent\son, direction of arc, jump over preposition)
- similarity: (a) cosine distance (b) PMI
- order: (a) 1st order similarity (b) 2nd order similarity

Assignment 4
------------
**Relation Extraction**<br>
Given a small amount of data (news articles), extract Named Entities, from each sentence, and the relation between them.<br/>
i.e Yosi (work for) CBS<br/><br/>
Algorithm:
1. For each sentence, extract Named Entities, dependency tree and POS tagging using spacy library.
2. Generate a sequence from the path between the two entities on the dependency tree.
3. Run LSTM on the path, concat output with other feature vectors, such as: Named Entity type, Named Entity POS tag.
4. Pass through MLP with softmax activation.

<br/>
Challenges:

- small dataset
- missing labels (entities\relations that should have been included in the gold file)
- mismatches between gold file Named Entities, and spacy output Named Entities.

Architecture choice:<br/>
pure ML approach, instead of hybrid ML and rule based. A hybrid could be made after error analysis, for example: the model sometimes confuses relation (work for) with (kill), because both relations contain PERSONs.
