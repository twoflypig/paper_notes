# Abstract

language :chinese,japanese,korean,english

encoding  :UTF8,characters,words,romanized characters and romanized words.

models:linear models, fast Text and convolution networks

For convolutional networks:compare using character glyph images,one-hot encoding,and embedding

There are 473 models .

some **conlutions**:

- byte-level one-hot encoding based on UTF-8 consistently produces competitive results for convolutional networks 
- wordlevel n-grams linear models are competitive even without perfect word segmentation 
- fastText provides the best result using character-level n-gram encoding but can overfit when the features are overly rich 

# 1 Introcudtion

There are challengs to do word segmentation in CJK languages .

For one-hot encoding,consider using UTF8 and characters after romanization

For embedding, experiments on encoding levels including character,UTF-8 bytes,romanized characters,segmented word with prebuild word segmenter,and romanized word。

In this article we provide extensive comparisons using multinomial logisitc regression,with bag-of-charcter,bag-of-words and their n-gram and TF-IDF,and fastText.

# 2 Endcoding Mechanisms for Convolutional Networks

use available GNU Unifont,each character is converted to a 16 by 16 pixel image.

For large model,the glyph contains 8 parameterized layers with 6 spatial convolutional layers and 2 liner layers.The small

## 2.2 One-hot Encoding

one-hot encoding is only computationally feasible if the entity set is relatively small.And here are 2 simple solutions to this problem .The first one is to treat the text (in UTF-8) as a sequence of bytes and encode at byte-level. The second one, already presented in Zhang et al. (2015), is to romanize the text so that encoding using the English alphabet is feasible.

