# 1 Introduction

a key challenge in summarization is to optimally compress the original document in a lossy manner such that the key concepts in the original document are preserved ,whereas in MT,the **translation is excepted to be loss-less**. (在总结中，是把原始文件压缩成一个松散的形式，而在翻译中，是一个词在源于目标之间的对应，这在总结中很不明显)

contributions:

- apply the off-the-shelf attention . 
- a new model  
-  a new data set

# 2 Models

## 2.1 Encoder-Decoder RNN with Attention and Large Vocabulary Trick

This is used as a baseline.And there are some changes for the summary task.

- encoder: bidirectional GRU-RNN 
- decoder:uni-directional GRU-RNN with the same hidden-state size as that of the encoder. 
- A large vocabuary 'trick' : title: On  using very large target vocabulary for neural machine translation .
- the decoder-vocabuary is restricted to words  in the source documents of that batch.the most frequent words in thetarget dictionary are added until the vocabulary reaches a fixed size. The aim of this technique is to reduce the size of the soft-max layer of the decoder which is the main computational bottleneck 

## 2.2 Capturing Keywords using Feature-rich Encoder

In summarization,one of the key challenges is to identify the key concept and key entites in document.And the solutions is : create additional look-up based embedding matrices for the vocabulary of each tag-type,similar to the embeddings for words... Finally,for each word in the source document ,Finally, for each word in the source document, we simply look-up its embeddings from all of its associated tags and concatenate them into a single long vector.(意思就是说，在本来输入是词向量，加上一些额外的特征，拼成一个更大一点的向量)



![summaries_1](..\photo\paper\summaries_1.PNG)

 ## 2.3 Modeling Rare/Unseen Words using Switching Generator-Pointer

Often-times in summarization, the keywords or named-entities in a test document that are central to the summary may actually be unseen or rare with respect to training data .A most common way of handling these out-of-vocabuary wors is to emit an 'UNK' token as a placeholder.**The 'UNK' should  simply point to their location in the source document insted**(就是说，在正常情况下会使用decoer中的词汇，switch 关闭的情况下，产生一个指针指向encoder中的词。这个开关是一个sigmoid 函数，里面的输入时attention的语义向量，上一个提交的embeeding输入，还有此次的隐藏层状态)

![summaries_1](..\photo\paper\summaries_2.PNG)

## 2.4 Capuring Hierarchical Document Structure with Hierarchical Attention

In datasets where the source document is very long, in addition to identifying the keywords in the document, it is also important to identify the key sentences from which the summary can be drawn . word-level and sentence-level

Further, we also concatenate additional positional embeddings to the hidden state of the sentence-level RNN to model positional importance of sentences in the document.  (不仅对句子而且还对词语加权重)

![summaries_1](..\photo\paper\summaries_3.PNG)