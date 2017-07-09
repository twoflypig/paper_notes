# train own NER model

应该看的两个代码 **CRFClassifier **  和 **NERFeatureFactory**

训练数据以Tab隔开。最低限度words token是一列，class label是另一列。（你需要每个词都带有标记的连续文本作为监督训练数据。给出数据文件，应该被回答的东西的意义，和通过属性文件产生怎样的特征。或者至少是正常和最好的方式-就是把所有的信息作为命令行参数输入，map属性被用来定义列的意义。行是从0开始计数。一列是选定词，另一列是NER的类别，同时另一个应该被叫做‘词’并且拥有tokens(**意味的词必须分对**)。存在的特征抽取也知道一些其他的列名例如tag。有相当的文件在Javadoc **NERFeatureFactory** 

以下是一个NER设置文件的例子

```
trainFile = training-data.col
serializeTo = ner-model.ser.gz
map = word=0,answer=1

useClassFeature=true
useWord=true
useNGrams=true
noMidNGrams=true
maxNGramLeng=6
usePrev=true
useNext=true
useSequences=true
usePrevSequences=true
maxLeft=1
useTypeSeqs=true
useTypeSeqs2=true
useTypeySequences=true
wordShape=chris2useLC
useDisjunctive=true
```

回答的剩余部分会给出一个简单但是完整的NER训练的例子。假设我们想要去建立一个Jane Austen novels的NER系统。我们可以在第一章训练数据下载后进行训练

java -cp stanford-ner.jar edu.stanford.nlp.process.PTBTokenizer jane-austen-emma-ch1.txt > jane-austen-emma-ch1.tok

我们给每一列添加了一个实体类型，但是我们可以额外的增加一个实体类型例如LOC作为location。在训练文件中