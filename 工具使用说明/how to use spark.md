# spark 和hadoop 是什么关系？

##### 他们两个是不同的东西

Hadoop 和 spark 同样是大数据框架，但是他们的目的各不相同。Hadoop本质上是一个分布式数据架构:它在一个服务器集群内往多个节点上分布存储数据，这意味着你不需要去购买和维护传统的硬件。它也能索引和保持对数据的跟踪，能够使大数据处理和分析能够比之前的更有效.另一方面，spark是一个运行在分布式数据集合上面的数据处理工具，它并不涉及分布式存储。

##### 他们两个不是相互依赖的

hadoop不仅包含一个广为人知的Hadoop Distributed File System的存储功能，而且包含一个叫做 MapReduce的处理功能。传统上，你可以在没有Hadoop的情况下使用Spark ，但是Spark和Hadoop一起工作的会更好，因为Spark 是针对Hadoop设计的。

##### Spark 更快

Spark 比MapReduce的速度要快的多，因为Spark处理数据的方式。MapReduce 分成一些步骤来进行操作，而Spark是一次性操作整个数据集。MapReduce 的工作流程看起来就像这样:从集群中读取数据，执行一个操作，保存结果到集群中，从集群中读取更新的数据，执行下一个操作，把这个结果保存到集群中，如此循环。另一方面，Spark是在内存中与接近实时完成整个数据的分析操作:从集群中读取数据，执行所有要求的分析操作，保存结果到集群中。Spark在批处理中能够比MapReduce要快近10倍，在内存内分析中能够快100倍。

Refer: [Five things you need to know about Hadoop v. Apache Spark](https://www.infoworld.com/article/3014440/big-data/five-things-you-need-to-know-about-hadoop-v-apache-spark.html)

# spark 读取

At the `scala` REPL prompt enter:

```scala
val file = sc.textFile("/tmp/data.txt")
val counts = file.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey(_ + _)
```

Save `counts` to a file:

```scala
cacounts.saveAsTextFile("/tmp/wordcount")
```

**Viewing the WordCount output with Scala Shell**

To view the output, at the `scala>` prompt type:

```scala
counts.count()
```

You should see an output screen similar to:

```scala
...
16/02/25 23:12:20 INFO DAGScheduler: Job 1 finished: count at <console>:32, took 0.541229 s
res1: Long = 341

scala>
```

To print the full output of the WordCount job type:

```scala
counts.toArray().foreach(println)c
```

You should see an output screen similar to:

```scala
...
((Hadoop,1)
(compliance,1)
(log4j.appender.RFAS.layout.ConversionPattern=%d{ISO8601},1)
(additional,1)
(default,2)

scala>
```

**Viewing the WordCount output with HDFS**

To read the output of WordCount using HDFS command:
Exit the Scala shell.

```scala
exit
```
# 一些常用的函数
spark 2.1.0的写法: df.select($"name", $"age" + 1).show()  
spark 1.6.0的写法:df.select(df("name"),df("age")+1).show() 
$"yaer"好像是string的意思

# RDD

RDD，全称为Resilient Distributed Datasets，是一个容错的、并行的数据结构，可以让用户显式地将数据存储到磁盘和内存中，并能控制数据的分区。

所以说，RDD是spark所特有的结构