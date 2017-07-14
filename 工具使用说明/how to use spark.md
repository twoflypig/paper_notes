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