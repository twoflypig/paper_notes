#### q:  matmul of input matrix with batch data



```python
embed = tf.reshape(embed, [-1, m])
h = tf.matmul(embed, U)
h = tf.reshape(h, [-1, n, c])
```

https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data