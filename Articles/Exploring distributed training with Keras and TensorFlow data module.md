# Exploring distributed training with Keras and TensorFlow data module

Since integrated as a core high-level API (Application Programming Interface) in TensorFlow(`tf`), ***Keras*** (`tf.keras`) has gradually become compatiable with many useful modules in TensorFlow. Among many, this article would explore how TensorFlow's **`tf.data`** and **`tf.distribute.Strategy`** APIs can be used with `tf.keras` to build efficient data input pipelines and to enable distributed training using existing user-friendly `tf.keras` models/codes with minimal changes.

## what is [tf.distribute.Strategy](http://www.tensorflow.org/guide/distribute_strategy) module for?

When relatively small amount of data is on hand to train relatively simple model with a single processing unit (CPU, GPU or TPU), training may not take too much time and a practitioner can go through severl rounds of trial-and-errors to optimize their models. When the amount of data gets bigger or the model gets complicated, training becomes computationally expensive and the corresponding training may take a few hours, days, even weeks, making the overall model development process very inefficient. The solution is the ***distributed training*** or going palellel with multiple processing units (or even workers)! There are two general types of parallelism:

* Model Parallelism: when model is too large
* Data Parallelism: when data is too large

<p align="center"><img src="../images/distrib_model_data_parallelism.png" width="80%"></p>
source: [[Uber Open Summit 2018] Horovod: Distributed Deep Learning in 5 Lines of Python](https://www.youtube.com/watch?v=4y0TDK3KoCA&t=1311s)

However, it should be noted that the model parallelism is not so much effective these days since modern accelerators (GPU or TPU) have enough RAMs to store a single model with no difficulties.

### [Data parallelism](https://en.wikipedia.org/wiki/Data_parallelism)

When a single processing unit is not enough to train a large amount of data, a natural way to fasten training time is to use multiple accelerators (GPU or TPU) in a single machine or to use multiple machines with single/multiple accelerators in a parallel manner. In other words, the data can be splitted into smaller pieces to be consumed in different accelerators to build the ultimate model. Such approach is doable when training deep learning models because the learning algorithm (especially the gradient computing parts during the back-prop) is represented as sums of functions. In general, there are two common ways of distributing training with data parallelism:

* *Asynchronous training*: all workers train over the input data independently and update variables asynchronously. ***Example***: asynchronous parameter server architecture - parameter servers (PS) hold the parameters and other workers fetch and update model parameters independently. The primary cons of this approach is that workers may work based on different parameters, delaying the convergence. This approach is preferable when a training system is composed of a large number of not much powerful/reliable machines (i.e. CPUs). 
* ***Synchronous training***: all workers train over different slices of input data in sync. ***Example***: synchronous all-reduce architecture - model parameters are mirrored across the workers and each worker computes loss and gradient based on the subset of input data given. The gradients get aggregated at each step and the result becomes available on each worker via all-reduce. Based on the shared gradients, each worker then update the model parameters to the identical values. When the communication among different workers is controlled well (, this approach can enables highly effective distributed training. This approach is preferable when a training system is composed of powerful/reliable machines (i.e. GPUs or TPUs) with strong communication. 

<p align="center"><img src="../images/distrib_data_parallelism_summary.png" width="80%"></p>
source: [Distributed TensorFlow training (Google I/O '18)](https://www.youtube.com/watch?v=bRMGoPqsn20)

By `tf.distribute.Strategy` API, TensorFlow provides highly effective ways to implement the powerful distributed training into not only the custom training loops, but also TensorFlow's high-level APIs including `tf.keras` and `tf.estimator`. A few strategies it provides include:

* `tf.distribute.MirroredStrategy`: synchronous training --> multiple GPUs on a single machine using NVIDIA NCCL as the default all-reduce implementation
* `tf.distribute.CentralStorageStrategy`: synchronous training --> a single CPU stores the model parameters and multiple GPUs on a single machine perform operations
* `tf.distribute.MultiWorkerMirroredStrategy`: synchronous training --> similar to `tf.distribute.MirroredStrategy` but operates on multiple machines potentially with multiple GPUs
* `tf.distribute.ParameterServerStrategy`: asynchronous training --> model varaibles are stored on one parameter server and computation is replicated across all GPUs of the all the workers

A full list of available strategies can be found [here](http://www.tensorflow.org/guide/distribute_strategy).

## what is [tf.data](http://www.tensorflow.org/guide/datasets) module for?

In normal cases, the input data is read in a `numpy` array format and gets fed into the model for training. The `tf.data` API provides an alternative way to do the similar task while being much more efficient when handling large amounts of data especially in case of abovementioned distributed training scenarios. This API enables to aggregate data from files in a distributed file system (***Extract***), to apply desired transformation (***Transform***) and to form randomized mini-batches (***Load***) of data-label pair for training. Since this article will explore the need/benefit of distributed training, the `tf.data` API needs to be investigated.

In a synchronous distributed training architecture, the dataset is processed by slow CPUs while the heavy-load training is performed by much faster accelerators. In this case, the bottlenect of the system is the slow CPUs. The Extract-Transform-Load (ETL) process of `tf.data` effectively utilizes CPUs via ***pipelining***, thereby improving the efficiecny of the overall synchronous distributed training. 

<p align="center"><img src="../images/distrib_pipelining_summary.png" width="80%"></p>
source: [Data input pipelines: tf.data Performance](https://www.tensorflow.org/beta/guide/data_performance)

As described in the image above, CPUs (preparing input data) and accelerators (training) experience severe idel times during the typical synchronous distributed training process without pipelining. With pipelining, the CPUs are fully utilized by overlapping data producing and data consuming steps, minimizing the overall training time. More details about how pipelining is achieved with `tf.data` will be explained later with actual code examples.

## Distributed training with Keras - sample codes for training image classification on MNIST

In order to demonstrate the model building process using `tf.keras` API for distributed training, an arbitrary model can be built to classify MNIST digits. Please note that this model was meant to be highly simple, not necessarily producing the best model. The full codes and the output results are presented [***HERE***](https://github.com/sungsujaing/ML_DL_articles_resources/blob/master/Code%20appendix/Exploring%20distributed%20training%20with%20Keras%20and%20TensorFlow%20data%20module.ipynb).

```python
import tensorflow as tf
import numpy as np

epochs = 10
batch = 256

def build_cnn(): # (1)
    inputs = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.Conv2D(8,(3,3),padding='same',activation='relu')(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10,activation='softmax')(x)
    model = tf.keras.Model(inputs,outputs)
    return model
```

(1) arbitrary model built with `tf.keras.Model` to be used for regular training witn `numpy` array input, regular training with `tf.data.Dataset` input and distributed training with `tf.data.Dataset` input

### regular training with numpy array as inputs

```python
(X_train_1,y_train_1),(X_test_1,y_test_1) = tf.keras.datasets.mnist.load_data()
X_train_1 = X_train_1.astype(np.float32)/255. # (1)
X_train_1 = np.expand_dims(X_train_1,axis=-1)
X_test_1 = X_test_1.astype(np.float32)/255.
X_test_1 = np.expand_dims(X_test_1,axis=-1)
y_train_1 = tf.keras.utils.to_categorical(y_train_1,10)
y_test_1 = tf.keras.utils.to_categorical(y_test_1,10)

model_1 = build_cnn()
model_1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_1.fit(X_train_1,y_train_1,epochs=epochs,batch_size=batch,validation_data=(X_test_1,y_test_1),verbose=2)
```

(1) preprocessing: X - normalization and shaping, y - one-hot encoding

### regular training with tf.data.Dataset as inputs

```python
def preprocessing(img,label): # (1)
    img = tf.cast(img,tf.float64)
    img = img/255. # (2)
    img = tf.expand_dims(img,axis=-1)
    label = tf.one_hot(label,10,dtype=tf.int32) # (3)
    return img,label # (4)
```

(1) custom `preprocessing` function to be applied to each element of the input dataset (transformation)

(2) normalization and shaping for image

(3) one-hot encoding for label

(4) return a pair of data and label

```python
(X_train_2,y_train_2),(X_test_2,y_test_2) = tf.keras.datasets.mnist.load_data()

train_ds = tf.data.Dataset.from_tensor_slices((X_train_2,y_train_2)) # (1)
train_ds = train_ds.map(preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE) # (2)
train_ds = train_ds.shuffle(256) # (3)
train_ds = train_ds.batch(batch) # (4)
train_ds = train_ds.repeat(epochs) # (5)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # (6)

test_ds = tf.data.Dataset.from_tensor_slices((X_test_2,y_test_2)) # (7)
test_ds = test_ds.map(preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch)
test_ds = test_ds.repeat(epochs)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

model_2 = build_cnn()
model_2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_2.fit(train_ds,epochs=epochs,steps_per_epoch=len(X_train_2)//batch,validation_data=test_ds,validation_steps=5,verbose=2) # (8)
```

(1) create a `tf.data.Dataset` instance

(2) transformation with custom `preprocessing` function. Since input elements are not dependent on one another, this `tf.data.Dataset.map` transformation can be parallelized across multiple CPU cores. In order to automatically optimize the number of cores to parallelize, `tf.data.experimental.AUTOTUNE` is used.

(3) randomly shuffle the data with a fixed buffer size. To ensure proper shuffling, the buffer size needs to be greater than the batch size.

(4) define the batch size for training

(5) define how many times to iterate over a dataset in multiple epochs

(6) enables pipelining mechanism by decoupling the data producing and consuming steps using a background thread and an internal buffer. The number of elements to prefetch should be at least the number of batches to be consumed. In order to automatically optimize the buffer size for prefetch, `tf.data.experimental.AUTOTUNE` is used.

(7) Similar transformation for test data

(8) Since batch_size is provided during the transformation, `fit` requires `steps_per_epoch` and `steps` argument for training and test datasets

### synchronous distributed training with tf.data.Dataset as inputs

```python
(X_train_3,y_train_3),(X_test_3,y_test_3) = tf.keras.datasets.mnist.load_data()

mirrored = tf.distribute.MirroredStrategy()
print('\nnumber of replicas in sync: {}'.format(mirrored.num_replicas_in_sync))

batch_per_replica = batch
global_batch = batch_per_replica * mirrored.num_replicas_in_sync
print('global batch: {}'.format(global_batch))

train_ds_dist = tf.data.Dataset.from_tensor_slices((X_train_3,y_train_3)).map(preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(256).batch(global_batch).repeat(epochs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds_dist = tf.data.Dataset.from_tensor_slices((X_test_3,y_test_3)).map(preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(global_batch).repeat(epochs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

with mirrored.scope():
    model_3 = build_cnn()
    model_3.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_3.fit(train_ds_dist,epochs=epochs,steps_per_epoch=len(X_train_3)//batch,validation_data=test_ds_dist,validation_steps=5,verbose=2)
```









 



`tf.data.Dataset.shuffle` does not signal the end of an epoch until the shuffle buffer is empty. Shuffle places before a repeat will show every element of one epoch before moving to the next

  



