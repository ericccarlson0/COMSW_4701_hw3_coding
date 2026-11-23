
Name: Eric Carlson
UNI: ecc2221

## Questions

1. What is one-hot encoding? Why is this important and how do you implement it in Keras?
There are two ways of returning a category label: by returning the index (`1...N`), or by returning a vector the size of the number of categories (`N`) with zeroes everywhere except for the actual (or, predicted) category. It is important because some interfaces may expect one, and some the other; one-hot decoding is nice because it can be interpreted as a probability distribution (all probability mass at the actual/predicted category; used to compare with softmax output, as in this assignment), or could be used in masking. In Keras, we can use `to_categorical(labels, num_classes=N)`.

2. What is dropout and how does it help overfitting?
Dropout is a type of regularization inspired by the notion of a Neural Network as an ensemble of sub-networks. When a Dropout layer is put between two layers, a certain percentage `rho` of its activations (not weights) are zeroed out. For instance, the activations between Dense(128) and the output Dense(25) are zeroed out in our implementation. Usually, the rest of the activations are rescaled appropriately as well (divide by `1-rho`). This helps with overfitting by reducing reliance on any one parameter or sub-network, encouraging a sort of ensembling.

3. How does ReLU differ from the sigmoid activation function?
ReLU completely clips negative activations, which means no gradient can be propagated back for a negative pre-activation; for positive activations, it has a constant slope, so unlike sigmoid (or tanh), it does not saturate for extremely high positive activations. Further, it is asymmetric. While ReLU can result in "dead" neurons, sigmoid can result in the saturation of neurons.

4. Why is the softmax function necessary in the output layer?
The softmax function is used to turn a set of pre-activations (logits) into weighting or probability distribution; in this case, a probability distribution over output categories.

5. Calculation:
(100, 100, 1) -> (96, 96, 16) -> (48, 48, 16)

6. Compare the final validation accuracy with the final test accuracy.
The final validation accuracy is 100%, whereas the test accuracy is 96.74%. 30 epochs is enough to overfit, although the test accuracy is high, so it is hard to say that we are overfitting much. It is easier for the model to memorize the training examples, because the training dataset used data augmentation to turn a smaller number of original examples into a larger dataset, and the training/validation split did not take care to keep the augmented images and the originals together (a form of data leakage). That is, there are validation images that are near-duplicates of the training images (where the test set comes from a completely separate set of images).

## Architecture Choices

The model is a standard CNN appropriate to the size of the task at hand. There are three blocks of Conv2D (3-by-3 as it is more parameter-efficient to stack smaller convolutions; standard practice) and MaxPool (2-by-2; standard practice). The combination of the Conv2D and the MaxPool layers, with the output channels we chose, allows us to trade-off spatial detail for abstract features (with more channels) while keeping approximately constant the amount of information per layer. The activation function usually does not make much of a difference as long as the network is training well, and the network trained well for this task; sigmoid or Leaky ReLU probably would have done well, if we used those. Adam (with categorical cross-entropy) is a standard optimizer, which balances complexity with the benefits of adaptive optimization. The relative complexity and memory requirements of Adam could be foregone for small batch sizes (and therefore more implicit regularization from SGD), but we kept the batch size at 128 (and we are not at the level of optimizing for memory requirements here).
Finally, before the Flatten layer, the layers are (28 × 28 × 1) -> (14 × 14 × 32) -> (7 × 7 × 64) -> (3 × 3 × 128), which means that there are 1,152 features going in to the Dense(128) layer. There is not much of a reason for choosing specifically 128, except that it is between 1,152 and 25 and is a power of two.
