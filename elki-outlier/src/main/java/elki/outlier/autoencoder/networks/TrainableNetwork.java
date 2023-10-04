package elki.outlier.autoencoder.networks;

import elki.database.relation.Relation;

/**
 * Interface for trainable neural networks
 * @param <I> Input data type
 * @param <O> Output data type
 *
 * @author Niklas Strahmann
 */
public interface TrainableNetwork<I, O> {
    /**
     * A forward pass through the network.
     * This should not include any gradient calculation (or preparation for it).
     * @param input network input
     * @return network output
     */
    O forward(I input);

    /**
     * Training method for the network, currently limited in the options of optimizers to use by given parameters.
     * Maybe modularize this in the future to support more general types of training.
     * This method also includes parameters for adaptive data size and limiting maximum ratio of used data points.
     *
     * @param trainingData The data to train on.
     * @param rho The friction parameter for the optimizer
     * @param learningRate The learning rate for the optimizer
     * @param maxIterations The maximum Number of Iterations used for Training
     * @param adaptiveFactor The adaptive factor which describes how many data points should be used in an iteration by: <code>a[t] = adaFac * a[t-1]</code>. The initial size should be determined via <code>maxIterations</code>.
     * @param maxSize The ratio of the <code>trainingData</code> used to train the network. This is for ensemble methods which use multiple networks and independent of adaptive Size.
     * @param regularizer The regularization weight. This could alternatively be used for weight decay.
     */
    void train(Relation<I> trainingData, double rho, double learningRate, int maxIterations, double adaptiveFactor, double maxSize, double regularizer);

}
