package elki.outlier.autoencoder.networks;

import elki.data.NumberVector;
import elki.database.ids.ArrayModifiableDBIDs;
import elki.database.ids.DBIDArrayIter;
import elki.database.ids.DBIDUtil;
import elki.database.relation.Relation;
import elki.logging.Logging;
import elki.logging.statistics.AtomicLongCounter;
import elki.math.geometry.XYCurve;
import elki.math.linearalgebra.VMath;
import elki.result.Metadata;

import java.util.Arrays;
import java.util.Random;

/**
 * Base class for Autoencoder Network for outlier detection.
 * The encoder and decoder networks can't be accessed on their own.
 * The forward pass should return the reconstruction error.
 * The training uses RMSprop optimizer with an adaptive iteration size.
 * @param <V> input vector type
 */
public abstract class AbstractAutoencoderNetwork<V extends NumberVector> implements TrainableNetwork<V, Double> {

    static AtomicLongCounter networkNumber = new AtomicLongCounter(AbstractAutoencoderNetwork.class.toString());
    protected final int NUM_LAYERS;

    /**
     *
     */
    protected NetworkWeights networkWeights;
    protected NetworkWeights RMSprop;
    protected NetworkWeights batchGradient;

    double cumulativeTrainingError;

    protected double EPS = 0.0001;


    protected Random random;

    AbstractAutoencoderNetwork(int numberLayer, Random random) {
        this.random = random;
        NUM_LAYERS = numberLayer;


        networkWeights = NetworkWeights.init(numberLayer);
        batchGradient = NetworkWeights.init(numberLayer);
        RMSprop = NetworkWeights.init(numberLayer);
    }

    /**
     * @return Logger for instance
     */
    abstract Logging getLog();

    /**
     * Performs a forward pass followed by a backward pass to calculate the gradient.
     * @param input Input values for the network
     * @return Gradient with respect to the network weights and biases.
     */
    abstract NetworkWeights getGradient(V input);

    /**
     * Perform only a forward pass. Because no backward pass is calculated, there is no need to cache intermediate values,
     * intended to be similar to <code>torch.no_grad</code>.
     * @param input network input
     * @return reproduction error
     */
    @Override
    abstract public Double forward(V input);

    /**
     * Trains the network weights with the RMSprop optimizer and adaptive Sampling.
     * Adds a <code>XYCurve</code> with iteration mean training error.
     * @param trainingData The data to train on.
     * @param rho The friction parameter for RMSprop optimizer
     * @param learningRate The learning rate for RMSprop optimizer
     * @param maxIterations The maximum Number of Iterations used for Training
     * @param adaptiveFactor The adaptive factor which describes how many data points should be used in an iteration by: <code>a[t] = adaFac * a[t-1]</code>.
     *                       The initial size is determined as <code>N / (adaFactor ^ maxIterations)</code>.
     * @param maxSize The ratio of the <code>trainingData</code> used to train the network. This is intended for ensemble methods which use multiple networks.
     *               This number influences the initial size of adaptive sampling by reducing the training size <code>N = N * maxSize</code>.
     * @param weightDecay The weight decay.
     */
    @Override
    public void train(Relation<V> trainingData, double rho, double learningRate, int maxIterations, double adaptiveFactor, double maxSize, double weightDecay) {
        double adaptiveSize = trainingData.size() * maxSize / Math.pow(adaptiveFactor, maxIterations);

        adaptiveSize = Math.max(1.0, adaptiveSize);

        ArrayModifiableDBIDs dbids = DBIDUtil.newArray(trainingData.getDBIDs());
        DBIDUtil.randomShuffle(dbids, random);

        XYCurve trainingError = new XYCurve("Iteration network " + networkNumber.increment(),"Error");

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            DBIDArrayIter iter = dbids.iter();

            cumulativeTrainingError = 0.0;
            for (int trainingSample = 0; trainingSample < (int) adaptiveSize; trainingSample++) {
                if (!iter.valid()) {
                    break;
                }

                if (iteration == 0) {
                    //getLog().verbose("1. Id: " + iter.toString());
                }

                V input = trainingData.get(iter);
                NetworkWeights gradient = getGradient(input);

                for (int i = 0; i < NUM_LAYERS - 1; i++) {
                    VMath.plusEquals(batchGradient.weight[i], VMath.transpose(gradient.weight[i]));
                    VMath.plusEquals(batchGradient.bias[i], gradient.bias[i]);
                }

                iter.advance();
            }
            if (iteration == maxIterations - 1) {
                getLog().verbose("Training error in iteration " + iteration + " with " + (int) adaptiveSize + " samples: " + cumulativeTrainingError / (int) adaptiveSize);
            }
            trainingError.addAndSimplify(iteration, cumulativeTrainingError/(int)adaptiveSize);


            for (int i = 0; i < NUM_LAYERS - 1; i++) {
                VMath.timesEquals(batchGradient.weight[i], 1.0 / (int) adaptiveSize);
                VMath.timesEquals(batchGradient.bias[i], 1.0 / (int) adaptiveSize);
            }

            for (int i = 0; i < NUM_LAYERS - 1; i++) {
                //RMSprop optimization, maybe modularize? complicated due to dependence on meta-information regarding network architecture
                RMSprop.weight[i] = VMath.plusTimes(VMath.times(RMSprop.weight[i], rho), NetworkMathHelper.hadamardSquare(batchGradient.weight[i]), 1 - rho);
                RMSprop.bias[i] = VMath.plusTimes(VMath.times(RMSprop.bias[i], rho), NetworkMathHelper.hadamardSquare(batchGradient.bias[i]), 1 - rho);

                //update parameters
                //TODO find out why this is plus, not minus
                VMath.plusTimesEquals(networkWeights.weight[i],
                        VMath.hadamard(
                                VMath.root(RMSprop.weight[i], EPS),
                                batchGradient.weight[i]),
                        1.0 / learningRate);

                VMath.plusTimesEquals(networkWeights.bias[i],
                        VMath.times(
                                NetworkMathHelper.root(RMSprop.bias[i], EPS),
                                batchGradient.bias[i]),
                        1.0 / learningRate);


                //weight decay
                VMath.minusTimesEquals(networkWeights.weight[i], networkWeights.weight[i], learningRate * weightDecay);
                VMath.minusTimesEquals(networkWeights.bias[i], networkWeights.bias[i], learningRate * weightDecay);
            }
            //Reset batch gradient after iteration
            for (int layer = 0; layer < NUM_LAYERS - 1; layer++) {
                for (int i = 0; i < batchGradient.weight[layer].length; i++) {
                    Arrays.fill(batchGradient.weight[layer][i], 0);
                }
                Arrays.fill(batchGradient.bias[layer], 0);
            }
            adaptiveSize *= adaptiveFactor;
            adaptiveSize = Math.min(adaptiveSize, trainingData.size() * maxSize);
        }

        Metadata.hierarchyOf(trainingData).addChild(trainingError);

    }

    /**
     * Samples n ints in [0,max-1] with replacement
     * @param max exclusive upper bound
     * @param n number of digits
     * @return sampled ints, multiple occurrences removed
     */
    protected int[] sampleInts(int max, int n){
        return random.ints(0, max).limit(n).distinct().toArray();
    }

    /**
     * Data class for holding weights and biases
     */
    static class NetworkWeights {
        double[][][] weight;
        double[][] bias;

        /**
         * Initialize a new Weights class. Only the first dimension of the arrays will be initialized with the number of
         * layers. The dimensionality of the layers will not be considered.
         * @param numberLayers Number of layers, to track weights and bias for each separately.
         * @return Empty weights and biases.
         */
        static NetworkWeights init(int numberLayers) {
            NetworkWeights w = new NetworkWeights();
            w.weight = new double[numberLayers - 1][][];
            w.bias = new double[numberLayers - 1][];
            return w;
        }
    }
}
