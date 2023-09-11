package elki.outlier.autoencoder.networks;

import elki.data.NumberVector;
import elki.database.ids.ArrayModifiableDBIDs;
import elki.database.ids.DBIDArrayIter;
import elki.database.ids.DBIDUtil;
import elki.database.relation.Relation;
import elki.logging.Logging;
import elki.math.linearalgebra.VMath;
import elki.utilities.random.RandomFactory;
import jdk.internal.misc.VM;

import java.util.Arrays;
import java.util.Random;

public abstract class AbstractAutoencoderNetwork<V extends NumberVector> implements TrainableNetwork<V, Double> {
    protected final int NUM_LAYERS;

    protected NetworkWeights networkWeights;
    protected NetworkWeights RMSprop;
    protected NetworkWeights batchGradient;

    double cumulativeTrainingError;

    protected double EPS = 0.0001;


    protected Random random;

    AbstractAutoencoderNetwork(int numberLayer, Random random) {
        this.random = random;
        NUM_LAYERS = numberLayer;


        networkWeights = new NetworkWeights().init(numberLayer);
        batchGradient = new NetworkWeights().init(numberLayer);
        RMSprop = new NetworkWeights().init(numberLayer);
    }

    abstract Logging getLog();


    abstract NetworkWeights getGradient(V input);

    @Override
    abstract public Double forward(V input);


    @Override
    public void train(Relation<V> trainingData, double rho, double learningRate, int maxIterations, double adaptiveFactor, double maxSize, double weightDecay) {
        double adaptiveSize = trainingData.size() * maxSize / Math.pow(adaptiveFactor, maxIterations);

        adaptiveSize = Math.max(1.0, adaptiveSize);

        ArrayModifiableDBIDs dbids = DBIDUtil.newArray(trainingData.getDBIDs());
        DBIDUtil.randomShuffle(dbids, random);

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

    }

    class NetworkWeights {
        double[][][] weight;
        double[][] bias;

        NetworkWeights init(int numberLayers) {
            NetworkWeights w = new NetworkWeights();
            w.weight = new double[numberLayers - 1][][];
            w.bias = new double[numberLayers - 1][];
            return w;
        }
    }
}
