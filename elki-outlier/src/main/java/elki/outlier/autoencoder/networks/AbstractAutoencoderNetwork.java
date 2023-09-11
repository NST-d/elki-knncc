package elki.outlier.autoencoder.networks;

import elki.data.NumberVector;
import elki.database.ids.ArrayModifiableDBIDs;
import elki.database.ids.DBIDArrayIter;
import elki.database.ids.DBIDUtil;
import elki.database.relation.Relation;
import elki.logging.Logging;
import elki.math.linearalgebra.VMath;
import elki.utilities.random.RandomFactory;

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
                if (!iter.valid()){
                    break;
                }

                if (iteration == 0){
                    //getLog().verbose("1. Id: " + iter.toString());
                }


                V input = trainingData.get(iter);
                NetworkWeights gradient = getGradient(input);

                //RMSprop optimization, maybe modularize? complicated due to dependence on meta-information regarding network architecture
                for (int i = 0; i < NUM_LAYERS - 1; i++) {
                    RMSprop.weight[i] = VMath.plusTimes(VMath.times(RMSprop.weight[i], rho), NetworkMathHelper.transposeHadamardSquare(gradient.weight[i]), 1 - rho);
                    RMSprop.bias[i] = VMath.plusTimes(VMath.times(RMSprop.bias[i], rho), NetworkMathHelper.hadamardSquare(gradient.bias[i]), 1 - rho);

                    VMath.plusEquals(batchGradient.weight[i], VMath.hadamard(VMath.transpose(gradient.weight[i]), VMath.divideEquals(learningRate, VMath.root(RMSprop.weight[i], EPS))));
                    VMath.plusEquals(batchGradient.bias[i], VMath.times(gradient.bias[i], NetworkMathHelper.divideEquals(learningRate, NetworkMathHelper.root(RMSprop.bias[i], EPS))));
                }

                iter.advance();
            }
            if(iteration == maxIterations -1) {
                getLog().verbose("Training error in iteration " + iteration + " with " + (int) adaptiveSize + " samples: " + cumulativeTrainingError / (int) adaptiveSize);
            }
            //weight decay
            for (int i = 0; i < NUM_LAYERS - 1; i++) {
                //TODO find out why this is plus, not minus
                VMath.plusTimesEquals(networkWeights.weight[i], batchGradient.weight[i], 1.0 / (int) adaptiveSize);
                VMath.minusTimesEquals(networkWeights.weight[i], networkWeights.weight[i], learningRate * weightDecay);

                VMath.plusTimesEquals(networkWeights.bias[i], batchGradient.bias[i], 1.0 / (int) adaptiveSize);
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
            adaptiveSize = Math.min(adaptiveSize, trainingData.size()*maxSize);
        }

    }
    class NetworkWeights {
        double[][][] weight;
        double[][] bias;

        NetworkWeights init(int numberLayers){
            NetworkWeights w = new NetworkWeights();
            w.weight = new double[numberLayers-1][][];
            w.bias = new double[numberLayers-1][];
            return w;
        }
    }
}
