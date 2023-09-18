package elki.outlier.autoencoder.networks;

import elki.data.NumberVector;
import elki.logging.Logging;
import elki.math.linearalgebra.VMath;
import elki.utilities.random.RandomFactory;

import java.util.Arrays;
import java.util.Random;


public class AutoencoderLayers3<V extends NumberVector> extends AbstractAutoencoderNetwork<V> {

    private static final Logging LOG = Logging.getLogger(AutoencoderLayers3.class);
    protected int LAYER_MIN_DIMENSION = 3;
    protected final int dataDimension;
    protected final int hiddenDimension;
    protected boolean[][][] ActiveConnections = new boolean[2][][];

    public AutoencoderLayers3(double alpha, int dataDimension, Random random ) {
        super(3, random);
        this.dataDimension = dataDimension;
        this.hiddenDimension = Math.max(LAYER_MIN_DIMENSION, (int) (alpha * dataDimension));

        InitArchitecture();

        //RANDNET drop random connections
        for (int layer = 0; layer < (NUM_LAYERS - 1) / 2; layer++) {
            int[] activeConnectionsIndices = sampleInts(dataDimension * hiddenDimension, dataDimension * hiddenDimension);
            for (int index : activeConnectionsIndices) {

                //Encoder
                ActiveConnections[layer][index / hiddenDimension][index % hiddenDimension] = true;
                //Decoder
                ActiveConnections[NUM_LAYERS - 2 - layer][index % hiddenDimension][index / hiddenDimension] = true;
            }
        }

        InitWeights();
    }


    private void InitArchitecture() {
        int[] dimension = new int[NUM_LAYERS];

        dimension[0] = dataDimension;
        dimension[1] = hiddenDimension;
        dimension[NUM_LAYERS - 1] = dataDimension;

        LOG.verbose("Dimensions " + Arrays.toString(dimension));

        for (int i = 1; i < NUM_LAYERS; i++) {
            networkWeights.weight[i - 1] = new double[dimension[i - 1]][dimension[i]];
            ActiveConnections[i - 1] = new boolean[dimension[i - 1]][dimension[i]];
            RMSprop.weight[i - 1] = new double[dimension[i - 1]][dimension[i]];
            batchGradient.weight[i - 1] = new double[dimension[i - 1]][dimension[i]];

            networkWeights.bias[i - 1] = new double[dimension[i]];
            RMSprop.bias[i - 1] = new double[dimension[i]];
            batchGradient.bias[i - 1] = new double[dimension[i]];
        }

    }

    //this uses Kaiming (aka He aka MSRA) init as a placeholder, tha paper uses "pre-training" to determine the initial weights
    private void InitWeights() {
        Random random = RandomFactory.DEFAULT.getRandom();
        for (int layer = 0; layer < NUM_LAYERS - 1; layer++) {
            int maxInDimension = ActiveConnections[layer].length;
            int maxOutDimension = ActiveConnections[layer][0].length;
            for (int inDimension = 0; inDimension < maxInDimension; inDimension++) {
                for (int outDimension = 0; outDimension < maxOutDimension; outDimension++) {
                    if (ActiveConnections[layer][inDimension][outDimension]) {
                        networkWeights.weight[layer][inDimension][outDimension] = random.nextGaussian() * Math.sqrt(2.0 / maxInDimension);
                    }
                }
            }
        }
    }

    @Override
    Logging getLog() {
        return LOG;
    }

    @Override
    NetworkWeights getGradient(V input) {

        //forward pass
        double[] inputArray = input.toArray();

        //first layer
        double[] hiddenLayer = VMath.transposeTimes(networkWeights.weight[0], inputArray);
        VMath.plusEquals(hiddenLayer, networkWeights.bias[0]);
        double[] sigmoidedHiddenLayer = NetworkMathHelper.sigmoid(hiddenLayer);

        //second layer
        double[] output = VMath.transposeTimes(networkWeights.weight[1], sigmoidedHiddenLayer);
        VMath.plusEquals(output, networkWeights.bias[1]);
        double[] sigmoidedOutput = NetworkMathHelper.sigmoid(output);

        //Reproduction error
        double[] error = VMath.minus(sigmoidedOutput, inputArray);
        double sse = VMath.squareSum(error);

        cumulativeTrainingError += sse;

        NetworkWeights gradient = NetworkWeights.init(NUM_LAYERS);

        //backward propagation
        double[] scoreGradient = VMath.times(error, 2);

        double[] l2SigmoidLocalGradient = NetworkMathHelper.sigmoidGradient(sigmoidedOutput);
        gradient.bias[1] = VMath.times(l2SigmoidLocalGradient, scoreGradient);
        gradient.weight[1] = NetworkMathHelper.RandLayerWeightGradient(hiddenLayer, gradient.bias[1], ActiveConnections[1]);

        double[] l2inputGradient = VMath.times(networkWeights.weight[1], gradient.bias[1]);

        double[] l1SigmoidLocalGradient = NetworkMathHelper.sigmoidGradient(sigmoidedHiddenLayer);
        gradient.bias[0] = VMath.times(l1SigmoidLocalGradient, l2inputGradient);
        gradient.weight[0] = NetworkMathHelper.RandLayerWeightGradient(inputArray, gradient.bias[0], ActiveConnections[0]);

        return gradient;
    }

    @Override
    public Double forward(V input) {
        double[] inputArray = input.toArray();
        //first layer
        double[] hiddenLayer = VMath.transposeTimes(networkWeights.weight[0], inputArray);
        VMath.plusEquals(hiddenLayer, networkWeights.bias[0]);
        NetworkMathHelper.sigmoid(hiddenLayer);

        //second layer
        double[] output = VMath.transposeTimes(networkWeights.weight[1], hiddenLayer);
        VMath.plusEquals(hiddenLayer, networkWeights.bias[1]);
        NetworkMathHelper.sigmoid(output);

        //Reproduction error
        double[] error = VMath.minus(output, inputArray);
        double sse = VMath.squareSum(error);
        return sse;
    }

}
