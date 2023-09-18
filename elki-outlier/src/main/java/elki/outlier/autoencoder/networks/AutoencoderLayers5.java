package elki.outlier.autoencoder.networks;

import elki.data.NumberVector;
import elki.logging.Logging;
import elki.math.linearalgebra.VMath;
import elki.utilities.random.RandomFactory;

import java.util.Arrays;
import java.util.Random;

public class AutoencoderLayers5<V extends NumberVector> extends AbstractAutoencoderNetwork<V>{

    private static final Logging LOG = Logging.getLogger(AutoencoderLayers5.class);
    protected int LAYER_MIN_DIMENSION = 3;
    protected final int dataDimension;
    protected final int[] hiddenDimension = new int[(NUM_LAYERS -1)/2];
    protected boolean[][][] ActiveConnections = new boolean[NUM_LAYERS - 1][][];

    public AutoencoderLayers5(double alpha, int dataDimension, Random random){
        super(5, random);

        this.dataDimension = dataDimension;
        this.hiddenDimension[0] = Math.max(LAYER_MIN_DIMENSION, (int) (alpha * dataDimension));
        this.hiddenDimension[1] = Math.max(LAYER_MIN_DIMENSION, (int) (alpha * hiddenDimension[0]));

        InitArchitecture();

        //RANDNET drop random connections
        for( int layer = 0; layer < (NUM_LAYERS - 1)/2; layer++){
            int in = networkWeights.weight[layer].length;
            int out = networkWeights.weight[layer][0].length;

            int[] activeConnectionsIndices = sampleInts( in * out, in * out);

            LOG.verbose("Use " + activeConnectionsIndices.length + "/"+ in*out + " connections.");
            for(int index : activeConnectionsIndices){

                //Encoder
                ActiveConnections[layer][index /out][index % out] = true;
                //Decoder
                ActiveConnections[NUM_LAYERS -2 -layer][index % out][index /out] = true;
            }
        }

        InitWeights();
    }

    private void InitArchitecture(){
        int[] dimension = new int[NUM_LAYERS];

        dimension[0] = dataDimension;

        for(int i = 1; i < (NUM_LAYERS - 1)/2; i++) {
            dimension[i] = hiddenDimension[i-1];
            dimension[NUM_LAYERS - i -1] = hiddenDimension[i-1];
        }

        dimension[NUM_LAYERS-1] = dataDimension;

        LOG.verbose("Dimensions " + Arrays.toString(dimension));

        for(int i = 1; i < NUM_LAYERS; i++){
            networkWeights.weight[i-1] = new double[dimension[i-1]][dimension[i]];
            ActiveConnections[i-1] = new boolean[dimension[i-1]][dimension[i]];
            RMSprop.weight[i-1] = new double[dimension[i-1]][dimension[i]];
            batchGradient.weight[i-1] = new double[dimension[i-1]][dimension[i]];

            networkWeights.bias[i-1] = new double[dimension[i]];
            RMSprop.bias[i-1] = new double[dimension[i]];
            batchGradient.bias[i-1] = new double[dimension[i]];
        }
    }

    //this uses Kaiming (aka He aka MSRA) init as a placeholder, tha paper uses "pre-training" to determine the initial weights
    private void InitWeights(){
        Random random = RandomFactory.DEFAULT.getRandom();
        for(int layer = 0; layer < NUM_LAYERS - 1; layer++){
            int maxInDimension = ActiveConnections[layer].length;
            int maxOutDimension = ActiveConnections[layer][0].length;
            for (int inDimension = 0; inDimension < maxInDimension; inDimension++){
                for(int outDimension = 0; outDimension < maxOutDimension; outDimension++){
                    if(ActiveConnections[layer][inDimension][outDimension]){
                        networkWeights.weight[layer][inDimension][outDimension] = random.nextGaussian() * Math.sqrt(2.0 / maxInDimension);
                    }
                }
            }
        }
    }

    @Override
    public Double forward(V input) {
        double[] inputArray = input.toArray();

        //first layer
        double[] hiddenLayer1 = VMath.transposeTimes(networkWeights.weight[0], inputArray);
        VMath.plusEquals(hiddenLayer1, networkWeights.bias[0]);
        NetworkMathHelper.sigmoid(hiddenLayer1);

        //second layer
        double[] hiddenLayer2 = VMath.transposeTimes(networkWeights.weight[1], hiddenLayer1);
        VMath.plusEquals(hiddenLayer2, networkWeights.bias[1]);
        hiddenLayer2 = NetworkMathHelper.ReLu(hiddenLayer2);

        //third layer - decoder starts here
        double[] hiddenLayer3 = VMath.transposeTimes(networkWeights.weight[2], hiddenLayer2);
        VMath.plusEquals(hiddenLayer3, networkWeights.bias[2]);
        hiddenLayer3 = NetworkMathHelper.ReLu(hiddenLayer3);

        //fourth layer
        double[] output = VMath.transposeTimes(networkWeights.weight[3], hiddenLayer3);
        VMath.plusEquals(output, networkWeights.bias[3]);
        NetworkMathHelper.sigmoid(output);

        //Reproduction error
        double[] error = VMath.minus(output, inputArray);
        double sse = VMath.squareSum(error);
        return sse;
    }
    @Override
    Logging getLog() {
        return LOG;
    }

    @Override
    public NetworkWeights getGradient(V input) {
        //forward pass
        double[] inputArray = input.toArray();

        //first layer
        double[] hiddenLayer1 = VMath.transposeTimes(networkWeights.weight[0], inputArray);
        VMath.plusEquals(hiddenLayer1, networkWeights.bias[0]);
        double[] sigmoidedLayer1 = NetworkMathHelper.sigmoid(hiddenLayer1);

        //second layer
        double[] hiddenLayer2 = VMath.transposeTimes(networkWeights.weight[1], sigmoidedLayer1);
        VMath.plusEquals(hiddenLayer2, networkWeights.bias[1]);
        double[] reluLayer2 = NetworkMathHelper.ReLu(hiddenLayer2);

        //third layer - decoder starts here
        double[] hiddenLayer3 = VMath.transposeTimes(networkWeights.weight[2], reluLayer2);
        VMath.plusEquals(hiddenLayer3, networkWeights.bias[2]);
        double[] reluLayer3 = NetworkMathHelper.ReLu(hiddenLayer3);

        //fourth layer
        double[] output = VMath.transposeTimes(networkWeights.weight[3], reluLayer3);
        VMath.plusEquals(output, networkWeights.bias[3]);
        double[] sigmoidedOutput = NetworkMathHelper.sigmoid(output);

        //Reproduction error
        double[] error = VMath.minus(sigmoidedOutput, inputArray);
        double sse = VMath.squareSum(error);


        NetworkWeights gradient = NetworkWeights.init(NUM_LAYERS);
        cumulativeTrainingError += sse;

        //backward propagation
        double[] scoreGradient = VMath.times(error, 2);

        double[] l4SigmoidLocalGradient = VMath.plus(VMath.times(sigmoidedOutput,-1), 1);
        gradient.bias[3] = VMath.times(l4SigmoidLocalGradient, scoreGradient);
        gradient.weight[3] = NetworkMathHelper.RandLayerWeightGradient(reluLayer3, gradient.bias[3], ActiveConnections[3]);
        double[] l4inputGradient = VMath.times(networkWeights.weight[3], gradient.bias[3]);

        gradient.bias[2] = NetworkMathHelper.ReLuGradient(hiddenLayer3, l4inputGradient);
        gradient.weight[2] = NetworkMathHelper.RandLayerWeightGradient(reluLayer2, gradient.bias[2], ActiveConnections[2]);
        double[] l3inputGradient = VMath.times(networkWeights.weight[2], gradient.bias[2]);

        gradient.bias[1] = NetworkMathHelper.ReLuGradient(hiddenLayer2, l3inputGradient);
        gradient.weight[1] = NetworkMathHelper.RandLayerWeightGradient(sigmoidedLayer1, gradient.bias[1], ActiveConnections[1]);
        double[] l2inputGradient = VMath.times(networkWeights.weight[1], gradient.bias[1]);


        double[] l1SigmoidLocalGradient = NetworkMathHelper.sigmoidGradient(sigmoidedLayer1);
        gradient.bias[0] = VMath.times(l1SigmoidLocalGradient, l2inputGradient);
        gradient.weight[0] = NetworkMathHelper.RandLayerWeightGradient(inputArray, gradient.bias[0], ActiveConnections[0]);

        return gradient;
    }
}
