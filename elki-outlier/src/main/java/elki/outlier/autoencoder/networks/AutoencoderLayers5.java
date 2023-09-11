package elki.outlier.autoencoder.networks;

import elki.data.NumberVector;
import elki.database.ids.ArrayModifiableDBIDs;
import elki.database.ids.DBIDArrayIter;
import elki.database.ids.DBIDUtil;
import elki.database.relation.Relation;
import elki.logging.Logging;
import elki.math.linearalgebra.VMath;
import elki.utilities.random.RandomFactory;

import java.util.Random;

public class AutoencoderLayers5<V extends NumberVector> implements TrainableNetwork<V, Double>{

    private static final Logging LOG = Logging.getLogger(AutoencoderLayers3.class);
    protected int LAYER_MIN_DIMENSION = 3;
    protected int NUM_LAYERS = 5;
    protected final int dataDimension;
    protected final int[] hiddenDimension = new int[(NUM_LAYERS -1)/2];
    protected double[][][] NetworkWeights = new double[NUM_LAYERS - 1][][];

    protected double[][][] RMSprop = new double[NUM_LAYERS - 1][][];

    protected double[][][] IterationGradient = new double[NUM_LAYERS - 1][][];
    protected boolean[][][] ActiveConnections = new boolean[NUM_LAYERS - 1][][];

    public AutoencoderLayers5(double alpha, int dataDimension){
        this.dataDimension = dataDimension;
        this.hiddenDimension[0] = Math.max(LAYER_MIN_DIMENSION, (int) (alpha * dataDimension));
        this.hiddenDimension[1] = Math.max(LAYER_MIN_DIMENSION, (int) (alpha * hiddenDimension[0]));

        //Encoder: data dimension -> hidden dimension[0] -> hiddendimension[1]
        NetworkWeights[0] = new double[dataDimension][hiddenDimension[0]];
        ActiveConnections[0] = new boolean[dataDimension][hiddenDimension[0]];
        RMSprop[0] = new double[dataDimension][hiddenDimension[0]];
        IterationGradient[0] =  new double[dataDimension][hiddenDimension[0]];

        NetworkWeights[1] = new double[hiddenDimension[0]][hiddenDimension[1]];
        ActiveConnections[1] = new boolean[hiddenDimension[0]][hiddenDimension[1]];
        RMSprop[1] = new double[hiddenDimension[0]][hiddenDimension[1]];
        IterationGradient[1] =  new double[hiddenDimension[0]][hiddenDimension[1]];

        //Decoder: hidden dimension[1] -> hidden dimension[0] -> data dimension
        NetworkWeights[2] = new double[hiddenDimension[1]][hiddenDimension[0]];
        ActiveConnections[2] = new boolean[hiddenDimension[1]][hiddenDimension[0]];
        RMSprop[2] = new double[hiddenDimension[1]][hiddenDimension[0]];
        IterationGradient[2] =  new double[hiddenDimension[1]][hiddenDimension[0]];

        NetworkWeights[3] = new double[hiddenDimension[0]][dataDimension];
        ActiveConnections[3] = new boolean[hiddenDimension[0]][dataDimension];
        RMSprop[3] = new double[hiddenDimension[0]][dataDimension];
        IterationGradient[3] = new double[hiddenDimension[0]][dataDimension];

        for( int layer = 0; layer < (NUM_LAYERS - 1)/2; layer++){
            int in = NetworkWeights[layer].length;
            int out = NetworkWeights[layer][0].length;

            int[] activeConnectionsIndices = sampleInts( in * out, in * out);
            for(int index : activeConnectionsIndices){

                //Encoder
                ActiveConnections[layer][index /out][index % out] = true;
                //Decoder
                ActiveConnections[NUM_LAYERS -2 -layer][index % out][index /out] = true;
            }
        }

        InitWeights();
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
                        NetworkWeights[layer][inDimension][outDimension] = random.nextGaussian() * Math.sqrt(2.0 / maxInDimension);
                    }
                }
            }
        }
    }

    @Override
    public void train(Relation<V> trainingData, double rho, double learningRate, int maxIterations, double initialSize, double adaptiveFactor, double weightDecay){

        int n =(int) (trainingData.size() * initialSize);
        double s = n;

        ArrayModifiableDBIDs dbids = DBIDUtil.newArray(trainingData.getDBIDs());
        DBIDUtil.randomShuffle(dbids, RandomFactory.DEFAULT);

        for (int iteration = 0; iteration < maxIterations; iteration++){
            DBIDArrayIter iter = dbids.iter();

            double runningCumulativeError = 0.0;
            for(int trainingSample = 0; trainingSample < (int) s; trainingSample++ ){

                //forward pass
                V input = trainingData.get(iter);
                double[] inputArray = input.toArray();

                //first layer
                double[] hiddenLayer1 = VMath.transposeTimes(NetworkWeights[0], inputArray);
                double[] sigmoidedLayer1 = new double[hiddenDimension[0]];

                //maybe move in VMath??
                for( int i = 0; i < hiddenDimension[0]; i++){
                    sigmoidedLayer1[i] =sigmoid(hiddenLayer1[i]);
                }

                //second layer
                double[] hiddenLayer2 = VMath.transposeTimes(NetworkWeights[1], sigmoidedLayer1);
                double[] reluLayer2 = new double[hiddenDimension[1]];

                for(int i = 0; i < hiddenDimension[1]; i++){
                    reluLayer2[i] = Math.max(0, hiddenLayer2[i]);
                }

                //third layer - decoder starts here
                double[] hiddenLayer3 = VMath.transposeTimes(NetworkWeights[2], reluLayer2);
                double[] reluLayer3 = new double[hiddenDimension[0]];

                for(int i = 0; i < hiddenDimension[0]; i++){
                    reluLayer3[i] = Math.max(0, hiddenLayer3[i]);
                }

                //fourth layer
                double[] output = VMath.transposeTimes(NetworkWeights[3], reluLayer3);
                double[] sigmoidedOutput = new double[dataDimension];

                for(int i = 0; i < dataDimension; i++){
                    sigmoidedOutput[i] = sigmoid(output[i]);
                }

                //Reproduction error
                double[] error = VMath.minus(sigmoidedOutput, inputArray);
                double sse = VMath.squareSum(error);

                runningCumulativeError += sse;

                //backward propagation
                double[] scoreGradient = VMath.times(error, 2);
                double[] l4SigmoidLocalGradient = VMath.plus(VMath.times(sigmoidedOutput,-1), 1);
                double[] l4SigmoidDownstreamGradient = VMath.times(l4SigmoidLocalGradient, scoreGradient);

                double[][] l4WeightGradient = NetworkMathHelper.RandLayerWeightGradient(reluLayer3, l4SigmoidDownstreamGradient, ActiveConnections[3]);
                double[] l4inputGradient = VMath.times(NetworkWeights[3], l4SigmoidDownstreamGradient);

                double[] l3ReluGradient = NetworkMathHelper.ReLuGradient(hiddenLayer3, l4inputGradient);
                double[][] l3WeightGradient = NetworkMathHelper.RandLayerWeightGradient(reluLayer2, l3ReluGradient, ActiveConnections[2]);
                double[] l3inputGradient = VMath.times(NetworkWeights[2], l3ReluGradient);

                double[] l2ReluGradient = NetworkMathHelper.ReLuGradient(hiddenLayer2, l3inputGradient);
                double[][] l2WeightGradient = NetworkMathHelper.RandLayerWeightGradient(sigmoidedLayer1, l2ReluGradient, ActiveConnections[1]);
                double[] l2inputGradient = VMath.times(NetworkWeights[1], l2ReluGradient);


                double[] l1SigmoidLocalGradient = VMath.plus(VMath.times(sigmoidedLayer1, -1), 1);
                double[] l1SigmoidDownstreamGradient = VMath.times(l1SigmoidLocalGradient, l2inputGradient);

                double[][] l1WeightGradient = NetworkMathHelper.RandLayerWeightGradient(inputArray, l1SigmoidDownstreamGradient, ActiveConnections[0]);

                l1WeightGradient = VMath.transpose(l1WeightGradient);
                l2WeightGradient = VMath.transpose(l2WeightGradient);
                l3WeightGradient = VMath.transpose(l3WeightGradient);
                l4WeightGradient = VMath.transpose(l4WeightGradient);

                //RMSprop optimization, maybe modularize? complicated due dependence on meta-information regarding network architecture
                RMSprop[0] = VMath.plusTimes(VMath.times(RMSprop[0], rho), VMath.hadamard(l1WeightGradient, l1WeightGradient), 1- rho);
                RMSprop[1] = VMath.plusTimes(VMath.times(RMSprop[1], rho), VMath.hadamard(l2WeightGradient, l2WeightGradient), 1- rho);
                RMSprop[2] = VMath.plusTimes(VMath.times(RMSprop[2], rho), VMath.hadamard(l3WeightGradient, l3WeightGradient), 1- rho);
                RMSprop[3] = VMath.plusTimes(VMath.times(RMSprop[3], rho), VMath.hadamard(l4WeightGradient, l4WeightGradient), 1- rho);

                VMath.plusEquals(IterationGradient[0], VMath.hadamard(l1WeightGradient, VMath.divideEquals(learningRate, VMath.root(RMSprop[0], 0.00001) )));
                VMath.plusEquals(IterationGradient[1], VMath.hadamard(l2WeightGradient, VMath.divideEquals(learningRate, VMath.root(RMSprop[1], 0.00001) )));
                VMath.plusEquals(IterationGradient[2], VMath.hadamard(l3WeightGradient, VMath.divideEquals(learningRate, VMath.root(RMSprop[2], 0.00001) )));
                VMath.plusEquals(IterationGradient[3], VMath.hadamard(l4WeightGradient, VMath.divideEquals(learningRate, VMath.root(RMSprop[3], 0.00001) )));

                iter.advance();
            }

            LOG.verbose("Training error in iteration " + iteration + " with " + (int) s+ " samples: "+ runningCumulativeError/(int)s);

            //TODO find out why this is not working with minus
            VMath.plusTimesEquals(NetworkWeights[0], IterationGradient[0], 1.0/(int)s);
            VMath.plusTimesEquals(NetworkWeights[1], IterationGradient[1], 1.0/(int)s);
            VMath.plusTimesEquals(NetworkWeights[2], IterationGradient[2], 1.0/(int)s);
            VMath.plusTimesEquals(NetworkWeights[3], IterationGradient[3], 1.0/(int)s);

            //weight decay
            VMath.minusTimesEquals(NetworkWeights[0], NetworkWeights[0], learningRate * weightDecay);
            VMath.minusTimesEquals(NetworkWeights[1], NetworkWeights[1], learningRate * weightDecay);
            VMath.minusTimesEquals(NetworkWeights[2], NetworkWeights[2], learningRate * weightDecay);
            VMath.minusTimesEquals(NetworkWeights[3], NetworkWeights[3], learningRate * weightDecay);

            //Reset batch gradient after iteration
            for (int layer = 0; layer < (NUM_LAYERS-1)/2; layer++) {
                for (int i = 0; i < IterationGradient[layer].length; i++) {
                    for (int j = 0; j < IterationGradient[layer][0].length; j++) {
                        IterationGradient[layer][i][j] = 0;
                        IterationGradient[NUM_LAYERS - layer - 2][j][i] = 0;
                    }
                }
            }
            s *= adaptiveFactor;
            s = Math.min(s, trainingData.size());
        }
    }

    @Override
    public Double forward(V input) {
        double[] inputArray = input.toArray();
        //first layer
        double[] hiddenLayer1 = VMath.transposeTimes(NetworkWeights[0], inputArray);
        //maybe move in VMath??
        for( int i = 0; i < hiddenDimension[0]; i++){
            hiddenLayer1[i] = sigmoid(hiddenLayer1[i]);
        }

        //second layer
        double[] hiddenLayer2 = VMath.transposeTimes(NetworkWeights[1], hiddenLayer1);

        for(int i = 0; i < hiddenDimension[1]; i++){
            hiddenLayer2[i] = Math.max(0, hiddenLayer2[i]);
        }

        //third layer - decoder starts here
        double[] hiddenLayer3 = VMath.transposeTimes(NetworkWeights[2], hiddenLayer2);
        for(int i = 0; i < hiddenDimension[0]; i++){
            hiddenLayer3[i] = Math.max(0, hiddenLayer3[i]);
        }

        //fourth layer
        double[] output = VMath.transposeTimes(NetworkWeights[3], hiddenLayer3);
        for(int i = 0; i < dataDimension; i++){
            output[i] = sigmoid(output[i]);
        }

        //Reproduction error
        double[] error = VMath.minus(output, inputArray);
        double sse = VMath.squareSum(error);
        return sse;
    }

    private double sigmoid(double a){
        return 1.0/ (1.0 + Math.exp(a));
    }

    /**
     * Samples n ints in [0,max-1] with replacement
     * @param max
     * @param n
     * @return sampled ints, multiple occurrences removed
     */
    private int[] sampleInts(int max, int n){
        Random random = RandomFactory.DEFAULT.getRandom();
        return random.ints(0, max).limit(n).distinct().toArray();
    }
}
