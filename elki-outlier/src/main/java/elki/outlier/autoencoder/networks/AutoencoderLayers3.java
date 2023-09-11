package elki.outlier.autoencoder.networks;

import elki.data.NumberVector;
import elki.data.VectorUtil;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.ids.*;
import elki.database.relation.Relation;
import elki.logging.Logging;
import elki.math.MathUtil;
import elki.math.linearalgebra.VMath;
import elki.utilities.random.RandomFactory;

import java.util.Random;


public class AutoencoderLayers3 <V extends NumberVector> implements TrainableNetwork<V, Double> {

    private static final Logging LOG = Logging.getLogger(AutoencoderLayers3.class);
    protected int LAYER_MIN_DIMENSION = 3;
    protected int NUM_LAYERS = 3;
    protected final int dataDimension;
    protected final int hiddenDimension;
    protected double[][][] NetworkWeights = new double[2][][];

    protected double[][][] RMSprop = new double[2][][];

    protected double[][][] IterationGradient = new double[2][][];
    protected boolean[][][] ActiveConnections = new boolean[2][][];

    public AutoencoderLayers3(double alpha, int dataDimension){
        this.dataDimension = dataDimension;
        this.hiddenDimension = Math.max(LAYER_MIN_DIMENSION, (int) (alpha * dataDimension));

        //Encoder: data dimension -> hidden dimension
        NetworkWeights[0] = new double[dataDimension][hiddenDimension];
        ActiveConnections[0] = new boolean[dataDimension][hiddenDimension];
        RMSprop[0] = new double[dataDimension][hiddenDimension];
        IterationGradient[0] =  new double[dataDimension][hiddenDimension];
        //Decoder: hidden dimension -> data dimension
        NetworkWeights[1] = new double[hiddenDimension][dataDimension];
        ActiveConnections[1] = new boolean[hiddenDimension][dataDimension];
        RMSprop[1] = new double[hiddenDimension][dataDimension];
        IterationGradient[1] = new double[hiddenDimension][dataDimension];


        for( int layer = 0; layer < (NUM_LAYERS - 1)/2; layer++){
            int[] activeConnectionsIndices = sampleInts(dataDimension*hiddenDimension, dataDimension*hiddenDimension);
            for(int index : activeConnectionsIndices){

                //Encoder
                ActiveConnections[layer][index /hiddenDimension][index % hiddenDimension] = true;
                //Decoder
                ActiveConnections[NUM_LAYERS -2 -layer][index % hiddenDimension][index /hiddenDimension] = true;
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
    public void train(Relation<V> trainingData, double rho, double learningRate, int maxIterations, double adaptiveFactor, double initialSize, double weightDecay){

        int n = trainingData.size() / 10;
        double s = n;


        ArrayModifiableDBIDs dbids = DBIDUtil.newArray(trainingData.getDBIDs());
        DBIDUtil.randomShuffle(dbids, RandomFactory.DEFAULT);


        for (int iteration = 0; iteration < maxIterations; iteration++){
            DBIDArrayIter iter = dbids.iter();

            for(int trainingSample = 0; trainingSample < (int) s; trainingSample++ ){

                //forward pass
                V input = trainingData.get(iter);
                double[] inputArray = input.toArray();

                //first layer
                double[] hiddenLayer = VMath.transposeTimes(NetworkWeights[0], inputArray);
                double[] sigmoidedHiddenLayer = new double[hiddenDimension];
                //maybe move in VMath??
                for( int i = 0; i < hiddenDimension; i++){
                    sigmoidedHiddenLayer[i] = sigmoid(hiddenLayer[i]);
                }

                //second layer
                double[] output = VMath.transposeTimes(NetworkWeights[1], sigmoidedHiddenLayer);
                double[] sigmoidedOutput = new double[dataDimension];

                for(int i = 0; i < dataDimension; i++){
                    sigmoidedOutput[i] = sigmoid(output[i]);
                }

                //Reproduction error
                double[] error = VMath.minus(sigmoidedOutput, inputArray);
                double sse = VMath.squareSum(error);

                LOG.verbose("Training error: " + sse);

                //backward propagation
                double[] scoreGradient = VMath.times(error, 2);
                double[] l2SigmoidLocalGradient = VMath.plus(VMath.times(sigmoidedOutput,-1), 1);
                double[] l2SigmoidDownstreamGradient = VMath.times(l2SigmoidLocalGradient, scoreGradient);

                double[][] l2WeightGradient = VMath.timesTranspose(l2SigmoidDownstreamGradient, sigmoidedHiddenLayer);
                for(int i = 0; i < ActiveConnections[0].length; i++){
                    for(int j = 0; j < ActiveConnections[0][0].length; j++){
                        if(!ActiveConnections[0][i][j]){
                            l2WeightGradient[i][j] = 0;
                        }
                    }
                }

                double[] l2inputGradient = VMath.times(NetworkWeights[1], l2SigmoidDownstreamGradient);

                double[] l1SigmoidLocalGradient = VMath.plus(VMath.times(sigmoidedHiddenLayer, -1), 1);
                double[] l1SigmoidDownstreamGradient = VMath.times(l1SigmoidLocalGradient, l2inputGradient);

                double[][] l1WeightGradient = VMath.transpose(VMath.timesTranspose(l1SigmoidDownstreamGradient, inputArray));

                for(int i = 0; i < l1WeightGradient.length; i++){
                    for(int j = 0; j < l1WeightGradient[0].length; j++){
                        if(!ActiveConnections[1][j][i]){
                            l1WeightGradient[i][j] = 0;
                        }
                    }
                }

                l2WeightGradient = VMath.transpose(l2WeightGradient);

                //RMSprop optimization, maybe modularize? complicated due dependence on meta-information regarding network architecture
                RMSprop[0] = VMath.plusTimes(VMath.times(RMSprop[0], rho), VMath.hadamard(l1WeightGradient, l1WeightGradient), 1- rho);
                RMSprop[1] = VMath.plusTimes(VMath.times(RMSprop[1], rho), VMath.hadamard(l2WeightGradient, l2WeightGradient), 1- rho);

                VMath.plusEquals(IterationGradient[0], VMath.hadamard(l1WeightGradient, NetworkMathHelper.divideEquals(learningRate, VMath.root(RMSprop[0], 0.00001) )));
                VMath.plusEquals(IterationGradient[1], VMath.hadamard(l2WeightGradient, NetworkMathHelper.divideEquals(learningRate, VMath.root(RMSprop[1], 0.00001) )));

                iter.advance();
            }

            VMath.minusEquals(NetworkWeights[0], IterationGradient[0]);
            VMath.minusEquals(NetworkWeights[1], IterationGradient[1]);

            //Reset batch gradient after iteration
            for (int i = 0; i < IterationGradient[0].length; i++){
                for (int j = 0; j < IterationGradient[0][0].length; j++){
                    IterationGradient[0][i][j] = 0;
                    IterationGradient[1][j][i] = 0;
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
        double[] hiddenLayer = VMath.transposeTimes(NetworkWeights[0], inputArray);
        double[] sigmoidedHiddenLayer = new double[hiddenDimension];
        //maybe move in VMath??
        for( int i = 0; i < hiddenDimension; i++){
            sigmoidedHiddenLayer[i] = sigmoid(hiddenLayer[i]);
        }

        //second layer
        double[] output = VMath.transposeTimes(NetworkWeights[1], sigmoidedHiddenLayer);
        double[] sigmoidedOutput = new double[dataDimension];
        for(int i = 0; i < dataDimension; i++){
            sigmoidedOutput[i] = sigmoid(output[i]);
        }

        //Reproduction error
        double[] error = VMath.minus(sigmoidedOutput, inputArray);
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
