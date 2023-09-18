package elki.outlier.autoencoder;

import elki.data.NumberVector;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.ids.DBIDIter;
import elki.database.relation.DoubleRelation;
import elki.database.relation.MaterializedDoubleRelation;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.logging.Logging;
import elki.math.DoubleMinMax;
import elki.math.MeanVariance;
import elki.math.linearalgebra.VMath;
import elki.outlier.OutlierAlgorithm;
import elki.outlier.autoencoder.networks.AutoencoderLayers7;
import elki.outlier.autoencoder.networks.TrainableNetwork;
import elki.result.outlier.BasicOutlierScoreMeta;
import elki.result.outlier.OutlierResult;
import elki.result.outlier.OutlierScoreMeta;
import elki.utilities.ensemble.EnsembleVoting;
import elki.utilities.ensemble.EnsembleVotingMedian;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.RandomParameter;
import elki.utilities.random.RandomFactory;

import java.util.ArrayList;
import java.util.Random;

public class Autoencoder<V extends NumberVector> implements OutlierAlgorithm {

    private static final Logging LOG = Logging.getLogger(Autoencoder.class);

    /**
     * Ratio of per-layer dimension decrease in autoencoder.
     */
    protected double alpha;
    protected int iterations;
    protected double rho;
    protected double learningRate;
    protected int numberNetworks;
    protected double maxSizeFraction;
    protected double adaptiveSizeRate;
    protected double weightDecay;

    protected Random random;

    public Autoencoder(double alpha, int iterations, double rho, double learningRate, int numberNetworks, double maxSizeFraction, double adaptiveSizeRate, double weightDecay, RandomFactory random) {
        super();
        this.alpha = alpha;
        this.iterations = iterations;
        this.rho = rho;
        this.learningRate = learningRate;
        this.numberNetworks = numberNetworks;
        this.maxSizeFraction = maxSizeFraction;
        this.adaptiveSizeRate = adaptiveSizeRate;
        this.weightDecay = weightDecay;
        this.random = random.getSingleThreadedRandom();
    }

    public OutlierResult run(Relation<V> relation) {
        int dim = RelationUtil.dimensionality(relation);

        ArrayList<TrainableNetwork<V, Double>> autoencoders = new ArrayList<>(numberNetworks);

        for (int i = 0; i < numberNetworks; i++) {
            TrainableNetwork<V, Double> network = new AutoencoderLayers7<>(alpha, dim, random);
            network.train(relation, rho, learningRate, iterations, adaptiveSizeRate, maxSizeFraction, weightDecay);
            autoencoders.add(network);
        }

        double[][] scores = new double[numberNetworks][relation.size()];
        for (int i = 0; i < numberNetworks; i++) {

            MeanVariance meanVariance = new MeanVariance();

            int j = 0;
            for (DBIDIter iter = relation.iterDBIDs(); iter.valid(); iter.advance()) {
                double score = autoencoders.get(i).forward(relation.get(iter));
                scores[i][j] = score;
                meanVariance.put(score);

                j++;
            }
            //normalize per ensemble component
            VMath.timesEquals(scores[i], 1.0 / meanVariance.getSampleStddev());

            MeanVariance assertion = new MeanVariance();
            assertion.put(scores[i]);
            assert Math.abs(assertion.getSampleStddev() - 1.0) < Double.MIN_VALUE;
        }

        scores = VMath.transpose(scores);


        DoubleMinMax minMax = new DoubleMinMax();
        EnsembleVoting voting = new EnsembleVotingMedian(0.25);

        DoubleRelation rel = new MaterializedDoubleRelation("autoencoder-outlier", relation.getDBIDs());

        int i = 0;
        for (DBIDIter iter = relation.iterDBIDs(); iter.valid(); iter.advance()) {
            double outlierscore = voting.combine(scores[i]);

            //LOG.verbose("Ensemble scores: " + Arrays.toString(scores[i]) + ", median: " + outlierscore);
            rel.set(iter, outlierscore);
            minMax.put(outlierscore);

            i++;
        }

        OutlierScoreMeta meta = new BasicOutlierScoreMeta(minMax.getMin(), minMax.getMax(), 0, Double.POSITIVE_INFINITY);
        return new OutlierResult(meta, rel);
    }

    @Override
    public TypeInformation[] getInputTypeRestriction() {
        return TypeUtil.array(TypeUtil.NUMBER_VECTOR_FIELD);
    }

    public static class Par<V> implements Parameterizer {

        protected double alpha;

        public static final OptionID ALPHA_ID = new OptionID("autoencoder.alpha", "The ratio at wich dimension gets decreased in the encoder");

        protected int iteration;
        public static final OptionID ITERATION_ID = new OptionID("autoencoder.iteration", "Iteration that the autoencoder is trained.");

        protected double rho;
        public static final OptionID RHO_ID = new OptionID("autoencoder.rho", "Rho parameter for RMSprop optimizer");

        protected double learningRate;
        public static final OptionID LEARNING_RATE_ID = new OptionID("autoencoder.learningRate", "Learning rate for RMSprop optimizer");

        protected int numberAutoencoders;
        public static final OptionID NUMBER_AUTOENCODERS_ID = new OptionID("autoencoder.numberNetworks", "Number of autoencoders in ensemble");

        protected double adaptiveRate;
        public static final OptionID ADAPTIVE_RATE_ID = new OptionID("autoencoder.adaptiveRate", "The rate at which the samples per iteraton increases");

        protected double maxFraction;
        public static OptionID MAX_FRACTION_ID = new OptionID("autoencoder.maxFraction", "The fraction of total samples used for each component of the ensemble.");


        protected double weight_decay;
        public static OptionID WEIGHT_DECAY_ID = new OptionID("autoencoder.weightDecay", "Weight for weight decay.");

        protected TrainableNetwork<V, Double> network;
        public static OptionID NETWORK_ID = new OptionID("autoencoder.network", "The specific autoencoder to use.");


        protected RandomFactory random;

        public static OptionID RANDOM_SEED = new OptionID("autoencoder.seed", "The seed used to generate the RandNet architectures");

        @Override
        public void configure(Parameterization config) {
            new DoubleParameter(ALPHA_ID).addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE)
                    .grab(config, x -> alpha = x);

            new IntParameter(ITERATION_ID).addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT)
                    .grab(config, x -> iteration = x);

            new DoubleParameter(RHO_ID).addConstraint(CommonConstraints.LESS_EQUAL_ONE_DOUBLE)
                    .addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE)
                    .grab(config, x -> rho = x);

            new DoubleParameter(LEARNING_RATE_ID).addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE)
                    .grab(config, x -> learningRate = x);

            new IntParameter(NUMBER_AUTOENCODERS_ID).addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT)
                    .grab(config, x -> numberAutoencoders = x);

            new DoubleParameter(ADAPTIVE_RATE_ID).addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE)
                    .grab(config, x -> adaptiveRate = x);

            new DoubleParameter(MAX_FRACTION_ID).addConstraint(CommonConstraints.LESS_EQUAL_ONE_DOUBLE)
                    .addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE)
                    .grab(config, x -> maxFraction = x);

            new DoubleParameter(WEIGHT_DECAY_ID).addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE)
                    .grab(config, x -> weight_decay = x);

            new RandomParameter(RANDOM_SEED).grab(config, x -> random = x);


        }


        @Override
        public Object make() {
            return new Autoencoder<>(alpha, iteration, rho, learningRate, numberAutoencoders, maxFraction, adaptiveRate, weight_decay, random);
        }
    }
}
