package elki.outlier.density;

import elki.data.NumberVector;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableDoubleDataStore;
import elki.database.ids.*;
import elki.database.relation.DoubleRelation;
import elki.database.relation.MaterializedDoubleRelation;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.logging.Logging;
import elki.logging.progress.FiniteProgress;
import elki.logging.progress.StepProgress;
import elki.math.DoubleMinMax;
import elki.math.MathUtil;
import elki.math.MeanVariance;
import elki.outlier.OutlierAlgorithm;
import elki.result.outlier.BasicOutlierScoreMeta;
import elki.result.outlier.OutlierResult;
import elki.result.outlier.OutlierScoreMeta;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.RandomParameter;
import elki.utilities.random.RandomFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


import net.jafama.FastMath;

public class SCiForest implements OutlierAlgorithm {

    /**
     * Class logger
     */
    private static final Logging LOG = Logging.getLogger(SCiForest.class);


    private final int amountTrees;
    private int subsampleSize;
    private final int amountAttributes;
    private final int hyperplanesPerNode;
    private final RandomFactory random;

    public SCiForest(int amountTrees, int subsampleSize, int amountAttributes, int amountSampledHyperplanesPerNode, RandomFactory random) {
        this.amountTrees = amountTrees;
        this.subsampleSize = subsampleSize;
        this.amountAttributes = amountAttributes;
        this.hyperplanesPerNode = amountSampledHyperplanesPerNode;
        this.random = random;
    }

    public OutlierResult run(Relation<? extends NumberVector> relation) {

        if (relation.size() < subsampleSize) {
            subsampleSize = relation.size();
        }
        StepProgress stepProgress = LOG.isVerbose() ? new StepProgress(2) : null;
        LOG.beginStep(stepProgress, 1, "Generating SCiForest trees");
        FiniteProgress progess = LOG.isVerbose() ? new FiniteProgress("SCi forest construction", amountTrees, LOG) : null;

        final Random rnd = random.getSingleThreadedRandom();
        List<SciForestNode> trees = new ArrayList<>(amountTrees);
        SciForestBuilder builder = new SciForestBuilder(relation, rnd, amountAttributes, subsampleSize, hyperplanesPerNode);
        for (int t = 0; t < amountTrees; t++) {
            trees.add(builder.newTree());
            LOG.incrementProcessed(progess);
        }
        LOG.ensureCompleted(progess);
        LOG.beginStep(stepProgress, 2, "Computing forest scores");
        progess = LOG.isVerbose() ? new FiniteProgress("Forest scores", relation.size(), LOG) : null;
        WritableDoubleDataStore scores = DataStoreUtil.makeDoubleStorage(relation.getDBIDs(), DataStoreFactory.HINT_DB);
        DoubleMinMax minmax = new DoubleMinMax();
        final double f = -MathUtil.LOG2 / (trees.size() * TreeUtil.c(subsampleSize));
        // Iterate over all objects
        for (DBIDIter iter = relation.iterDBIDs(); iter.valid(); iter.advance()) {
            final NumberVector v = relation.get(iter);
            // Score against each tree:
            double avgPathLength = 0;
            for (SciForestNode tree : trees) {
                avgPathLength += isolationScore(tree, v, 0);
            }
            final double score = FastMath.exp(f * avgPathLength);
            if(Double.isNaN(score)){
                LOG.error("Isolation score is NaN.");
            }
            scores.putDouble(iter, score);
            minmax.put(score);
            LOG.incrementProcessed(progess);
        }
        LOG.ensureCompleted(progess);
        //this is buggy...
        LOG.ensureCompleted(stepProgress);

        // Wrap the result in the standard containers
        OutlierScoreMeta meta = new BasicOutlierScoreMeta(
                // Actually observed minimum and maximum values
                minmax.getMin(), minmax.getMax(),
                // Theoretical minimum and maximum: no variance to infinite variance
                0, Double.POSITIVE_INFINITY);
        DoubleRelation rel = new MaterializedDoubleRelation("SCiForest", relation.getDBIDs(), scores);
        return new OutlierResult(meta, rel);

    }

    protected double isolationScore(SciForestNode n, NumberVector v, int level) {
        if (n.externalNode) {
            return level + TreeUtil.c(n.size);
        }
        double y = n.hyperplane.f(v.toArray());
        if (y <= 0) {
            return isolationScore(n.right, v, level + y < n.upperLimit ? 1 : 0);
        } else {
            return isolationScore(n.left, v, y > n.lowerLimit ? 1 : 0);
        }
    }

    @Override
    public TypeInformation[] getInputTypeRestriction() {
        return TypeUtil.array(TypeUtil.NUMBER_VECTOR_FIELD);
    }

    protected static class SciForestBuilder {

        Relation<? extends NumberVector> relation;

        ArrayModifiableDBIDs dbids;
        DBIDArrayMIter iter;

        Random random;

        int amountAttributes;

        int dimensionality;

        int subsampleSize;

        int hyperplanesPerNode;


        protected SciForestBuilder(Relation<? extends NumberVector> relation, Random random, int amountAttributes, int subsampleSize, int hyperplanesPerNode) {
            this.relation = relation;
            this.random = random;
            this.amountAttributes = amountAttributes;
            this.dimensionality = RelationUtil.dimensionality(relation);
            this.dbids = DBIDUtil.newArray(relation.getDBIDs());
            this.subsampleSize = subsampleSize;
            this.hyperplanesPerNode = hyperplanesPerNode;
            this.iter = dbids.iter();
        }

        protected SciForestNode newTree() {
            int[] allIndices = IntStream.range(0, subsampleSize).toArray();
            return buildTree(allIndices);
        }

        /**
         * Builds a single Tree recursively
         *
         * @return Tree node
         */
        protected SciForestNode buildTree(int[] indices) {

            if (indices.length <= 2) {
                return new SciForestNode(null,
                        0,
                        0,
                        indices.length,
                        null,
                        null);
            } else {
                double[] stdDev = perDimensionStddev(indices);

                //can't construct a separating hyperplane if stdDev in all dimensions is zero
                if (TreeUtil.allZeros(stdDev, 0.00001)) {
                    return new SciForestNode(null,
                            0,
                            0,
                            indices.length,
                            null,
                            null);
                }

                Hyperplane plane = findBestHyperplane(indices, stdDev);

                ArrayList<Integer> lowerIndices = new ArrayList<>();
                ArrayList<Integer> higherIndices = new ArrayList<>();

                DoubleMinMax minMax = new DoubleMinMax();

                for (int i : indices) {
                    double value = plane.f(relation.get(iter.seek(i)).toArray());
                    minMax.put(value);
                    if (value >= 0) {
                        higherIndices.add(i);
                    } else {
                        lowerIndices.add(i);
                    }
                }

                int[] lowIndices = new int[lowerIndices.size()];
                Arrays.setAll(lowIndices, lowerIndices::get);
                int[] highIndices = new int[higherIndices.size()];
                Arrays.setAll(highIndices, higherIndices::get);
                double maxValueDifference = minMax.getDiff();
                return new SciForestNode(plane,
                        +maxValueDifference,
                        -maxValueDifference,
                        indices.length,
                        buildTree(lowIndices),
                        buildTree(highIndices));
            }

        }


        protected Hyperplane findBestHyperplane(int[] indices, double[] stdDev) {

            double bestSdGain = Double.NEGATIVE_INFINITY;
            Hyperplane bestHyperplane = null;


            for (int i = 0; i < hyperplanesPerNode; i++) {

                Hyperplane plane = sampleHyperplane(stdDev);

                double sdGain = setBestSeparatingPoint(plane, indices);
                if (sdGain > bestSdGain) {
                    bestSdGain = sdGain;
                    bestHyperplane = plane;
                } else if (bestHyperplane == null){
                    bestHyperplane = plane;
                }
            }

            return bestHyperplane;
        }

        /**
         * Finds the best separating point for a hyperplane according to sd gain, sets in hyperplane and returns the sdGain value.
         * If sdGain can't be calculated for any separating point, a random point is chosen which separates the elements.
         *
         * @param hyperplane Hyperplane for that the best separation point is set
         * @param indices    Indices of DBIDs of elements in the relation which are checked as seperation points
         * @return SD Gain of separating point that is selected
         */
        protected double setBestSeparatingPoint(Hyperplane hyperplane, int[] indices) {
            double[] hyperplaneProjection = new double[indices.length];
            MeanVariance allValueVariance = new MeanVariance();
            //project all points with hyperplane
            for (int i = 0; i < indices.length; i++) {
                double y = hyperplane.f(relation.get(iter.seek(indices[i])).toArray());
                allValueVariance.put(y);
                hyperplaneProjection[i] = y;
            }
            double allStdDev = allValueVariance.getSampleStddev();
            //find point which induces highest sd gain
            int bestSeperatingPointIndex = -1;
            double bestSeperatigValue = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < hyperplaneProjection.length; i++) {
                MeanVariance lowerVariance = new MeanVariance();
                MeanVariance higherVariance = new MeanVariance();
                double separatingPoint = hyperplaneProjection[i];
                for (int j = 0; j < indices.length; j++) {
                    if (hyperplaneProjection[j] < separatingPoint) {
                        lowerVariance.put(hyperplaneProjection[j]);
                    } else {
                        higherVariance.put(hyperplaneProjection[j]);
                    }
                }
                //can't calculate stdDev for 0 or 1 sample
                if (lowerVariance.getCount() <= 1 || higherVariance.getCount() <= 1) {
                    //but still use if hyperplane is separating and no hyperplane is found by now
                    if (lowerVariance.getCount() > 0 && higherVariance.getCount() > 0 && bestSeperatigValue == Double.NEGATIVE_INFINITY) {
                        bestSeperatingPointIndex = i;
                    }
                    continue;
                }
                double sdGain = sdGain(allStdDev, lowerVariance.getSampleStddev(), higherVariance.getSampleStddev());
                if (sdGain > bestSeperatigValue) {
                    bestSeperatigValue = sdGain;
                    bestSeperatingPointIndex = i;
                }
            }
            hyperplane.splitPoint = hyperplaneProjection[bestSeperatingPointIndex];
            return bestSeperatigValue;
        }

        protected double sdGain(double stdDevAll, double stdDevLow, double stdDevHigh) {
            return (stdDevAll - (stdDevLow + stdDevHigh) / 2) / stdDevAll;
        }

        protected Hyperplane sampleHyperplane(double[] stdDev) {

            //gets amountAttributes if there are enough separating dimensions, or all separating dimensions otherwise
            List<Integer> separatingDimensions = IntStream.range(0, dimensionality)
                    .filter(i -> Math.abs(stdDev[i]) > 0.00001)
                    .boxed()
                    .collect(Collectors.toCollection(ArrayList::new));

            Collections.shuffle(separatingDimensions);

            //select amountAttribute dimensions, or all if all < amountAttributes
            int[] dimensions = Arrays.copyOfRange(separatingDimensions.stream().mapToInt(i->i).toArray(), 0, Math.min(amountAttributes, separatingDimensions.size()));

            double[] coefficients = new double[dimensions.length];
            double[] stdDevInChosenDimension = new double[dimensions.length];

            for (int i = 0; i < dimensions.length; i++) {
                //random value uniform distributed in [-1, 1]
                coefficients[i] = 2 * (random.nextDouble() - 0.5);
                stdDevInChosenDimension[i] = stdDev[dimensions[i]];
            }

            return new Hyperplane(dimensions, coefficients, stdDevInChosenDimension, 0);
        }

        public double[] perDimensionStddev(int[] indices) {
            MeanVariance[] meanVariances = MeanVariance.newArray(dimensionality);

            for (int i : indices) {
                double[] v = relation.get(iter.seek(i)).toArray();
                for (int d = 0; d < dimensionality; d++) {
                    meanVariances[d].put(v[d]);
                }
            }

            double[] stdDev = new double[dimensionality];
            for (int d = 0; d < dimensionality; d++) {
                stdDev[d] = meanVariances[d].getSampleStddev();
            }
            return stdDev;
        }

    }

    protected static class SciForestNode {

        /**
         * Separating hyperplane.
         */
        Hyperplane hyperplane;

        double upperLimit;
        double lowerLimit;

        /**
         * Subtree size.
         */
        int size;

        /**
         * Left child, may be null.
         */
        SciForestNode left;

        /**
         * Right child, may be null.
         */
        SciForestNode right;

        boolean externalNode;

        public SciForestNode(Hyperplane hyperplane, double upperLimit, double lowerLimit, int size, SciForestNode left, SciForestNode right) {

            this.hyperplane = hyperplane;
            this.upperLimit = upperLimit;
            this.lowerLimit = lowerLimit;
            this.size = size;
            this.left = left;
            this.right = right;
            this.externalNode = left == null && right == null;
        }


    }

    protected static class Hyperplane {
        /**
         * Indices of dimensions used in Hyperplane.
         */
        int[] dimensions;

        /**
         * Hyperplane coefficients.
         */
        double[] coefficients;

        /**
         * Std dev in the corresponding dimension of the data.
         */
        double[] stddev;

        /**
         * Hyperplane split point.
         */
        double splitPoint;


        protected Hyperplane(int[] dimensions, double[] coefficents, double[] stddev, double splitPoint) {
            assert coefficents.length == stddev.length;
            this.dimensions = dimensions;
            this.coefficients = coefficents;
            this.stddev = stddev;
            this.splitPoint = splitPoint;
        }

        protected double f(double[] x) {
            double sum = 0;
            for (int i = 0; i < coefficients.length; i++) {
                sum += coefficients[i] * x[dimensions[i]] / stddev[i];
            }
            sum -= splitPoint;
            return sum;
        }

    }


    public static class Par implements Parameterizer {

        /**
         * Parameter for the number of trees
         */
        public static final OptionID NUM_TREES_ID = new OptionID("sciforest.numtrees", "Number of trees to use.");

        /**
         * Parameter for the sub sample size
         */
        public static final OptionID SUBSAMPLE_SIZE_ID = new OptionID("sciforest.subsample", "Subsampling size.");

        public static final OptionID AMOUNT_ATTRIBUTES_ID = new OptionID("sciforest.attributes", "Number of attributes used for hyperplanes.");

        public static final OptionID AMOUNT_HYPERPLANES_ID = new OptionID("sciforest.hyperplanes", "Number of sampled hyperplanes per tree node");

        /**
         * Parameter to specify the seed to initialize Random.
         */
        public static final OptionID SEED_ID = new OptionID("sciforest.seed", "The seed to use for initializing Random.");


        /**
         * Number of trees
         */
        protected int numTrees = 100;

        /**
         * Size of the sample set
         */
        protected int subsampleSize = 256;

        protected int attributes;

        protected int hyperplanes;
        /**
         * Random generator
         */
        protected RandomFactory rnd;

        @Override
        public void configure(Parameterization config) {
            new IntParameter(NUM_TREES_ID, 100) //
                    .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
                    .grab(config, x -> numTrees = x);
            new IntParameter(SUBSAMPLE_SIZE_ID, 256) //
                    .addConstraint(CommonConstraints.GREATER_THAN_ONE_INT) //
                    .grab(config, x -> subsampleSize = x);
            new IntParameter(AMOUNT_ATTRIBUTES_ID, 2)
                    .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT)
                    .grab(config, x -> attributes = x);
            new IntParameter(AMOUNT_HYPERPLANES_ID, 10)
                    .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT)
                    .grab(config, x -> hyperplanes = x);
            new RandomParameter(SEED_ID).grab(config, x -> this.rnd = x);
        }

        @Override
        public SCiForest make() {
            return new SCiForest(this.numTrees, this.subsampleSize, attributes, hyperplanes, this.rnd);
        }
    }
}
