package elki.clustering.neighborhood;

import elki.clustering.kmeans.AbstractKMeans;
import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.clustering.neighborhood.helper.ClosedNeighborhoodSetGenerator;
import elki.clustering.neighborhood.helper.MutualNeighborClosedNeighborhoodSetGenerator;
import elki.clustering.neighborhood.model.CNSrepresentor;
import elki.data.Clustering;
import elki.data.DoubleVector;
import elki.data.NumberVector;
import elki.data.model.KMeansModel;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDs;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.distance.NumberVectorDistance;
import elki.logging.Logging;
import elki.logging.statistics.Duration;
import elki.math.linearalgebra.VMath;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.ClassParameter;

import java.util.*;

import static elki.clustering.neighborhood.helper.ClosedNeighborhoodSetGenerator.CNS_GENERATOR_ID;

/**
 * Improves {@link FastKMeansCP} by not recompute <code>cns.avg * cns.size</code>  of all CNS every Iteration but instead uses the precomputed <code>cns.sum</code>.
 * @bug Currently not always converging.
 */
public class FasterKMeansCP<V extends NumberVector> extends AbstractKMeans<V, KMeansModel> {

    private double EPSILON = Math.pow(10, -10);

    private static final Logging LOG = Logging.getLogger(FasterKMeansCP.class);
    private final ClosedNeighborhoodSetGenerator<V> closedNeighborhoodSetGenerator;


    public FasterKMeansCP(int kCluster, int maxiter, KMeansInitialization initializer, ClosedNeighborhoodSetGenerator<V> closedNeighborhoodSetGenerator){
        super(kCluster, maxiter, initializer);
        this.closedNeighborhoodSetGenerator = closedNeighborhoodSetGenerator;
    }

    @Override
    public Clustering<KMeansModel> run(Relation<V> rel){
        Instance instance = new Instance(rel, distance, initialMeans(rel));
        instance.run(maxiter);
        return instance.buildResult();
    }

    protected DBIDs[] getCNS(Relation<V> relation){
        Duration cnsTime = LOG.newDuration(closedNeighborhoodSetGenerator.getClass().getName() + ".time").begin();
        DBIDs[] dbids = closedNeighborhoodSetGenerator.getClosedNeighborhoods(relation);
        LOG.statistics(cnsTime.end());
        return dbids;
    }

    @Override
    protected Logging getLogger() {
        return LOG;
    }

    protected class Instance extends AbstractKMeans.Instance {

        protected DBIDs[] CNSs;
        protected CNSrepresentor[] cnsRepresentors;
        protected List<List<CNSrepresentor>> CnsClusters;
        protected Map<CNSrepresentor, Integer> cnsAssignment;


        /**
         * Constructor.
         *
         * @param relation Relation to process
         * @param df Distance function
         * @param means    Initial mean
         */
        public Instance(Relation<V> relation, NumberVectorDistance<?> df, double[][] means) {
            super(relation, df, means);
            CNSs = getCNS(relation);
            cnsRepresentors = initalizeCNSrepresentors(CNSs);
            CnsClusters = new ArrayList<>(k);
            for(int i = 0; i < k; i++){
                CnsClusters.add(new ArrayList<>());
            }
            cnsAssignment = new HashMap<>(CNSs.length);

        }

        @Override
        protected int iterate(int iteration) {
            means = iteration == 1 ? means : weightedMeans(CnsClusters, means);
            return assignToNearestCluster(cnsRepresentors, means);
        }

        @Override
        protected Logging getLogger() {
            return LOG;
        }

        /**
         * Creates a representative for each closed neighborhood set.
         * @param closedNeighborhoodSets closed neighborhood sets to operate on
         * @return representative consisting of mean and sizer of set
         */
        private CNSrepresentor[] initalizeCNSrepresentors(DBIDs[] closedNeighborhoodSets){

            int dim = RelationUtil.dimensionality(relation);

            CNSrepresentor[] representors = new CNSrepresentor[closedNeighborhoodSets.length];

            for(int currentCNSindex = 0; currentCNSindex < closedNeighborhoodSets.length; currentCNSindex++  ){
                double[] sum = new double[dim];
                int currentCNSsize =  closedNeighborhoodSets[currentCNSindex].size();
                for(DBIDIter element = closedNeighborhoodSets[currentCNSindex].iter(); element.valid(); element.advance()){
                    VMath.plusEquals(sum, relation.get(element).toArray());
                }
                double[] avg = VMath.times(sum, 1.0 / currentCNSsize);
                double[] roundedSum = VMath.times(avg, currentCNSsize);
                double[] diff = VMath.minus(sum, roundedSum);
                if(Arrays.stream(diff).sum() > 0){
                    LOG.verbose("Difference for sum and rounded sum");
                }
                CNSrepresentor rep =  new CNSrepresentor(avg, roundedSum, currentCNSsize, closedNeighborhoodSets[currentCNSindex]);

                double[] sumN = rep.elementSum;
                double[] slowSum = VMath.times(rep.cnsMean, rep.size);

                if (!Arrays.equals(sumN,slowSum)){
                    double[] diffN = VMath.minus(sum, slowSum);
                    LOG.verbose("Difference of "+ Arrays.stream(diffN).sum());
                }


                representors[currentCNSindex] = rep;


            }

            return representors;
        }

        protected double[][] weightedMeans(List<List<CNSrepresentor>> clusters, double[][]means){
            final int k = means.length;
            double[][] newMeans = new double[k][];
            for(int clusterIndex = 0; clusterIndex < k; clusterIndex++){
                List<CNSrepresentor> currentCluster = clusters.get(clusterIndex);
                if(currentCluster.size() == 0){
                    newMeans[clusterIndex] = means[clusterIndex];
                    continue;
                }

                CNSrepresentor firstCNS = currentCluster.get(0);
                int amountElements = firstCNS.size;

                double[] slowSum = VMath.times(firstCNS.cnsMean, firstCNS.size);
                double[] sum = firstCNS.elementSum;
                //double[] sum = slowSum;

                if (!Arrays.equals(firstCNS.elementSum, slowSum)){
                    double[] diff = VMath.minus(firstCNS.elementSum, slowSum);
                    LOG.verbose("Difference of "+ Arrays.stream(diff).sum());
                }

                for(int i = 1; i < currentCluster.size(); i++ ){
                    CNSrepresentor currentCNS = currentCluster.get(i);

                    double[] cnsSum = currentCNS.elementSum;
                    amountElements += currentCNS.size;

                    slowSum = VMath.times(currentCNS.cnsMean, currentCNS.size);


                    VMath.plusEquals(sum, cnsSum );
                    //VMath.plusEquals(sum, slowSum );

                    if (!Arrays.equals(cnsSum, slowSum)){
                        double[] diff = VMath.minus(cnsSum, slowSum);
                        LOG.verbose("Difference of "+ Arrays.stream(diff).sum());
                    }
                }
                newMeans[clusterIndex] = VMath.times(sum, 1.0/ amountElements);
            }
            return newMeans;
        }

        protected int assignToNearestCluster(CNSrepresentor[] representatives, double[][] clusterMeans) {
            int changed = 0;

            for(List<CNSrepresentor> cluster : CnsClusters){
                cluster.clear();
            }

            for(CNSrepresentor representative: representatives) {
                NumberVector cnsMean = DoubleVector.wrap(representative.cnsMean);

                int minIndex = cnsAssignment.getOrDefault(representative, 0);
                double minDist = distance.distance(cnsMean, DoubleVector.wrap(clusterMeans[minIndex]));


                for (int i = 0; i < k; i++) {
                    double dist = distance.distance(cnsMean, DoubleVector.wrap(clusterMeans[i]));
                    double diff = minDist - dist;
                    if (diff > EPSILON) {
                        LOG.verbose("Improvement of " + diff);
                        minIndex = i;
                        minDist = dist;
                    }
                }
                varsum[minIndex] += isSquared ? minDist : (minDist * minDist);
                CnsClusters.get(minIndex).add(representative);
                Integer oldIndex = cnsAssignment.put(representative, minIndex);

                if( !((Integer)minIndex).equals(oldIndex) ) {
                    changed += representative.size;
                    //LOG.verbose("old index: " + oldIndex + ", new Index" + minIndex);
                }
            }
            return changed;
        }

        @Override
        public Clustering<KMeansModel> buildResult() {
            for(int i = 0; i < CnsClusters.size(); i++){
                for(CNSrepresentor cns : CnsClusters.get(i)){
                    clusters.get(i).addDBIDs(cns.cnsElements);
                }
            }
            return super.buildResult();
        }
    }

    public static class Par<V extends NumberVector> extends AbstractKMeans.Par<V> {
        protected ClosedNeighborhoodSetGenerator<V> closedNeighborhoodSetGenerator;

        @Override
        public void configure(Parameterization config){
            super.configure(config);

            new ClassParameter<ClosedNeighborhoodSetGenerator<V>>(CNS_GENERATOR_ID, ClosedNeighborhoodSetGenerator.class, MutualNeighborClosedNeighborhoodSetGenerator.class)
                    .grab(config, x -> closedNeighborhoodSetGenerator = x);
        }

        @Override
        public FasterKMeansCP<V> make() {
            return new FasterKMeansCP<>( k, maxiter, initializer, closedNeighborhoodSetGenerator);
        }
    }

}
