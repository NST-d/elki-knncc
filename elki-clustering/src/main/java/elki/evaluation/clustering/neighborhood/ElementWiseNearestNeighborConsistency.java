package elki.evaluation.clustering.neighborhood;

import elki.data.Cluster;
import elki.data.Clustering;
import elki.database.Database;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.WritableDoubleDataStore;
import elki.database.ids.*;
import elki.database.query.QueryBuilder;
import elki.database.query.knn.KNNSearcher;
import elki.database.relation.MaterializedDoubleRelation;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.distance.minkowski.EuclideanDistance;
import elki.evaluation.Evaluator;
import elki.result.EvaluationResult;
import elki.result.Metadata;
import elki.result.ResultUtil;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;

import java.util.List;

public class ElementWiseNearestNeighborConsistency<O> implements Evaluator {

    private final Distance<? super O> distance;

    protected final int k;
    protected final int kPlus;

    public ElementWiseNearestNeighborConsistency(Distance<? super O> distance, int k) {
        super();
        this.distance = distance;
        this.k = k;
        this.kPlus = k+1;
    }

    /**
     * Calculate fractional kNN consistency for all datapoints and average them.
     * @param clustering Cluster to evaluate
     * @param relation datapoints
     * @return fractional kNN consistency
     */
    public double evaluateClustering(Clustering<?> clustering, Relation<O> relation){

        WritableDoubleDataStore elementKNNConsistency = DataStoreFactory.FACTORY.makeDoubleStorage(relation.getDBIDs(), DataStoreFactory.HINT_DB, 0.);
        List<? extends Cluster<?>> clusters = clustering.getAllClusters();
        KNNSearcher<DBIDRef> knnQuery = new QueryBuilder<>(relation, distance).precomputed().kNNByDBID(kPlus);

        double kNNc = 0.0;
        for(Cluster<?> cluster: clusters){
            DBIDs clusterIDs = cluster.getIDs();
            for(DBIDIter clusterElement = clusterIDs.iter(); clusterElement.valid(); clusterElement.advance()){
                int neighborsInCluster = 0;
                KNNList neighbors = knnQuery.getKNN(clusterElement,kPlus);
                for (DBIDIter neighbor = neighbors.iter(); neighbor.valid(); neighbor.advance()){
                    if(DBIDUtil.equal(neighbor , clusterElement)){
                        continue;
                    }
                    if(clusterIDs.contains(neighbor)){
                        neighborsInCluster++;
                    }
                }
                double fractionalKNNc = (double)neighborsInCluster/(neighbors.size()-1); //neighbors size can be greater than kPlus for elements with same distance
                elementKNNConsistency.put(clusterElement, fractionalKNNc);
                kNNc += fractionalKNNc;
            }
        }

        kNNc =  kNNc/ relation.size();

        EvaluationResult ev = EvaluationResult.findOrCreate(clustering, "Clustering Evaluation");
        EvaluationResult.MeasurementGroup g = ev.findOrCreateGroup("Distance-based");
        g.addMeasure("EW " + k + "-NN Consistency", kNNc, 0, 1., false);
        if(!Metadata.hierarchyOf(clustering).addChild(ev)) {
            Metadata.of(ev).notifyChanged();
        }
        Metadata.hierarchyOf(clustering).addChild(new MaterializedDoubleRelation("EW "+ k + "-NN Consistency", relation.getDBIDs(), elementKNNConsistency));

        return kNNc;
    }

    @Override
    public void processNewResult(Object result) {
        List<Clustering<?>> clusters = Clustering.getClusteringResults(result);
        if(clusters.isEmpty()){
            return;
        }
        Database db = ResultUtil.findDatabase(result);
        assert db != null;
        Relation<O> relation = db.getRelation(distance.getInputTypeRestriction());
        for(Clustering<?> cluster : clusters){
            evaluateClustering(cluster, relation);
        }


    }

    public static class Par<O> implements Parameterizer{

        public static final OptionID DISTANCE_ID = new OptionID("knnc.distance", "Distance function to use for computing the knnc.");

        public static final OptionID NUMBER_K = new OptionID("knnc.k", "Number of Neighbors checked.");
        private Distance<? super O> distance;
        private int k;

        @Override
        public void configure(Parameterization config) {
            new ObjectParameter<Distance<? super O>>(DISTANCE_ID, Distance.class, EuclideanDistance.class).grab(config, x-> distance = x);
            new IntParameter(NUMBER_K)
                    .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT)
                    .grab(config, x -> k = x);
        }

        @Override
        public ElementWiseNearestNeighborConsistency<O> make() {
            return new ElementWiseNearestNeighborConsistency<>(distance, k);
        }
    }
}
