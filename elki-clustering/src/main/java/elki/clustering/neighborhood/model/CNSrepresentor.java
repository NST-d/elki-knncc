package elki.clustering.neighborhood.model;

import elki.database.ids.DBIDs;

public class CNSrepresentor {
    public final int size;
    public final double[] cnsMean;

    public final double[] elementSum;

    public final DBIDs cnsElements;

    public CNSrepresentor(double[] cnsMean, double[] elementSum, int size, DBIDs cnsElements) {
        this.cnsMean = cnsMean;
        this.size = size;
        this.cnsElements = cnsElements;
        this.elementSum = elementSum; //currently unstable
    }

}
