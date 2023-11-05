package elki.outlier.density;

import elki.data.NumberVector;
import elki.database.ids.DBIDIter;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.math.MathUtil;
import elki.math.MeanVariance;
import net.jafama.FastMath;

public class TreeUtil {
    /**
     * Returns the average path length of an unsuccessful search.
     * Returns 0 if the value is less than or equal to 1.
     *
     * @param n Depth
     * @return Expected average
     */
    public static double c(double n) {
        return n <= 1.0 ? 0 : 2 * (FastMath.log(n - 1) + MathUtil.EULERMASCHERONI) - (2. * (n - 1) / n);
    }

    public static double[] perDimensionStddev(Relation<? extends NumberVector> relation) {
        int dimensionality = RelationUtil.dimensionality(relation);

        MeanVariance[] meanVariances = MeanVariance.newArray(dimensionality);

        for (DBIDIter iter = relation.iterDBIDs(); iter.valid(); iter.advance()) {
            double[] v = relation.get(iter).toArray();
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

    /**
     * Check if all entries of an array are zero or close to zero.
     *
     * @param array array to check
     * @param eps closeness allowed to zero
     * @return true if all entries are zero, false otherwise.
     */
    public static boolean allZeros(double[] array, double eps) {
        for (double o : array) {
            if (Math.abs(o) > eps) {
                return false;
            }
        }
        return true;
    }


}
