package elki.outlier.autoencoder;

import elki.data.DoubleVector;
import org.junit.Test;

import elki.database.Database;
import elki.outlier.AbstractOutlierAlgorithmTest;
import elki.result.outlier.OutlierResult;
import elki.utilities.ELKIBuilder;


public class AutoEncoderTest extends AbstractOutlierAlgorithmTest {
    @Test
    public void testAutoEncoder(){
        Database db = makeSimpleDatabase(UNITTEST + "outlier-3d-3clusters.ascii", 960);
        OutlierResult result = new ELKIBuilder<Autoencoder<DoubleVector>>(Autoencoder.class)
                .with(Autoencoder.Par.WEIGHT_DECAY_ID, 0.1)
                .with(Autoencoder.Par.LEARNING_RATE_ID, 0.001)
                .with(Autoencoder.Par.RANDOM_SEED, 0)
                .build().autorun(db);
        assertAUC(db, "Noise", result, 0.7066);
        assertSingleScore(result, 939, 1.7822);
    }
}
