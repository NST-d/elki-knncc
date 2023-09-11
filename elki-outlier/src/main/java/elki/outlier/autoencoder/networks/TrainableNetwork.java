package elki.outlier.autoencoder.networks;

import elki.data.type.TypeInformation;
import elki.database.relation.Relation;

public interface TrainableNetwork<I, O> {
    O forward(I input);
    void train(Relation<I> trainingsData, double rho, double learningRate, int maxIterations, double adaptiveFactor, double maxSize, double regularizer);

}
