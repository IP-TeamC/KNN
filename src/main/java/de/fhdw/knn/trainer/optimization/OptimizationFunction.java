package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.Network;

public interface OptimizationFunction {

    void epoch(int epoch, double previousLoss);

    Adjustments compute(Network network, double[] input, double[] output, int batchSize);

}
