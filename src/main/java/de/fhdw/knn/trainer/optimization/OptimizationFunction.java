package de.fhdw.knn.trainer.optimization;

public interface OptimizationFunction {

    void init();

    void epoch(int epoch, double previousLoss);

    Adjustments compute(double[] input, double[] output, int batchSize);

}
