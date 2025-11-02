package de.fhdw.knn.trainer.optimization;

public interface OptimizationFunction {

    void init();

    Adjustments compute(double[] input, double[] output);

}
