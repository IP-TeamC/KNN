package de.fhdw.knn.trainer.loss;

public class MeanSquaredError implements LossFunction {

    @Override
    public double loss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            double error = expected[i] - predicted[i];
            sum += error * error;
        }
        return sum / expected.length;
    }

    @Override
    public double derivedLoss(double[] expected, double[] predicted, int neuron) {
        return ((double) -2 / expected.length) * (expected[neuron] - predicted[neuron]);
    }

}
