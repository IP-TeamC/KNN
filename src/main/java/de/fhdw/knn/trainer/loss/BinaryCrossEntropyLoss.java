package de.fhdw.knn.trainer.loss;

public class BinaryCrossEntropyLoss implements LossFunction {

    public static final Double DELTA = 0.000_000_000_000_1;

    @Override
    public double loss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum -= expected[i] * Math.log(predicted[i] + DELTA) + (1 - expected[i]) * Math.log(1 - predicted[i] + DELTA);
        }
        return sum;
    }

    @Override
    public double derivedLoss(double[] expected, double[] predicted, int neuron) {
        return (1 - expected[neuron]) / ((1 - predicted[neuron]) + DELTA) - expected[neuron] / (predicted[neuron] + DELTA);
    }

}
