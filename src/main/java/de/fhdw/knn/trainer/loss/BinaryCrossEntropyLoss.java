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
    public double derivedLoss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum += (1 - expected[i]) / ((1 - predicted[i]) * Math.log(10) + DELTA) - expected[i] / (predicted[i] * Math.log(10) + DELTA);
        }
        return sum;
    }

}
