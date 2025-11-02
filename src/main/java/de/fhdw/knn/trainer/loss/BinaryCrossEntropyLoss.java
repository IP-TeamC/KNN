package de.fhdw.knn.trainer.loss;

public class BinaryCrossEntropyLoss implements LossFunction {

    @Override
    public double loss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum -= expected[i] * Math.log(predicted[i]) + (1 - expected[i]) * Math.log(1 - predicted[i]);
        }
        return sum;
    }

    @Override
    public double derivedLoss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum += (1 - expected[i]) / ((1 - predicted[i]) * Math.log(10) + 0.000_000_000_1) - expected[i] / (predicted[i] * Math.log(10) + 0.000_000_000_1);
        }
        return sum;
    }

}
