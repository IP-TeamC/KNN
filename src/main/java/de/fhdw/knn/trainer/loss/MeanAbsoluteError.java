package de.fhdw.knn.trainer.loss;

public class MeanAbsoluteError implements LossFunction {

    @Override
    public double loss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            double error = expected[i] - predicted[i];
            sum += Math.abs(error);
        }
        return sum / expected.length;
    }

    @Override
    public double derivedLoss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum += expected[i] < predicted[i] ? 1 : -1;
        }
        return sum / expected.length;
    }

}
