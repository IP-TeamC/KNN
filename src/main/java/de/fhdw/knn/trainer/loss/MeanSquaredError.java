package de.fhdw.knn.trainer.loss;

public class MeanSquaredError implements LossFunction {

    public static final MeanSquaredError DEFAULT = new MeanSquaredError();

    @Override
    public double loss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            //System.out.println("Expected: " + expected[i] + ", Predicted: " + predicted[i]);
            double error = expected[i] - predicted[i];
            sum += error * error;
        }
        return sum / expected.length;
    }

    @Override
    public double derivedLoss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            double error = expected[i] - predicted[i];
            sum -= 2 * error;
        }
        return sum / expected.length;
    }
}
