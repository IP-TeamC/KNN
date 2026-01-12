package de.fhdw.knn.trainer.loss;

public interface LossFunction {

    LossFunction MEAN_SQUARED_ERROR = new MeanSquaredError();
    LossFunction CROSS_ENTROPY_LOSS = new BinaryCrossEntropyLoss();
    LossFunction MEAN_ABSOLUTE_ERROR = new MeanAbsoluteError();

    double loss(double[] expected, double[] predicted);

    double derivedLoss(double[] expected, double[] predicted, int neuron);

    default double totalLoss(double[][] expected, double[][] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum += loss(expected[i], predicted[i]);
        }
        return sum / expected.length;
    }

}
