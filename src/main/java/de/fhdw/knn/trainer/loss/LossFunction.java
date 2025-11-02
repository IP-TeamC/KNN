package de.fhdw.knn.trainer.loss;

public interface LossFunction {

    LossFunction MEAN_SQUARED_ERROR = new MeanSquaredError();

    double loss(double[] expected, double[] predicted);

    double derivedLoss(double[] expected, double[] predicted);

    double totalLoss(double[][] expected, double[][] predicted);

}
