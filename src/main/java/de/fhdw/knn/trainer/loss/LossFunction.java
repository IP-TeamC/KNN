package de.fhdw.knn.trainer.loss;

public interface LossFunction {

    double loss(double[] expected, double[] predicted);

    double derivedLoss(double[] expected, double[] predicted);

}
