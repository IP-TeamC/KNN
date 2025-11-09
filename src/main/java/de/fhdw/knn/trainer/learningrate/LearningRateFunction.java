package de.fhdw.knn.trainer.learningrate;

public interface LearningRateFunction {

    double calc(int epoch, double previousLoss);

}
