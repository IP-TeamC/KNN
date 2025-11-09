package de.fhdw.knn.trainer.learningrate;

public class ConstantLearningRate implements LearningRateFunction {

    private final double learningRate;

    public ConstantLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public double calc(int epoch, double previousLoss) {
        return learningRate;
    }

}
