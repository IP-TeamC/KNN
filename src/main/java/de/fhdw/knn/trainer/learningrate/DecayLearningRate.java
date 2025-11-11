package de.fhdw.knn.trainer.learningrate;

public class DecayLearningRate implements LearningRateFunction {

    private double learningRate;
    private final double decay;

    public DecayLearningRate(double learningRate, double decay) {
        this.learningRate = learningRate;
        this.decay = decay;
    }

    @Override
    public double calc(int epoch, double previousLoss) {
        learningRate = learningRate * decay;
        return learningRate;
    }

}
