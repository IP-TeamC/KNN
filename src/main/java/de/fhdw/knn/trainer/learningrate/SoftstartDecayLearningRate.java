package de.fhdw.knn.trainer.learningrate;

public class SoftstartDecayLearningRate implements LearningRateFunction {

    private double learningRate;
    private final double softness;
    private final int softEpochs;
    private final double decay;

    public SoftstartDecayLearningRate(double learningRate, double softness, int softEpochs, double decay) {
        this.learningRate = learningRate;
        this.softness = softness;
        this.softEpochs = softEpochs;
        this.decay = decay;
    }

    @Override
    public double calc(int epoch, double previousLoss) {
        if (epoch < softEpochs) {
            return learningRate * softness;
        }
        learningRate = learningRate * decay;
        return learningRate;
    }

}
