package de.fhdw.knn.trainer.learningrate;

public class SoftstartLearningRate implements LearningRateFunction {

    private final double learningRate;
    private final double softness;
    private final int softEpochs;

    public SoftstartLearningRate(double learningRate, double softness, int softEpochs) {
        this.learningRate = learningRate;
        this.softness = softness;
        this.softEpochs = softEpochs;
    }

    @Override
    public double calc(int epoch, double previousLoss) {
        if (epoch < softEpochs) {
            return learningRate * softness;
        }
        return learningRate;
    }

}
