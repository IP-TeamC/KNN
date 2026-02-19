package de.fhdw.knn.trainer.learningrate;

/**
 * Verwendet eine über alle Epochen konstante Learning Rate
 */
public class ConstantLearningRate implements LearningRateFunction {

    /**
     * Konstante Learning Rate
     */
    private final double learningRate;

    /**
     * @see ConstantLearningRate
     */
    public ConstantLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public double calc(int epoch, double previousLoss) {
        return learningRate;
    }

}
