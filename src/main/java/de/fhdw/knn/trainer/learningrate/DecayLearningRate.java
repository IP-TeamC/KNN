package de.fhdw.knn.trainer.learningrate;

/**
 * Verringert die Learning Rate exponentiell mit jeder Epoche
 */
public class DecayLearningRate implements LearningRateFunction {

    /**
     * Learning Rate zu Beginn
     */
    private double learningRate;
    /**
     * Faktor mit dem die Learning Rate jede Epoche multipliziert wird
     */
    private final double decay;

    /**
     * Definiert eine exponentiell fallende Learning Rate
     *
     * @param learningRate Initiale Learning Rate
     * @param decay        Faktor, mit dem die Learning Rate jede Epoche multipliziert wird
     * @see DecayLearningRate
     */
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
