package de.fhdw.knn.trainer.learningrate;

/**
 * Verringert die Learning Rate für einige Epochen zu Beginn um einen konstanten Faktor
 */
public class SoftstartLearningRate implements LearningRateFunction {

    private final double learningRate;
    private final double softness;
    private final int softEpochs;

    /**
     * @param learningRate Learning Rate nach Ablauf von softEpochs
     * @param softness     Faktor, mit dem die Learning Rate während softEpochs multipliziert wird
     * @param softEpochs   Anzahl der Epochen, für die die Learning Rate mit softness multipliziert wird
     * @see SoftstartLearningRate
     */
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
