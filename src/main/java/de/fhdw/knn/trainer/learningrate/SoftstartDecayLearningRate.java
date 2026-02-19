package de.fhdw.knn.trainer.learningrate;

/**
 * Verringert die Learning Rate für einige Epochen zu Beginn um einen konstanten Faktor und
 * verringert die Learning Rate danach exponentiell mit jeder Epoche.
 */
public class SoftstartDecayLearningRate implements LearningRateFunction {

    private double learningRate;
    private final double softness;
    private final int softEpochs;
    private final double decay;

    /**
     * @param learningRate Learning Rate nach Ablauf von softEpochs (anschließend geringer durch Decay)
     * @param softness     Faktor, mit dem die Learning Rate während softEpochs multipliziert wird
     * @param softEpochs   Anzahl der Epochen, für die die Learning Rate mit softness multipliziert wird
     * @param decay        Faktor, mit dem die Learning Rate nach softEpochs multipliziert wird
     * @see SoftstartDecayLearningRate
     */
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
