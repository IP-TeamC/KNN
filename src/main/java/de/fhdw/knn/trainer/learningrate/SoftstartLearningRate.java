package de.fhdw.knn.trainer.learningrate;

/**
 * Verringert die Learning Rate für einige Epochen zu Beginn um einen konstanten Faktor
 */
public class SoftstartLearningRate implements LearningRateFunction {

    /**
     * Learning Rate zu Beginn
     */
    private final double learningRate;
    /**
     * Faktor, mit dem die Learning Rate während softEpochs multipliziert ist (konstante Learning Rate während softEpochs)
     */
    private final double softness;
    /**
     * Anzahl Epochen, während der die Learning Rate konstant durch softness verändert wird
     */
    private final int softEpochs;

    /**
     * Definiert eine zunächst abgeschwächte Learning Rate
     *
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
