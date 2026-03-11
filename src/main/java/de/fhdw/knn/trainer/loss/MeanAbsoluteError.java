package de.fhdw.knn.trainer.loss;

/**
 * Verlustfunktion Mean Absolute Error gibt die mittlere absolute Abweichung an.
 * Diese Verlustfunktion ist im Gegensatz zu {@link MeanSquaredError} weniger anfällig gegenüber Ausreißern.
 */
public class MeanAbsoluteError implements LossFunction {

    /**
     * Die LossFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link LossFunction} erzeugt werden
     */
    MeanAbsoluteError() {
    }

    @Override
    public double loss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            double error = expected[i] - predicted[i];
            sum += Math.abs(error);
        }
        return sum / expected.length;
    }

    @Override
    public double derivedLoss(double[] expected, double[] predicted, int neuron) {
        double abs = (double) 1 / expected.length;
        return expected[neuron] < predicted[neuron] ? abs : -abs;
    }

}
