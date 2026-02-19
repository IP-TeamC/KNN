package de.fhdw.knn.trainer.loss;

/**
 * Verlustfunktion Mean Squared Error gibt die mittlere quadratische Abweichung an.
 * Diese Verlustfunktion ist im Gegensatz zu {@link MeanAbsoluteError} anfälliger gegenüber Ausreißer,
 * funktioniert aber häufig sehr gut und sollte bei Nicht-Klassifikationsproblemen wahrscheinlich bevorzugt werden.
 */
public class MeanSquaredError implements LossFunction {

    @Override
    public double loss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            double error = expected[i] - predicted[i];
            sum += error * error;
        }
        return sum / expected.length;
    }

    @Override
    public double derivedLoss(double[] expected, double[] predicted, int neuron) {
        return ((double) -2 / expected.length) * (expected[neuron] - predicted[neuron]);
    }

}
