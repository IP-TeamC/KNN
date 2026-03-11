package de.fhdw.knn.trainer.loss;

/**
 * Verlustfunktion Binary Cross Entropy Loss für die Verwendung bei binären Klassifikationsproblemen
 * oder bei Klassifikationsproblemen mit voneinander unabhängigen Klassen.
 */
public class BinaryCrossEntropyLoss implements LossFunction {

    /**
     * Die LossFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link LossFunction} erzeugt werden
     */
    BinaryCrossEntropyLoss() {
    }

    /**
     * Konstante zur Vermeidung von NaN-Werten (Logarithmus von 0)
     */
    private static final Double DELTA_0 = 0.000_000_000_000_1;
    /**
     * Konstante zur Vermeidung von NaN-Werten (Logarithmus von 0)
     */
    private static final Double DELTA_1 = 1.000_000_000_000_1;

    @Override
    public double loss(double[] expected, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum -= expected[i] * Math.log(predicted[i] + DELTA_0) + (1 - expected[i]) * Math.log(DELTA_1 - predicted[i]);
        }
        return sum / expected.length;
    }

    @Override
    public double derivedLoss(double[] expected, double[] predicted, int neuron) {
        final double expectedNeuron = expected[neuron];
        final double predictedNeuron = predicted[neuron];
        return (1 - expectedNeuron) / (DELTA_1 - predictedNeuron) - expectedNeuron / (predictedNeuron + DELTA_0);

    }

}
