package de.fhdw.knn.network.activation;

/**
 * Sigmoid-Aktivierungsfunktion
 *
 * @see ActivationFunction#SIGMOID
 */
class SigmoidActivationFunction implements ActivationFunction {

    /**
     * Die ActivationFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link ActivationFunction} erzeugt werden
     */
    SigmoidActivationFunction() {
    }

    @Override
    public double calc(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derived(double input, double calc) {
        return calc * (1 - calc);
    }
}
