package de.fhdw.knn.network.activation;

/**
 * Softplus-Aktivierungsfunktion
 *
 * @see ActivationFunction#SOFTPLUS
 */
class SoftplusActivationFunction implements ActivationFunction {

    /**
     * Die ActivationFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link ActivationFunction} erzeugt werden
     */
    SoftplusActivationFunction() {
    }

    @Override
    public double calc(double input) {
        return Math.log(1 + Math.exp(input));
    }

    @Override
    public double derived(double input, double calc) {
        return 1 / (1 + Math.exp(-input));
    }
}
