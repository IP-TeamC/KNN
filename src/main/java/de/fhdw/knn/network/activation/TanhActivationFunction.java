package de.fhdw.knn.network.activation;

/**
 * @see ActivationFunction#TANH
 */
class TanhActivationFunction implements ActivationFunction {

    /**
     * Die ActivationFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link ActivationFunction} erzeugt werden
     */
    TanhActivationFunction() {
    }

    @Override
    public double calc(double input) {
        return Math.tanh(input);
    }

    @Override
    public double derived(double input, double calc) {
        return 1 - (calc * calc);
    }
}
