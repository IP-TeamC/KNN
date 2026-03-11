package de.fhdw.knn.network.activation;

/**
 * @see ActivationFunction#LINEAR
 */
class LinearActivationFunction implements ActivationFunction {

    /**
     * Die ActivationFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link ActivationFunction} erzeugt werden
     */
    LinearActivationFunction() {
    }

    @Override
    public double calc(double input) {
        return input;
    }

    @Override
    public double derived(double input, double calc) {
        return 1;
    }
}
