package de.fhdw.knn.network.activation;

/**
 * @see ActivationFunction#SIN
 */
class SinActivationFunction implements ActivationFunction {

    /**
     * Die ActivationFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link ActivationFunction} erzeugt werden
     */
    SinActivationFunction() {
    }

    @Override
    public double calc(double input) {
        return Math.sin(input);
    }

    @Override
    public double derived(double input, double calc) {
        return Math.cos(input);
    }

}
