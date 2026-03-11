package de.fhdw.knn.network.activation;

/**
 * @see ActivationFunction#RELU
 */
class ReLUActivationFunction implements ActivationFunction {

    /**
     * Die ActivationFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link ActivationFunction} erzeugt werden
     */
    ReLUActivationFunction() {
    }

    @Override
    public double calc(double input) {
        return input > 0 ? input : 0;
    }

    @Override
    public double derived(double input, double calc) {
        return input > 0 ? 1 : 0;
    }
}
