package de.fhdw.knn.network.activation;

/**
 * @see ActivationFunction#LINEAR
 */
class LinearActivationFunction implements ActivationFunction {

    @Override
    public double calc(double input) {
        return input;
    }

    @Override
    public double derived(double input, double calc) {
        return 1;
    }
}
