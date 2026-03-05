package de.fhdw.knn.network.activation;

/**
 * @see ActivationFunction#SIN
 */
class SinActivationFunction implements ActivationFunction {

    @Override
    public double calc(double input) {
        return Math.sin(input);
    }

    @Override
    public double derived(double input, double calc) {
        return Math.cos(input);
    }

}
