package de.fhdw.knn.network.activation;

/**
 * @see ActivationFunction#SIGMOID
 */
class SigmoidActivationFunction implements ActivationFunction {

    @Override
    public double calc(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derived(double input, double calc) {
        return calc * (1 - calc);
    }
}
