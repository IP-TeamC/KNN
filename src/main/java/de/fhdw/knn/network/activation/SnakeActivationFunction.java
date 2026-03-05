package de.fhdw.knn.network.activation;

/**
 * @see ActivationFunction#SNAKE
 */
class SnakeActivationFunction implements ActivationFunction {

    @Override
    public double calc(double input) {
        double sin = Math.sin(input);
        return input + sin * sin;
    }

    @Override
    public double derived(double input, double calc) {
        return 1 + Math.sin(2 * input);
    }

}
