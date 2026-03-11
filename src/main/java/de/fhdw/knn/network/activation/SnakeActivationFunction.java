package de.fhdw.knn.network.activation;

/**
 * @see ActivationFunction#SNAKE
 */
class SnakeActivationFunction implements ActivationFunction {

    /**
     * Die ActivationFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link ActivationFunction} erzeugt werden
     */
    SnakeActivationFunction() {
    }

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
