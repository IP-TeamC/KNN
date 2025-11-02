package de.fhdw.knn.network.activation;

public class SigmoidActivationFunction implements ActivationFunction {

    public double calc(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derived(double input, double calc) {
        return calc * (1 - calc);
    }
}
