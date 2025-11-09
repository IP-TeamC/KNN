package de.fhdw.knn.network.activation;

public class SoftplusActivationFunction implements ActivationFunction {

    @Override
    public double calc(double input) {
        return Math.log(1 + Math.exp(input));
    }

    @Override
    public double derived(double input, double calc) {
        return 1 / (1 + Math.exp(-input));
    }
}
