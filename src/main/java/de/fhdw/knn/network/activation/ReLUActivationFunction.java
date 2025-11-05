package de.fhdw.knn.network.activation;

public class ReLUActivationFunction implements ActivationFunction {

    public double calc(double input) {
        return input > 0 ? input : 0;
    }

    @Override
    public double derived(double input, double calc) {
        return input > 0 ? 1 : 0;
    }
}
