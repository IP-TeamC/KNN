package de.fhdw.knn.network.activation;

public class SigmoidActivationFunction implements ActivationFunction {

    public static final SigmoidActivationFunction DEFAULT = new SigmoidActivationFunction();

    public double calc(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derived(double input) {
        double sigmoid = calc(input);
        return sigmoid * (1 - sigmoid);
    }
}
