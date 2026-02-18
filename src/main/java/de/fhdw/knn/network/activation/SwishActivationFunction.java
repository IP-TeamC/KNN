package de.fhdw.knn.network.activation;

class SwishActivationFunction implements ActivationFunction {

    @Override
    public double calc(double input) {
        return input / (1 + Math.exp(-input));
    }

    @Override
    public double derived(double input, double calc) {
        return calc + (1 - calc) / (1 + Math.exp(-input));
    }
}
