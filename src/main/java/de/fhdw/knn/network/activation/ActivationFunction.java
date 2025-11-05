package de.fhdw.knn.network.activation;

public interface ActivationFunction {

    ActivationFunction SIGMOID = new SigmoidActivationFunction();
    ActivationFunction RELU = new ReLUActivationFunction();

    double calc(double input);

    double derived(double input, double calc);

}
