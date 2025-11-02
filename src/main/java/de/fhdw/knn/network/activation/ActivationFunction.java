package de.fhdw.knn.network.activation;

public interface ActivationFunction {

    /*
    OutputFunction LINEAR = input -> input;
    OutputFunction BINARY_STEP = input -> input < 0 ? 0 : 1;
    OutputFunction SIGMOID = input -> 1 / (1 + Math.exp(-input));
    OutputFunction TANH = Math::tanh;
    OutputFunction RELU = input -> Math.max(0, input);
    OutputFunction LEAKY_RELU = input -> Math.max(0.1 * input, input);
    */
    // TODO OutputFunction SOFTMAX;

    double calc(double input);

    double derived(double input);

}
