package de.fhdw.knn.network.activation;

import java.util.List;

public interface ActivationFunction {

    ActivationFunction SIGMOID = new SigmoidActivationFunction();
    ActivationFunction RELU = new ReLUActivationFunction();
    ActivationFunction LINEAR = new LinearActivationFunction();
    ActivationFunction TANH = new TanhActivationFunction();
    ActivationFunction SWISH = new SwishActivationFunction();
    ActivationFunction SOFTPLUS = new SoftplusActivationFunction();

    List<ActivationFunction> FUNCTIONS = List.of(SIGMOID, RELU, LINEAR, TANH, SWISH, SOFTPLUS);

    double calc(double input);

    double derived(double input, double calc);

}