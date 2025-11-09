package de.fhdw.knn.network.neuron;

import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.Connection;

public class DenseNeuron extends Neuron {

    public double bias = 0.000_000_000_1;
    public Connection[] incoming;
    public ActivationFunction activationFunction;

    public double compute(double[] input) {
        double weightedSum = 0;
        for (int i = 0; i < input.length; i++) {
            if (incoming[i].guard) {
                weightedSum += incoming[i].weight * input[i];
            }
        }
        return weightedSum + bias;
    }

}
