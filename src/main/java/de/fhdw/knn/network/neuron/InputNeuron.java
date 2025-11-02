package de.fhdw.knn.network.neuron;

import lombok.ToString;

@ToString
public class InputNeuron extends Neuron {

    @ToString.Exclude
    public double valueCache;

    @Override
    public void compute() {}

    @Override
    public double output() {
        return valueCache;
    }

}
