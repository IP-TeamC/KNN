package de.fhdw.knn.network.neuron;

public class InputNeuron extends Neuron {

    public double valueCache;

    @Override
    public void compute() {}

    @Override
    public double output() {
        return valueCache;
    }

}
