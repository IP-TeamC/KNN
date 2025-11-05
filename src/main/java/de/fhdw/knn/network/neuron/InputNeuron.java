package de.fhdw.knn.network.neuron;

public class InputNeuron extends Neuron {

    @Deprecated
    public double valueCache;

    @Override
    public void compute() {}

    @Deprecated
    @Override
    public double output() {
        return valueCache;
    }

}
