package de.fhdw.knn.network.layer;

import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.Neuron;

public class DenseLayer extends Layer {

    public AbstractDenseNeuron[] neurons;

    @Override
    public Neuron[] neurons() {
        return neurons;
    }

    public DenseLayer(int neurons) {
        this.neurons = new AbstractDenseNeuron[neurons];
        for (int i = 0; i < neurons; i++) {
            this.neurons[i] = new DenseNeuron();
        }
    }

    public DenseLayer withActivationFunction(ActivationFunction activationFunction) {
        for (AbstractDenseNeuron neuron : neurons) {
            ((DenseNeuron) neuron).activationFunction = activationFunction;
        }
        return this;
    }

    public static DenseLayer[] createLayers(ActivationFunction hiddenActivationFunction, ActivationFunction outputActivationFunction, int... neurons) {
        DenseLayer[] layers = new DenseLayer[neurons.length];
        for (int i = 0; i < neurons.length - 1; i++) {
            layers[i] = new DenseLayer(neurons[i]).withActivationFunction(hiddenActivationFunction);
        }
        layers[neurons.length - 1] = new DenseLayer(neurons[neurons.length - 1]).withActivationFunction(outputActivationFunction);
        return layers;
    }

}
