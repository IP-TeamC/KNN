package de.fhdw.knn.network.layer;

import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.Neuron;

public class DenseLayer extends Layer {

    public DenseNeuron[] neurons;

    @Override
    public Neuron[] neurons() {
        return neurons;
    }

    public DenseLayer(int neurons) {
        this.neurons = new DenseNeuron[neurons];
        for (int i = 0; i < neurons; i++) {
            this.neurons[i] = new DenseNeuron();
        }
    }

}
