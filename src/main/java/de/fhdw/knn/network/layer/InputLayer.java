package de.fhdw.knn.network.layer;

import de.fhdw.knn.network.neuron.InputNeuron;
import de.fhdw.knn.network.neuron.Neuron;
import lombok.ToString;

@ToString
public class InputLayer extends Layer {

    public InputNeuron[] neurons;

    @Override
    public Neuron[] neurons() {
        return neurons;
    }

    public InputLayer(int neurons) {
        this.neurons = new InputNeuron[neurons];
        for (int i = 0; i < neurons; i++) {
            this.neurons[i] = new InputNeuron();
        }
    }

}
