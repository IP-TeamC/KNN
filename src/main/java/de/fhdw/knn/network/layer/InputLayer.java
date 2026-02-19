package de.fhdw.knn.network.layer;

import de.fhdw.knn.network.neuron.InputNeuron;
import de.fhdw.knn.network.neuron.Neuron;

/**
 * 1. Layer des Netzwerks, welcher nur für die Eingabe genutzt wird.
 */
public class InputLayer extends Layer {

    /**
     * Neuronen des Input Layers
     */
    public InputNeuron[] neurons;

    @Override
    public Neuron[] neurons() {
        return neurons;
    }

    /**
     * Erstellt einen InputLayer mit der angegebenen Anzahl Neuronen
     */
    public InputLayer(int neurons) {
        this.neurons = new InputNeuron[neurons];
        for (int i = 0; i < neurons; i++) {
            this.neurons[i] = new InputNeuron();
        }
    }

}
