package de.fhdw.knn.network.layer;

import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.SuperNeuron;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.Neuron;

/**
 * DenseLayer sind alle Hidden Layer sowie der Output Layer.
 * Der DenseLayer besteht aus einem Array von {@link AbstractDenseNeuron}s.
 * Diese sind entweder ein {@link DenseNeuron} (konventionelles Neuron mit Aktivierungsfunktion) oder
 * ein {@link SuperNeuron} (Neuron ist selbst ein ganzes Netzwerk).
 * Jeder DenseLayer muss einen vorherigen Layer haben.
 */
public class DenseLayer extends Layer {

    public AbstractDenseNeuron[] neurons;

    @Override
    public Neuron[] neurons() {
        return neurons;
    }

    /**
     * Erzeugt einen DenseLayer bestehend aus {@link DenseNeuron} ohne Verbindungen und ohne Aktivierungsfunktionen
     *
     * @param neurons legt die Anzahl der Neuronen im Layer fest
     * @see DenseLayer
     */
    public DenseLayer(int neurons) {
        this.neurons = new AbstractDenseNeuron[neurons];
        for (int i = 0; i < neurons; i++) {
            this.neurons[i] = new DenseNeuron();
        }
    }

    /**
     * Setzt die Aktivierungsfunktion jedes Neurons im Layer auf die übergebene Aktivierungsfunktion.<br>
     * <strong>Der Layer darf keine anderen Neuronen-Arten außer {@link DenseNeuron} enthalten (z.B. {@link SuperNeuron} ist nicht erlaubt).</strong>
     *
     * @return Die Rückgabe kann ignoriert werden, da diese nur sich selbst zurückgibt.
     */
    public DenseLayer withActivationFunction(ActivationFunction activationFunction) {
        for (AbstractDenseNeuron neuron : neurons) {
            ((DenseNeuron) neuron).activationFunction = activationFunction;
        }
        return this;
    }

    /**
     * Erstellt mehrere Hidden Layer mit der gleichen Aktivierungsfunktion sowie einen Output Layer.
     * @param hiddenActivationFunction Aktivierungsfunktion aller Hidden Layer
     * @param outputActivationFunction Aktivierungsfunktion des Output Layers
     * @param neurons Anzahl der Neuronen in den Layern (1. Wert = 1. Layer, letzter Wert = Output Layer)
     * @return Array, das die Hidden Layer sowie den Output Layer enthält
     */
    public static DenseLayer[] createLayers(ActivationFunction hiddenActivationFunction, ActivationFunction outputActivationFunction, int... neurons) {
        DenseLayer[] layers = new DenseLayer[neurons.length];
        for (int i = 0; i < neurons.length - 1; i++) {
            layers[i] = new DenseLayer(neurons[i]).withActivationFunction(hiddenActivationFunction);
        }
        layers[neurons.length - 1] = new DenseLayer(neurons[neurons.length - 1]).withActivationFunction(outputActivationFunction);
        return layers;
    }

}
