package de.fhdw.knn.network.neuron;

/**
 * Abstraktes Neuron als Super-Klasse aller anderen Neuronen
 */
public abstract class Neuron {

    /**
     * Kann nicht direkt instanziiert werden (nur über {@link InputNeuron}, {@link DenseNeuron} oder {@link SuperNeuron})
     */
    protected Neuron() {
    }

}
