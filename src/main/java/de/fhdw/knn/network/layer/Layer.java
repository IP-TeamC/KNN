package de.fhdw.knn.network.layer;

import de.fhdw.knn.network.neuron.Neuron;

/**
 * Abstrakte, minimale Definition eines Layers
 */
public abstract class Layer {

    /**
     * Kann nicht direkt instanziiert werden (entweder {@link InputLayer} oder {@link DenseLayer})
     */
    protected Layer() {
    }

    /**
     * Liefert die Neuronen des Layers zurück
     *
     * @return alle Neuronen des Layers
     */
    public abstract Neuron[] neurons();

}
