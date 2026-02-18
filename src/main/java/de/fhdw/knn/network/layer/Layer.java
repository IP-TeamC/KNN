package de.fhdw.knn.network.layer;

import de.fhdw.knn.network.neuron.Neuron;

/**
 * Abstrakte, minimale Definition eines Layers
 */
public abstract class Layer {

    /**
     * Liefert die Neuronen des Layers zurück
     */
    public abstract Neuron[] neurons();

}
