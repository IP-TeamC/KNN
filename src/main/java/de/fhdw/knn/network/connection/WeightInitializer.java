package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;

import java.util.Random;

/**
 * Interface zur Implementierung der Gewichtsinitialisierung bei Erstellung eines neuen Netzes
 */
public interface WeightInitializer {

    /**
     * Normalverteilung mit Standardabweichung sqrt(2 / Summe der Neuronen im vorherigen und folgenden Layer)
     */
    WeightInitializer GLOROT = new GlorotWeightInitializer();

    /**
     * Gleichmäßige Verteilung zwischen +-sqrt(6 / Summe der Neuronen im vorherigen und folgenden Layer)
     */
    WeightInitializer GLOROT_UNIFORM = new GlorotUniformWeightInitializer();

    /**
     * Normalverteilung mit Standardabweichung sqrt(2 / Summe der Neuronen im vorherigen Layer)
     */
    WeightInitializer HE = new HeWeightInitializer();

    /**
     * Gleichmäßige Verteilung zwischen +-sqrt(6 / Summe der Neuronen im vorherigen Layer)
     */
    WeightInitializer HE_UNIFORM = new HeUniformWeightInitializer();

    /**
     * Initialisiert alle Gewichte als 0
     */
    WeightInitializer ZERO = new ZeroWeightInitializer();

    /**
     * Bestimmt das Gewicht einer Verbindung
     *
     * @param random  Zufallsgenerator
     * @param network zur Bestimmung der Eigenschaften des Netzwerks
     * @param layer   Layer, in dem sich das Neuron befindet, zu dem die Verbindung mit dem zu bestimmenden Gewicht erstellt wird
     */
    double nextWeight(Random random, Network network, int layer);

}