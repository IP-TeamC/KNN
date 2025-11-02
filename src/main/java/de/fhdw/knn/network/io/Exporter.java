package de.fhdw.knn.network.io;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.Connection;
import de.fhdw.knn.network.neuron.DenseNeuron;
import lombok.SneakyThrows;

import java.io.FileWriter;

public class Exporter {

    // Anzahl Input-Neurons
    // Anzahl Dense Layer
    // { je Layer
    // Anzahl Neuronen
    // }
    // { je Neuron je Layer (0 -> n)
    // BIAS
    // Weights... je Connection (0 -> n)
    // }
    @SneakyThrows
    public static void export(Network network, String fileName) {
        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write(network.inputLayer.neurons.length + "\n");
            writer.write(network.denseLayers.length + "\n");
            for (DenseLayer layer : network.denseLayers) {
                writer.write(layer.neurons.length + "\n");
            }

            for (DenseLayer layer : network.denseLayers) {
                for (DenseNeuron neuron : layer.neurons) {
                    writer.write(Double.doubleToLongBits(neuron.bias) + "\n");
                    for (Connection conn : neuron.incoming) {
                        writer.write(Double.doubleToLongBits(conn.weight) + "\n");
                    }
                }
            }
        }
    }

}
