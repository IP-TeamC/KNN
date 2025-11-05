package de.fhdw.knn.network.io;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.Connection;
import de.fhdw.knn.network.neuron.DenseNeuron;
import lombok.SneakyThrows;

import java.io.FileWriter;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

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
        int size = 8 + network.denseLayers.length * 4;
        for (int i = 0; i < network.denseLayers.length; i++) {
            size += 8 * network.denseLayers[i].neurons.length * (1 + network.denseLayers[i].neurons[0].incoming.length);
        }
        ByteBuffer buffer = ByteBuffer.allocate(size);
        buffer.putInt(network.inputLayer.neurons.length);
        buffer.putInt(network.denseLayers.length);
        for (DenseLayer layer : network.denseLayers) {
            buffer.putInt(layer.neurons.length);
        }

        for (DenseLayer layer : network.denseLayers) {
            for (DenseNeuron neuron : layer.neurons) {
                buffer.putDouble(neuron.bias);
                for (Connection conn : neuron.incoming) {
                    buffer.putDouble(conn.weight);
                }
            }
        }
        Files.write(Paths.get(fileName), buffer.array(), StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    }

    @SneakyThrows
    public static void exportText(Network network, String fileName) {
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
