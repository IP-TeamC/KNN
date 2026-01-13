package de.fhdw.knn.network.io;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.DenseNeuron;
import lombok.SneakyThrows;

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
    // ActivationFunction
    // BIAS
    // Weights... je Connection (0 -> n)
    // }
    @SneakyThrows
    public static void export(Network network, String fileName) {
        int size = 8 + network.denseLayers.length * 4;
        for (int i = 0; i < network.denseLayers.length; i++) {
            // ActivationFunction (4 ...)
            size += 4 * network.denseLayers[i].neurons.length;
            // Bias (...1), Weights (...incoming.length)
            size += 8 * network.denseLayers[i].neurons.length * (1 + network.denseLayers[i].neurons[0].incoming.length);
        }
        ByteBuffer buffer = ByteBuffer.allocate(size);
        buffer.putInt(network.inputLayer.neurons.length);
        buffer.putInt(network.denseLayers.length);
        for (DenseLayer layer : network.denseLayers) {
            buffer.putInt(layer.neurons.length);
        }

        for (DenseLayer layer : network.denseLayers) {
            for (AbstractDenseNeuron neuron : layer.neurons) {
                DenseNeuron dn = (DenseNeuron) neuron;
                buffer.putInt(ActivationFunction.FUNCTIONS.indexOf(dn.activationFunction));
                buffer.putDouble(dn.bias);
                for (Connection conn : neuron.incoming) {
                    buffer.putDouble(conn.weight);
                }
            }
        }
        Files.write(Paths.get(fileName), buffer.array(), StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    }

}
