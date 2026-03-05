package de.fhdw.knn.network.io;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.SuperNeuron;
import lombok.SneakyThrows;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * Exporter zum Speichern des Netzwerks in einer kompakten Datei (binäres Format)
 */
public class Exporter {

    /**
     * Exportiert das Netzwerk zu einem Byte-Array
     */
    // Anzahl Input-Neurons
    // Anzahl Dense Layer
    // { je Layer
    // Anzahl Neuronen
    // }
    // { je Neuron je Layer (0 -> n)
    // ActivationFunction (Integer.MAX_VALUE = Super-Neuron)
    // { wenn DenseNeuron
    // BIAS
    // }
    // { sonst wenn SuperNeuron
    // Network-Size
    // Network
    // }
    // (Guard, Weight)... je Connection (0 -> n)
    // }
    public static byte[] export(Network network) {
        List<byte[]> exportedSubs = new LinkedList<>();
        for (DenseLayer layer : network.denseLayers) {
            for (AbstractDenseNeuron neuron : layer.neurons) {
                if (neuron instanceof SuperNeuron sn) {
                    exportedSubs.add(export(sn.network));
                }
            }
        }

        int size = 8 + network.denseLayers.length * 4
                - exportedSubs.size() * 4 // halb so groß wie Bias
                + exportedSubs.stream().reduce(0, (sizeSum, exported) -> sizeSum + exported.length, Integer::sum);
        for (int i = 0; i < network.denseLayers.length; i++) {
            // ActivationFunction (4 ...)
            size += 4 * network.denseLayers[i].neurons.length;
            // Bias (...1), Weights (...incoming.length)
            size += 8 * network.denseLayers[i].neurons.length * (1 + network.denseLayers[i].neurons[0].incoming.length);
            // Guard
            size += network.denseLayers[i].neurons.length * network.denseLayers[i].neurons[0].incoming.length;
        }
        ByteBuffer buffer = ByteBuffer.allocate(size);
        buffer.putInt(network.inputLayer.neurons.length);
        buffer.putInt(network.denseLayers.length);
        for (DenseLayer layer : network.denseLayers) {
            buffer.putInt(layer.neurons.length);
        }

        Iterator<byte[]> exportedSubsIterator = exportedSubs.iterator();
        for (DenseLayer layer : network.denseLayers) {
            for (AbstractDenseNeuron neuron : layer.neurons) {
                if (neuron instanceof SuperNeuron) {
                    byte[] exportedSub = exportedSubsIterator.next();
                    buffer.putInt(Integer.MAX_VALUE); // ActivationFunction = SuperNeuron
                    buffer.putInt(exportedSub.length);
                    buffer.put(exportedSub);
                } else if (neuron instanceof DenseNeuron dn) {
                    buffer.putInt(ActivationFunction.FUNCTIONS.indexOf(dn.activationFunction));
                    buffer.putDouble(dn.bias);
                } else {
                    throw new IllegalArgumentException("unknown dense neuron type");
                }

                for (Connection conn : neuron.incoming) {
                    buffer.put((byte) (conn.guard ? 1 : 0));
                    buffer.putDouble(conn.weight);
                }
            }
        }
        return buffer.array();
    }

    /**
     * Exportiert das Netzwerk und schreibt das Ergebnis als Datei in den übergebenen Pfad
     */
    @SneakyThrows
    public static void export(Network network, String fileName) {
        byte[] exported = export(network);
        Files.write(Paths.get(fileName), exported, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    }

}
