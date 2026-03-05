package de.fhdw.knn.network.io;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.SuperNeuron;
import lombok.SneakyThrows;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Consumer;

/**
 * Importer zum Erzeugen eines {@link Network}-Objekts aus einem exportierten Netzes (KNN-Datei)
 */
public class Importer {

    private Consumer<ByteBuffer> state = this::createInputLayer;

    private InputLayer inputLayer;
    private DenseLayer[] denseLayers;
    private Network network;

    private int denseLayer = 0;
    private int neuron = 0;
    private int conn = 0;

    /**
     * Importiert das Netzwerk aus dem übergebenen KNN-Datei-Pfad
     */
    public static Network importNetwork(String fileName) {
        return new Importer().load(fileName);
    }

    /**
     * Importiert das Netzwerk aus dem übergebenen Byte-Array<br>
     * <strong>WICHTIG! Diese Methode darf für das gleiche Objekt nur einmal aufgerufen werden.
     * Jeder Import muss mit einem weiteren {@link Importer}-Objekt passieren.</strong>
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
    public Network load(byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        while (buffer.position() != buffer.capacity()) {
            state.accept(buffer);
        }
        return network;
    }

    /**
     * Importiert das Netzwerk aus dem übergebenen KNN-Datei-Pfad
     * <strong>WICHTIG! Diese Methode darf für das gleiche Objekt nur einmal aufgerufen werden.
     * Jeder Import muss mit einem weiteren {@link Importer}-Objekt passieren.</strong>
     */
    @SneakyThrows
    public Network load(String fileName) {
        return load(Files.readAllBytes(Paths.get(fileName)));
    }

    private void createInputLayer(ByteBuffer buffer) {
        inputLayer = new InputLayer(buffer.getInt());
        state = this::createDenseLayers;
    }

    private void createDenseLayers(ByteBuffer buffer) {
        denseLayers = new DenseLayer[buffer.getInt()];
        state = this::createDenseLayer;
    }

    private void createDenseLayer(ByteBuffer buffer) {
        denseLayers[denseLayer++] = new DenseLayer(buffer.getInt());
        if (denseLayer >= denseLayers.length) {
            network = new Network(0, WeightInitializer.ZERO, inputLayer, denseLayers);
            denseLayer = 0;
            state = this::updateActivationFunctionAndBias;
        }
    }

    private void updateActivationFunctionAndBias(ByteBuffer buffer) {
        int activationFunction = buffer.getInt();
        if (activationFunction == Integer.MAX_VALUE) {
            state = this::updateSuperNeuron;
            return;
        }

        DenseNeuron dn = (DenseNeuron) denseLayers[denseLayer].neurons[neuron];
        dn.activationFunction = ActivationFunction.FUNCTIONS.get(activationFunction);
        dn.bias = buffer.getDouble();
        state = this::updateWeights;
    }

    private void updateSuperNeuron(ByteBuffer buffer) {
        int subnetSize = buffer.getInt();
        byte[] subnetRaw = new byte[subnetSize];
        buffer.get(subnetRaw);
        Network subnet = new Importer().load(subnetRaw);

        new SuperNeuron(subnet).insert(network, denseLayer, neuron);
        state = this::updateWeights;
    }

    private void updateWeights(ByteBuffer buffer) {
        denseLayers[denseLayer].neurons[neuron].incoming[conn].guard = buffer.get() >= 1;
        denseLayers[denseLayer].neurons[neuron].incoming[conn++].weight = buffer.getDouble();
        if (conn >= denseLayers[denseLayer].neurons[neuron].incoming.length) {
            neuron += 1;
            conn = 0;
            state = this::updateActivationFunctionAndBias;

            if (neuron >= denseLayers[denseLayer].neurons.length) {
                denseLayer += 1;
                neuron = 0;
            }
        }
    }

}
