package de.fhdw.knn.network.io;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import lombok.SneakyThrows;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Consumer;

public class Importer {

    private Consumer<ByteBuffer> state = this::createInputLayer;

    private InputLayer inputLayer;
    private DenseLayer[] denseLayers;
    private Network network;

    private int denseLayer = 0;
    private int neuron = 0;
    private int conn = 0;

    public static Network importNetwork(String fileName) {
        return new Importer().load(fileName);
    }

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
    public Network load(String fileName) {
        ByteBuffer buffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(fileName)));
        while (buffer.position() != buffer.capacity()) {
            state.accept(buffer);
        }
        return network;
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
        denseLayers[denseLayer].neurons[neuron].activationFunction = ActivationFunction.FUNCTIONS.get(buffer.getInt());
        denseLayers[denseLayer].neurons[neuron].bias = buffer.getDouble();
        state = this::updateWeights;
    }

    private void updateWeights(ByteBuffer buffer) {
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
