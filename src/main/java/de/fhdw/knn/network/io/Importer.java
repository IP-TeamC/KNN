package de.fhdw.knn.network.io;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import lombok.SneakyThrows;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.function.Consumer;

public class Importer {

    private Consumer<String> state = this::createInputLayer;
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
    // BIAS
    // Weights... je Connection (0 -> n)
    // }
    @SneakyThrows
    public Network load(String fileName) {
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            br.lines().forEach(line -> state.accept(line));
        }
        return network;
    }

    private void createInputLayer(String line) {
        inputLayer = new InputLayer(Integer.parseInt(line));
        state = this::createDenseLayers;
    }

    private void createDenseLayers(String line) {
        denseLayers = new DenseLayer[Integer.parseInt(line)];
        state = this::createDenseLayer;
    }

    private void createDenseLayer(String line) {
        denseLayers[denseLayer++] = new DenseLayer(Integer.parseInt(line));
        if (denseLayer >= denseLayers.length) {
            network = new Network(0, inputLayer, denseLayers);
            denseLayer = 0;
            state = this::updateBias;
        }
    }

    private void updateBias(String line) {
        if (denseLayer >= denseLayers.length) {
            if (!line.isBlank()) {
                throw new IllegalStateException("finished");
            }
        }

        denseLayers[denseLayer].neurons[neuron].bias = Double.longBitsToDouble(Long.parseLong(line));
        state = this::updateWeights;
    }

    private void updateWeights(String line) {
        denseLayers[denseLayer].neurons[neuron].incoming[conn++].weight = Double.longBitsToDouble(Long.parseLong(line));
        if (conn >= denseLayers[denseLayer].neurons[neuron].incoming.length) {
            neuron += 1;
            conn = 0;
            state = this::updateBias;

            if (neuron >= denseLayers[denseLayer].neurons.length) {
                denseLayer += 1;
                neuron = 0;
            }
        }
    }

}
