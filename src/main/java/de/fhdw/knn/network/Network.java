package de.fhdw.knn.network;

import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.network.neuron.Connection;
import de.fhdw.knn.network.neuron.DenseNeuron;
import lombok.ToString;

@ToString
public class Network {

    public InputLayer inputLayer;
    public DenseLayer[] denseLayers;

    public Network(InputLayer inputLayer, DenseLayer... denseLayers) {
        this.inputLayer = inputLayer;
        this.denseLayers = denseLayers;
        this.connectAll();
    }

    public double[] feedForward(double[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            inputLayer.neurons[i].valueCache = inputs[i];
        }
        for (DenseLayer denseLayer : denseLayers) {
            for (DenseNeuron denseNeuron : denseLayer.neurons) {
                denseNeuron.compute();
            }
        }

        DenseLayer outputLayer = denseLayers[denseLayers.length - 1];
        double[] output = new double[outputLayer.neurons.length];
        for (int i = 0; i < outputLayer.neurons.length; i++) {
            output[i] = outputLayer.neurons[i].outputCache;
        }
        return output;
    }

    public void connectAll() {
        for (DenseNeuron denseNeuron : denseLayers[0].neurons) {
            denseNeuron.incoming = new Connection[inputLayer.neurons.length];
            for (int inputNeuron = 0; inputNeuron < inputLayer.neurons.length; inputNeuron++) {
                Connection conn = new Connection();
                conn.inputNeuron = inputLayer.neurons[inputNeuron];
                denseNeuron.incoming[inputNeuron] = conn;
            }
        }

        for (int layer = 1; layer < denseLayers.length; layer++) {
            DenseNeuron[] currentNeurons = denseLayers[layer].neurons;
            DenseNeuron[] prevNeurons = denseLayers[layer - 1].neurons;
            for (DenseNeuron currentNeuron : currentNeurons) {
                currentNeuron.incoming = new Connection[prevNeurons.length];
                for (int prevNeuron = 0; prevNeuron < prevNeurons.length; prevNeuron++) {
                    Connection conn = new Connection();
                    conn.inputNeuron = prevNeurons[prevNeuron];
                    currentNeuron.incoming[prevNeuron] = conn;
                }
            }
        }
    }

}
