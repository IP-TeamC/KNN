package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.layer.DenseLayer;

import java.util.Arrays;

public class HeatmapData {
    private final Network network;
    private final int totalNeurons;
    private final int[] layerOffsets;

    public HeatmapData(Network network) {
        this.network = network;
        this.totalNeurons = calculateTotalNeurons();
        this.layerOffsets = computeLayerOffsets();
    }

    public double[][] buildFullWeightMatrix() {
        double[][] matrix = new double[this.totalNeurons][this.totalNeurons];

        for (double[] row : matrix) {
            Arrays.fill(row, Double.NaN);
        }

        DenseLayer[] layers = this.network.denseLayers;

        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            DenseLayer currentLayer = layers[layerIndex];

            int sourceLayerStart = this.layerOffsets[layerIndex]; // Verbindung von
            int targetLayerStart = this.layerOffsets[layerIndex + 1]; // Verbindung zu

            for (int n = 0; n < currentLayer.neurons.length; n++) {
                var neuron = currentLayer.neurons[n];
                int targetGlobalIndex = targetLayerStart + n;

                for (int w = 0; w < neuron.incoming.length; w++) {
                    int sourceGlobalIndex = sourceLayerStart + w;

                    if (sourceGlobalIndex < this.totalNeurons) {
                        matrix[sourceGlobalIndex][targetGlobalIndex] = neuron.incoming[w].weight;
                    }
                }
            }
        }
        return matrix;
    }

    public String[] getNeuronLabels() {
        String[] labels = new String[this.totalNeurons];
        int globalIndex = 0;

        // Input Layer
        int inputCount = this.network.inputLayer.neurons.length;
        for (int i = 0; i < inputCount; i++) {
            labels[globalIndex++] = "Input " + i;
        }

        // Hidden / Output Layer
        for (int l = 0; l < this.network.denseLayers.length; l++) {
            DenseLayer layer = this.network.denseLayers[l];
            for (int n = 0; n < layer.neurons.length; n++) {
                String prefix = (l == this.network.denseLayers.length - 1) ? "Output " : "L" + (l + 1) + " N";
                labels[globalIndex++] = prefix + n;
            }
        }
        return labels;
    }

    private int calculateTotalNeurons() {
        int sum = this.network.inputLayer.neurons.length;
        for (DenseLayer layer : this.network.denseLayers) {
            sum += layer.neurons.length;
        }
        return sum;
    }

    // Neuronen haben globale Indizes, hier kurz berechnen, bei welchem Index ein neuer Layer beginnt
    private int[] computeLayerOffsets() {
        DenseLayer[] layers = this.network.denseLayers;

        int[] offsets = new int[layers.length + 1];
        offsets[0] = 0;

        int currentCount = 0;
        currentCount += this.network.inputLayer.neurons.length;

        for (int i = 0; i < layers.length; i++) {
            offsets[i + 1] = currentCount;
            currentCount += layers[i].neurons.length;
        }
        return offsets;
    }
}