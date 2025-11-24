package de.fhdw.knn.util;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.layer.DenseLayer;

public class HeatmapData {
    private final Network network;
    private final int totalNeurons;

    public HeatmapData(Network network){
        this.network = network;
        this.totalNeurons = this.countTotalNeurons();
    }

    public double[][] buildFullWeightMatrix() {

        DenseLayer[] layers = this.network.denseLayers;

        int[] offsets = computeLayerOffsets(layers);

        double[][] matrix = new double[this.totalNeurons][this.totalNeurons];

        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {

            DenseLayer layer = layers[layerIndex];

            int toOffset = offsets[layerIndex + 1];
            int fromOffset = offsets[layerIndex];

            for (int n = 0; n < layer.neurons.length; n++) {

                int toNeuronID = toOffset + n;

                var neuron = layer.neurons[n];

                for (int w = 0; w < neuron.incoming.length; w++) {

                    int globalFromID = fromOffset + w;

                    matrix[globalFromID][toNeuronID] = neuron.incoming[w].weight;
                }
            }
        }
        this.printMatrix(matrix);
        return matrix;
    }

    private int countTotalNeurons() {
        int sum = 0;
        for (DenseLayer layer : this.network.denseLayers) {
            sum += layer.neurons.length;
        }
        sum += this.network.inputLayer.neurons.length;
        return sum;
    }

    private int[] computeLayerOffsets(DenseLayer[] layers) { //neuronen haben globale indizes, hier kurz berechnet bei welchem dieser globalen indizes ein neuer layer beginnt
        int[] offsets = new int[layers.length + 1];
        int count = this.network.inputLayer.neurons.length;

        for (int i = 0; i < layers.length; i++) {
            offsets[i + 1] = count;
            count += layers[i].neurons.length;
        }
        return offsets;
    }

    public void printMatrix(double[][] matrix){
        for (double[] row : matrix) {
            for (double val : row) {
                System.out.printf("%.2f\t", val);
            }
            System.out.println();
        }
    }
}
