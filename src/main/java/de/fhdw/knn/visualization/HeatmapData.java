package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.layer.DenseLayer;

import java.util.Arrays;

/**
 * Stellt Daten dar, die für die Visualisierung einer Heatmap der Gewichte eines neuronalen Netzwerks erforderlich sind.
 */
public class HeatmapData {

    /**
     * Die Netzwerkinstanz, die die Struktur eines neuronalen Netzwerks darstellt.
     */
    private final Network network;
    /**
     * Enthält die Gesamtzahl der Neuronen über alle Schichten des zugehörigen neuronalen Netzwerks.
     */
    private final int totalNeurons;
    /**
     * Stellt die Startindizes der Neuronen jeder Schicht in einem Array aller Neuronen
     * über alle Schichten im Netzwerk dar.
     *
     * <p>Jede Schicht im Netzwerk besteht aus einem zusammenhängenden Block von Neuronen.
     * Das Array {@code layerOffsets} speichert den Startindex jedes Blocks einer Ebene.
     *
     * @see HeatmapData#computeLayerOffsets()
     */
    private final int[] layerOffsets;

    /**
     * Erstellt eine neue {@code HeatmapData}-Instanz für das angegebene Netzwerk.
     *
     * @param network Das neuronale Netzwerk, für das die Heatmap-Daten generiert werden sollen.
     */
    public HeatmapData(Network network) {
        this.network = network;
        this.totalNeurons = calculateTotalNeurons();
        this.layerOffsets = computeLayerOffsets();
    }

    /**
     * Erstellt eine vollständige Gewichtsmatrix, die die Gewichte zwischen allen Neuronen im Netzwerk darstellt.
     * Jedes Element in der Matrix bezeichnet das Gewicht von einem Quellneuron (Zeilenindex) zu einem Zielneuron (Spaltenindex).
     * Nicht verbundene Neuronenpaare werden mit {@code Double.NaN} dargestellt.
     *
     * @return Ein 2D-Array von Double-Werten, das die vollständige Gewichtsmatrix darstellt, wobei die Zeilen die Quellneuronen und
     *         die Spalten die Zielneuronen repräsentieren. Nicht verbundene Neuronen haben den Wert {@code Double.NaN}.
     *
     * @see Double#NaN
     */
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
                    matrix[sourceGlobalIndex][targetGlobalIndex] = neuron.incoming[w].weight;
                }
            }
        }
        return matrix;
    }

    /**
     * Erzeugt ein Array von Neuronenbezeichnungen, die alle Neuronen im Netzwerk repräsentieren.
     *
     * @return Ein Array von Strings, wobei jedes Element der Bezeichnung eines Neurons entspricht.
     */
    public String[] getNeuronLabels() {
        String[] labels = new String[this.totalNeurons];
        int globalIndex = 0;

        // Input Layer
        int inputCount = this.network.inputLayer.neurons.length;
        for (int i = 0; i < inputCount; i++) {
            labels[globalIndex++] = "I-N" + (i + 1);
        }

        // Hidden / Output Layer
        for (int l = 0; l < this.network.denseLayers.length; l++) {
            DenseLayer layer = this.network.denseLayers[l];
            for (int n = 0; n < layer.neurons.length; n++) {
                String prefix = (l == this.network.denseLayers.length - 1) ? "O" : "H" + (l + 1);
                labels[globalIndex++] = prefix + "-N" + (n + 1);
            }
        }
        return labels;
    }

    /**
     * Berechnet die Gesamtanzahl der Neuronen im Netzwerk, indem die Neuronen aus dem Input-Layer
     * sowie allen Dense-Layern summiert werden.
     *
     * @return Die Gesamtanzahl der Neuronen im Netzwerk.
     */
    private int calculateTotalNeurons() {
        int sum = this.network.inputLayer.neurons.length;
        for (DenseLayer layer : this.network.denseLayers) {
            sum += layer.neurons.length;
        }
        return sum;
    }

    /**
     * Berechnet die Startindizes der Neuronen jeder Schicht.
     *
     * @return Ein Array von {@code Integer}, wobei jedes Element den Startindex der Neuronen einer Schicht darstellt.
     *         Das erste Element ist immer 0 (Beginn der Eingabeschicht), und die nachfolgenden Elemente entsprechen
     *         der kumulativen Anzahl von Neuronen bis zu jeder Schicht.
     *
     * @see HeatmapData#layerOffsets
     */
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
