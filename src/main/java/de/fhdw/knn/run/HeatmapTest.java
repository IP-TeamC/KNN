package de.fhdw.knn.run;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.visualization.HeatmapData;
import de.fhdw.knn.visualization.HeatmapWindow;

public class HeatmapTest {
    public static void main(String[] args) {
        // 1. Netzwerk Setup
        InputLayer inputLayer = new InputLayer(1);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 200, 200, 1);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        // 2. Matrix generieren
        HeatmapData heatmapData = new HeatmapData(network);

        // 3. Bestehende Fenster-Klasse nutzen
        HeatmapWindow window = new HeatmapWindow();
        window.showSingleMatrix("Manuelle Gewichtsmatrix", heatmapData);
    }
}