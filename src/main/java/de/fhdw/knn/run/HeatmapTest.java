package de.fhdw.knn.run;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.util.Heatmap;
import de.fhdw.knn.util.HeatmapData;

public class HeatmapTest {

    public static void main(String[] args) {
        InputLayer inputLayer = new InputLayer(1);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 200, 200, 1);
        Network network = new Network(42, WeightInitializer.HE, inputLayer , denseLayers);
        HeatmapData heatmap = new HeatmapData(network);
        double[][] matrix = heatmap.buildFullWeightMatrix();
        Heatmap map = new Heatmap("test", matrix);
        map.drawHeatmap();
    }


}
