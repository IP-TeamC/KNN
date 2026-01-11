package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.io.Importer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.Neuron;
import eu.hansolo.fx.charts.SankeyPlot;
import eu.hansolo.fx.charts.data.PlotItem;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.List;

public class NetworkToSankeyConverter extends Application {

    @Override
    public void start(Stage primaryStage) {
        // Network network = Importer.importNetwork("models/datapoints_small.knn");
        Network network = Importer.importNetwork("models/datapoints_small5.knn");
        // Network network = Importer.importNetwork("models/datapoints.knn");
        // Network network = Importer.importNetwork("models/bq.knn");
        // Network network = Importer.importNetwork("models/sin_snake.knn");

        List<PlotItem> allItems = new ArrayList<>();

        Color[] layerColors = {
                Color.RED,
                Color.ORANGERED,
                Color.ORANGE,
                Color.GOLD,
                Color.LIGHTGREEN,
                Color.CYAN,
                Color.LIGHTBLUE,
                Color.BLUE,
        };

        int totalLayers = network.denseLayers.length + 1;

        // Output Layer
        DenseLayer outputLayer = network.denseLayers[network.denseLayers.length - 1];
        PlotItem[] outputItems = new PlotItem[outputLayer.neurons.length];

        for (int i = 0; i < outputLayer.neurons.length; i++) {
            outputItems[i] = new PlotItem(
                    "O-" + (i + 1),
                    1,
                    getColorForLayer(0, layerColors, totalLayers),
                    0
            );
            allItems.add(outputItems[i]);
        }

        // Hidden Layers
        PlotItem[][] hiddenItems = new PlotItem[network.denseLayers.length - 1][];

        for (int layer = network.denseLayers.length - 2; layer >= 0; layer--) {
            DenseLayer currentLayer = network.denseLayers[layer];
            int visualLayer = network.denseLayers.length - 1 - layer;

            hiddenItems[layer] = new PlotItem[currentLayer.neurons.length];

            for (int i = 0; i < currentLayer.neurons.length; i++) {
                hiddenItems[layer][i] = new PlotItem(
                        "H" + (layer + 1) + "-" + (i + 1),
                        1,
                        getColorForLayer(visualLayer, layerColors, totalLayers),
                        visualLayer
                );
                allItems.add(hiddenItems[layer][i]);
            }
        }

        // Input Layer
        Neuron[] inputNeurons = network.inputLayer.neurons;
        PlotItem[] inputItems = new PlotItem[inputNeurons.length];

        for (int i = 0; i < inputNeurons.length; i++) {
            inputItems[i] = new PlotItem(
                    "I-" + (i + 1),
                    1,
                    getColorForLayer(totalLayers - 1, layerColors, totalLayers),
                    totalLayers - 1
            );
            allItems.add(inputItems[i]);
        }

        // Verbindungen erstellen

        // Output -> Hidden (letzter Hidden Layer)
        if (network.denseLayers.length > 1) {
            createConnectionsReversed(outputItems, hiddenItems[network.denseLayers.length - 2], outputLayer);
        } else {
            createConnectionsReversed(outputItems, inputItems, outputLayer);
        }

        // Hidden Layers untereinander
        for (int layer = network.denseLayers.length - 2; layer > 0; layer--) {
            createConnectionsReversed(hiddenItems[layer], hiddenItems[layer - 1], network.denseLayers[layer]);
        }

        // Letzter Hidden Layer -> Input
        if (network.denseLayers.length > 1) {
            createConnectionsReversed(hiddenItems[0], inputItems, network.denseLayers[0]);
        }

        // SankeyPlot
        SankeyPlot sankey = new SankeyPlot();
        sankey.setItems(allItems);
        sankey.setStreamFillMode(SankeyPlot.StreamFillMode.GRADIENT);
        sankey.setShowFlowDirection(false);

        StackPane root = new StackPane(sankey);
        Scene scene = new Scene(root, 1200, 800);

        primaryStage.setTitle("Künstliches Neuronales Netz - " +
                network.inputLayer.neurons.length + " Inputs, " +
                outputLayer.neurons.length + " Outputs");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void createConnectionsReversed(PlotItem[] rightItems, PlotItem[] leftItems, DenseLayer rightLayer) {
        for (int rIdx = 0; rIdx < rightItems.length; rIdx++) {
            DenseNeuron rightNeuron = rightLayer.neurons[rIdx];
            Connection[] incomingFromLeft = rightNeuron.incoming;

            for (int lIdx = 0; lIdx < leftItems.length; lIdx++) {
                double weight = Math.abs(incomingFromLeft[lIdx].weight);

                if (weight > 0.1) {
                    rightItems[rIdx].addToOutgoing(leftItems[lIdx], weight);
                }
            }
        }
    }

    private Color getColorForLayer(int layer, Color[] colors, int totalLayers) {
        int colorIndex = (layer * (colors.length - 1)) / Math.max(1, totalLayers - 1);
        return colors[Math.min(colorIndex, colors.length - 1)];
    }

    public static void main(String[] args) {
        launch(args);
    }
}