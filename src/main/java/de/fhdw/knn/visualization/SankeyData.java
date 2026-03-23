package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.Neuron;
import eu.hansolo.fx.charts.data.PlotItem;
import javafx.scene.paint.Color;

import java.util.ArrayList;
import java.util.List;

/**
 * Die Klasse {@code SankeyData} bietet Funktionalität zur Umwandlung einer neuronalen Netzwerkstruktur
 * in eine Liste von {@code PlotItem}-Instanzen, die zur Erstellung eines Sankey-Diagramms geeignet sind.
 * Sie verarbeitet die Schichten des neuronalen Netzwerks und berechnet die entsprechende visuelle
 * Darstellung, indem sie Neuronen und deren Verbindungen auf {@code PlotItems} abbildet.
 *
 * @param network Die Netzwerkinstanz, die die Struktur eines neuronalen Netzwerks darstellt.
 */
public record SankeyData(Network network) {

    /**
     * Erstellt eine neue {@code SankeyData}-Instanz für das angegebene Netzwerk.
     *
     * @param network Das neuronale Netzwerk, das visualisiert werden soll.
     */
    public SankeyData {
    }

    /**
     * Konvertiert ein bestimmtes Netzwerk in eine Liste von {@code PlotItem} zur Visualisierung.
     * Das Netzwerk wird im Sankey-Diagramm für eine bessere Visualisierung der Gewichtsverteilung umgekehrt dargestellt.
     *
     * @param threshold    Der Schwellenwert; Verbindungen mit {@code |weight| <= threshold} werden ignoriert.
     * @param weightFilter Gibt an, ob positive, negative oder alle Gewichte dargestellt werden.
     * @return Eine Liste von {@code PlotItems}, die alle Schichten (Eingabe, versteckt und Ausgabe) des angegebenen Netzwerks darstellen.
     * @see PlotItem
     * @see WeightFilter
     */
    public List<PlotItem> convertToPlotItems(Double threshold, WeightFilter weightFilter) {
        List<PlotItem> allItems = new ArrayList<>();

        Color[] layerColors = {
                Color.RED, Color.ORANGERED, Color.ORANGE, Color.GOLD,
                Color.LIGHTGREEN, Color.CYAN, Color.LIGHTBLUE, Color.BLUE,
        };

        DenseLayer[] denseLayers = network().denseLayers;
        int totalLayers = denseLayers.length + 1;

        // Output Layer
        DenseLayer outputLayer = denseLayers[denseLayers.length - 1];
        PlotItem[] outputItems = new PlotItem[outputLayer.neurons.length];

        for (int i = outputLayer.neurons.length - 1; i >= 0; i--) {
            outputItems[i] = new PlotItem(
                    "O-N" + (i + 1),
                    1,
                    getColorForLayer(0, layerColors, totalLayers),
                    0
            );
            allItems.add(outputItems[i]);
        }

        // Hidden Layers
        PlotItem[][] hiddenItems = new PlotItem[denseLayers.length - 1][];

        for (int layer = denseLayers.length - 2; layer >= 0; layer--) {
            DenseLayer currentLayer = denseLayers[layer];
            int visualLayer = denseLayers.length - 1 - layer;

            hiddenItems[layer] = new PlotItem[currentLayer.neurons.length];

            for (int i = currentLayer.neurons.length - 1; i >= 0; i--) {
                hiddenItems[layer][i] = new PlotItem(
                        "H" + (layer + 1) + "-N" + (i + 1),
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

        for (int i = inputNeurons.length - 1; i >= 0; i--) {
            inputItems[i] = new PlotItem(
                    "I-N" + (i + 1),
                    1,
                    getColorForLayer(totalLayers - 1, layerColors, totalLayers),
                    totalLayers - 1
            );
            allItems.add(inputItems[i]);
        }

        // Verbindungen erstellen

        // Output -> Hidden (letzter Hidden Layer)
        if (denseLayers.length > 1) {
            createConnectionsReversed(outputItems, hiddenItems[denseLayers.length - 2], outputLayer, threshold, weightFilter);
        } else {
            createConnectionsReversed(outputItems, inputItems, outputLayer, threshold, weightFilter);
        }

        // Hidden Layers untereinander
        for (int layer = denseLayers.length - 2; layer > 0; layer--) {
            createConnectionsReversed(hiddenItems[layer], hiddenItems[layer - 1], network.denseLayers[layer], threshold, weightFilter);
        }

        // Letzter Hidden Layer -> Input
        if (denseLayers.length > 1) {
            createConnectionsReversed(hiddenItems[0], inputItems, denseLayers[0], threshold, weightFilter);
        }

        return allItems;
    }

    /**
     * Erstellt umgekehrte Verbindungen zwischen zwei Schichten von {@code PlotItems} basierend auf den Gewichten
     * der Neuronen in der angegebenen {@code DenseLayer} und unter Beachtung des {@code THRESHOLD}.
     *
     * @param rightItems   Ein Array von {@code PlotItems}, das die rechte Schicht repräsentiert.
     * @param leftItems    Ein Array von {@code PlotItems}, das die linke Schicht repräsentiert.
     * @param rightLayer   Die {@code DenseLayer}, die die Neuronen der rechten Schicht enthält,
     *                     einschließlich ihrer eingehenden Verbindungsgewichte von den Neuronen in der
     *                     linken Schicht.
     * @param threshold    Der Schwellenwert; Verbindungen unterhalb dieses Betrags werden ignoriert.
     * @param weightFilter Gibt an, ob positive, negative oder alle Gewichte berücksichtigt werden.
     * @see PlotItem
     * @see DenseLayer
     */
    private void createConnectionsReversed(PlotItem[] rightItems, PlotItem[] leftItems, DenseLayer rightLayer, Double threshold, WeightFilter weightFilter) {
        for (int rIdx = 0; rIdx < rightItems.length; rIdx++) {
            AbstractDenseNeuron rightNeuron = rightLayer.neurons[rIdx];
            Connection[] incomingFromLeft = rightNeuron.incoming;

            for (int lIdx = 0; lIdx < leftItems.length; lIdx++) {
                double weight = incomingFromLeft[lIdx].weight;

                boolean passesFilter = switch (weightFilter) {
                    case POSITIVE -> weight > 0;
                    case NEGATIVE -> weight < 0;
                    case BOTH -> true;
                };

                if (passesFilter && Math.abs(weight) > threshold) {
                    rightItems[rIdx].addToOutgoing(leftItems[lIdx], Math.abs(weight));
                }
            }
        }
    }

    /**
     * Bestimmt die Farbe für eine bestimmte Ebene basierend auf dem angegebenen Farbarray und der Gesamtzahl der Ebenen.
     *
     * @param layer       Der Index der Ebene, für die die Farbe bestimmt werden soll.
     * @param colors      Ein Array von {@code Color}-Objekten, die die verfügbaren Farben darstellen, aus denen ausgewählt werden kann.
     * @param totalLayers Die Gesamtzahl der Ebenen in der Struktur, die zur Skalierung der Farbauswahl verwendet werden.
     * @return Die {@code Farbe}, die dem angegebenen Layer-Index entspricht.
     * @see Color
     */
    private static Color getColorForLayer(int layer, Color[] colors, int totalLayers) {
        int colorIndex = (layer * (colors.length - 1)) / Math.max(1, totalLayers - 1);
        return colors[colorIndex % colors.length];
    }
}
