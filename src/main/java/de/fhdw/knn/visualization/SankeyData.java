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
 */
public class SankeyData {

    /**
     * Stellt den Schwellenwert dar, der zur Bestimmung der Signifikanz bestimmter Verbindungen
     * im Zusammenhang mit Sankeyplot-Visualisierungen verwendet wird.
     *
     * <p>Der Schwellwert ist derzeit auf fest auf 0.1 codiert.
     */
    private static final double THRESHOLD = 0.1;

    /**
     * Konvertiert ein bestimmtes Netzwerk in eine Liste von {@code PlotItem} zur Visualisierung.
     * Das Netzwerk wird im Sankey-Diagramm für eine bessere Visualisierung der Gewichtsverteilung umgekehrt dargestellt.
     *
     * @param network Das neuronale Netzwerk, das Eingabe-, versteckte und Ausgabeschichten enthält,
     *                die in {@code PlotItems} umgewandelt werden sollen.
     *
     * @return Eine Liste von {@code PlotItems}, die alle Schichten (Eingabe, versteckt und Ausgabe) des angegebenen Netzwerks darstellen.
     *
     * @see PlotItem
     */
    public static List<PlotItem> convertNetworkToItems(Network network) {
        List<PlotItem> allItems = new ArrayList<>();

        Color[] layerColors = {
                Color.RED, Color.ORANGERED, Color.ORANGE, Color.GOLD,
                Color.LIGHTGREEN, Color.CYAN, Color.LIGHTBLUE, Color.BLUE,
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

        return allItems;
    }

    /**
     * Erstellt umgekehrte Verbindungen zwischen zwei Schichten von {@code PlotItems} basierend auf den Gewichten
     * der Neuronen in der angegebenen {@code DenseLayer} und unter Beachtung des {@code THRESHOLD}.
     *
     * @param rightItems Ein Array von {@code PlotItems}, das die rechte Schicht repräsentiert.
     * @param leftItems  Ein Array von {@code PlotItems}, das die linke Schicht repräsentiert.
     * @param rightLayer Die {@code DenseLayer}, die die Neuronen der rechten Schicht enthält,
     *                   einschließlich ihrer eingehenden Verbindungsgewichte von den Neuronen in der
     *                   linken Schicht.
     *
     * @see PlotItem
     * @see DenseLayer
     * @see SankeyData#THRESHOLD
     */
    private static void createConnectionsReversed(PlotItem[] rightItems, PlotItem[] leftItems, DenseLayer rightLayer) {
        for (int rIdx = 0; rIdx < rightItems.length; rIdx++) {
            AbstractDenseNeuron rightNeuron = rightLayer.neurons[rIdx];
            Connection[] incomingFromLeft = rightNeuron.incoming;

            for (int lIdx = 0; lIdx < leftItems.length; lIdx++) {
                double weight = Math.abs(incomingFromLeft[lIdx].weight);

                if (weight > THRESHOLD) {
                    rightItems[rIdx].addToOutgoing(leftItems[lIdx], weight);
                }
            }
        }
    }

    /**
     * Bestimmt die Farbe für eine bestimmte Ebene basierend auf dem angegebenen Farbarray und der Gesamtzahl der Ebenen.
     *
     * @param layer Der Index der Ebene, für die die Farbe bestimmt werden soll.
     * @param colors Ein Array von {@code Color}-Objekten, die die verfügbaren Farben darstellen, aus denen ausgewählt werden kann.
     * @param totalLayers Die Gesamtzahl der Ebenen in der Struktur, die zur Skalierung der Farbauswahl verwendet werden.
     *
     * @return Die {@code Farbe}, die dem angegebenen Layer-Index entspricht.
     *
     * @see Color
     */
    private static Color getColorForLayer(int layer, Color[] colors, int totalLayers) {
        int colorIndex = (layer * (colors.length - 1)) / Math.max(1, totalLayers - 1);
        return colors[Math.min(colorIndex, colors.length - 1)];
    }
}
