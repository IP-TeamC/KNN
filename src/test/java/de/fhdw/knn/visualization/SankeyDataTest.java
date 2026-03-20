package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import eu.hansolo.fx.charts.data.PlotItem;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SankeyDataTest {

    @Generated("Claude AI")
    @Test
    void sankeyDataConvertsSimpleNetwork() {
        InputLayer inputLayer = new InputLayer(2);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 3, 2);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        assertDoesNotThrow(() -> SankeyData.convertNetworkToItems(network));
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataReturnsNonEmptyList() {
        InputLayer inputLayer = new InputLayer(2);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 3, 2);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        List<PlotItem> items = SankeyData.convertNetworkToItems(network);
        assertFalse(items.isEmpty());
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataItemCountMatchesNeurons() {
        InputLayer inputLayer = new InputLayer(2);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 3, 2);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        List<PlotItem> items = SankeyData.convertNetworkToItems(network);
        // 2 Input + 3 Hidden + 2 Output = 7
        assertEquals(7, items.size());
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataSingleLayerNetwork() {
        InputLayer inputLayer = new InputLayer(2);
        DenseLayer[] denseLayers = DenseLayer.createLayers(null, ActivationFunction.LINEAR, 1);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        List<PlotItem> items = SankeyData.convertNetworkToItems(network);
        // 2 Input + 1 Output = 3
        assertEquals(3, items.size());
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataLargeNetworkDoesNotThrow() {
        InputLayer inputLayer = new InputLayer(5);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 10, 8, 4, 2);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        assertDoesNotThrow(() -> SankeyData.convertNetworkToItems(network));
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataAllColorSchemesUsed() {
        // totalLayers > layerColors.length um den Math.min Pfad zu testen
        InputLayer inputLayer = new InputLayer(2);
        DenseLayer[] denseLayers = DenseLayer.createLayers(
                ActivationFunction.LINEAR, ActivationFunction.LINEAR,
                3, 3, 3, 3, 3, 3, 3, 3, 2 // 9 Hidden + 1 Output = 10 Layer -> mehr als 8 Farben
        );
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        assertDoesNotThrow(() -> SankeyData.convertNetworkToItems(network));
    }
}