package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.util.TestUtil;
import eu.hansolo.fx.charts.data.PlotItem;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SankeyDataTest {

    private static final double NO_THRESHOLD = 0.0;

    @Generated("Claude AI")
    @Test
    void sankeyDataConvertsSimpleNetwork() {
        InputLayer inputLayer = new InputLayer(2);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 3, 2);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        assertDoesNotThrow(() -> new SankeyData(network).convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH));
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataReturnsNonEmptyList() {
        InputLayer inputLayer = new InputLayer(2);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 3, 2);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        List<PlotItem> items = new SankeyData(network).convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH);
        assertFalse(items.isEmpty());
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataItemCountMatchesNeurons() {
        InputLayer inputLayer = new InputLayer(2);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 3, 2);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        List<PlotItem> items = new SankeyData(network).convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH);
        // 2 Input + 3 Hidden + 2 Output = 7
        assertEquals(7, items.size());
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataSingleLayerNetwork() {
        InputLayer inputLayer = new InputLayer(2);
        DenseLayer[] denseLayers = DenseLayer.createLayers(null, ActivationFunction.LINEAR, 1);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        List<PlotItem> items = new SankeyData(network).convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH);
        // 2 Input + 1 Output = 3
        assertEquals(3, items.size());
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataLargeNetworkDoesNotThrow() {
        InputLayer inputLayer = new InputLayer(5);
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 10, 8, 4, 2);
        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);

        assertDoesNotThrow(() -> new SankeyData(network).convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH));
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

        assertDoesNotThrow(() -> new SankeyData(network).convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH));
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataThresholdZeroIncludesAllConnections() {
        Network network = TestUtil.simpleDummyNetwork();
        List<PlotItem> itemsAll = new SankeyData(network).convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH);
        List<PlotItem> itemsHigh = new SankeyData(network).convertToPlotItems(Double.MAX_VALUE, WeightFilter.BOTH);

        // Mit threshold=0 müssen Verbindungen vorhanden sein; mit sehr hohem Threshold keine
        long outgoingAll = itemsAll.stream().mapToLong(i -> i.getOutgoing().size()).sum();
        long outgoingNone = itemsHigh.stream().mapToLong(i -> i.getOutgoing().size()).sum();

        assertTrue(outgoingAll > outgoingNone);
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataHighThresholdFiltersAllConnections() {
        Network network = TestUtil.simpleDummyNetwork();
        List<PlotItem> items = new SankeyData(network).convertToPlotItems(Double.MAX_VALUE, WeightFilter.BOTH);

        long totalOutgoing = items.stream().mapToLong(i -> i.getOutgoing().size()).sum();
        assertEquals(0, totalOutgoing);
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataFilterBothHasAtLeastAsManyConnectionsAsPositive() {
        Network network = TestUtil.complexDummyNetwork();
        SankeyData sankeyData = new SankeyData(network);

        long both = sankeyData.convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH)
                .stream().mapToLong(i -> i.getOutgoing().size()).sum();
        long positive = sankeyData.convertToPlotItems(NO_THRESHOLD, WeightFilter.POSITIVE)
                .stream().mapToLong(i -> i.getOutgoing().size()).sum();

        assertTrue(both >= positive);
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataFilterBothHasAtLeastAsManyConnectionsAsNegative() {
        Network network = TestUtil.complexDummyNetwork();
        SankeyData sankeyData = new SankeyData(network);

        long both = sankeyData.convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH)
                .stream().mapToLong(i -> i.getOutgoing().size()).sum();
        long negative = sankeyData.convertToPlotItems(NO_THRESHOLD, WeightFilter.NEGATIVE)
                .stream().mapToLong(i -> i.getOutgoing().size()).sum();

        assertTrue(both >= negative);
    }

    @Generated("Claude AI")
    @Test
    void sankeyDataPositiveAndNegativeFiltersSumToAtMostBoth() {
        Network network = TestUtil.complexDummyNetwork();
        SankeyData sankeyData = new SankeyData(network);

        long both = sankeyData.convertToPlotItems(NO_THRESHOLD, WeightFilter.BOTH)
                .stream().mapToLong(i -> i.getOutgoing().size()).sum();
        long positive = sankeyData.convertToPlotItems(NO_THRESHOLD, WeightFilter.POSITIVE)
                .stream().mapToLong(i -> i.getOutgoing().size()).sum();
        long negative = sankeyData.convertToPlotItems(NO_THRESHOLD, WeightFilter.NEGATIVE)
                .stream().mapToLong(i -> i.getOutgoing().size()).sum();

        assertEquals(both, positive + negative);
    }
}