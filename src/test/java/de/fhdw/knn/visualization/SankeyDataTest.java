package de.fhdw.knn.visualization;

class SankeyDataTest {

//    @Generated("Claude Anthropic AI")
//    @Test
//    void sankeyDataConvertsSimpleNetwork() {
//        InputLayer inputLayer = new InputLayer(2);
//        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 3, 2);
//        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);
//
//        assertDoesNotThrow(() -> SankeyData.convertNetworkToItems(network));
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void sankeyDataReturnsNonEmptyList() {
//        InputLayer inputLayer = new InputLayer(2);
//        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 3, 2);
//        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);
//
//        List<PlotItem> items = SankeyData.convertNetworkToItems(network);
//        assertFalse(items.isEmpty());
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void sankeyDataItemCountMatchesNeurons() {
//        InputLayer inputLayer = new InputLayer(2);
//        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 3, 2);
//        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);
//
//        List<PlotItem> items = SankeyData.convertNetworkToItems(network);
//        // 2 Input + 3 Hidden + 2 Output = 7
//        assertEquals(7, items.size());
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void sankeyDataSingleLayerNetwork() {
//        InputLayer inputLayer = new InputLayer(2);
//        DenseLayer[] denseLayers = DenseLayer.createLayers(null, ActivationFunction.LINEAR, 1);
//        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);
//
//        List<PlotItem> items = SankeyData.convertNetworkToItems(network);
//        // 2 Input + 1 Output = 3
//        assertEquals(3, items.size());
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void sankeyDataLargeNetworkDoesNotThrow() {
//        InputLayer inputLayer = new InputLayer(5);
//        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 10, 8, 4, 2);
//        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);
//
//        assertDoesNotThrow(() -> SankeyData.convertNetworkToItems(network));
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void sankeyDataAllColorSchemesUsed() {
//        // totalLayers > layerColors.length um den Math.min Pfad zu testen
//        InputLayer inputLayer = new InputLayer(2);
//        DenseLayer[] denseLayers = DenseLayer.createLayers(
//                ActivationFunction.LINEAR, ActivationFunction.LINEAR,
//                3, 3, 3, 3, 3, 3, 3, 3, 2 // 9 Hidden + 1 Output = 10 Layer -> mehr als 8 Farben
//        );
//        Network network = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);
//
//        assertDoesNotThrow(() -> SankeyData.convertNetworkToItems(network));
//    }
}