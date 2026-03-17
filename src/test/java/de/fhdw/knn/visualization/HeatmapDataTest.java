package de.fhdw.knn.visualization;

class HeatmapDataTest {

//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapDataMatrixHasCorrectSize() {
//        Network network = TestUtil.simpleDummyNetwork();
//        HeatmapData data = new HeatmapData(network);
//        double[][] matrix = data.buildFullWeightMatrix();
//
//        // 1 Input + 1 Output = 2 Neuronen → 2x2 Matrix
//        assertEquals(2, matrix.length);
//        assertEquals(2, matrix[0].length);
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapDataMatrixDiagonalIsNaN() {
//        Network network = TestUtil.simpleDummyNetwork();
//        HeatmapData data = new HeatmapData(network);
//        double[][] matrix = data.buildFullWeightMatrix();
//
//        // Neuronen verbinden sich nicht mit sich selbst
//        assertTrue(Double.isNaN(matrix[0][0]));
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapDataConnectionWeightIsSet() {
//        Network network = TestUtil.simpleDummyNetwork();
//        HeatmapData data = new HeatmapData(network);
//        double[][] matrix = data.buildFullWeightMatrix();
//
//        // Input → Output Verbindung muss gesetzt sein (nicht NaN)
//        assertFalse(Double.isNaN(matrix[0][1]));
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapDataLabelsHaveCorrectCount() {
//        Network network = TestUtil.simpleDummyNetwork();
//        HeatmapData data = new HeatmapData(network);
//        String[] labels = data.getNeuronLabels();
//
//        assertEquals(2, labels.length);
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapDataLabelsHaveCorrectFormat() {
//        Network network = TestUtil.simpleDummyNetwork();
//        HeatmapData data = new HeatmapData(network);
//        String[] labels = data.getNeuronLabels();
//
//        assertEquals("I-N1", labels[0]);
//        assertEquals("O-N1", labels[1]);
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapDataLargerNetworkLabels() {
//        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 2,
//                DenseLayer.createLayers(null, ActivationFunction.LINEAR, 3, 1));
//        HeatmapData data = new HeatmapData(network);
//        String[] labels = data.getNeuronLabels();
//
//        // 2 Input + 3 Hidden + 1 Output = 6
//        assertEquals(6, labels.length);
//        assertEquals("I-N1",  labels[0]);
//        assertEquals("I-N2",  labels[1]);
//        assertEquals("H1-N1", labels[2]);
//        assertEquals("O-N1",  labels[5]);
//    }
}
