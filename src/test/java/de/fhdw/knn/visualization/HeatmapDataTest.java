package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.util.TestUtil;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

class HeatmapDataTest {

    @Generated("Claude AI")
    @Test
    void heatmapDataMatrixHasCorrectSize() {
        Network network = TestUtil.simpleDummyNetwork();
        HeatmapData data = new HeatmapData(network);
        double[][] matrix = data.buildFullWeightMatrix();

        // Test, ob die Matrix die erwartete Größe hat (1 Input + 1 Output = 2 Neuronen → 2x2 Matrix)
        assertEquals(2, matrix.length, "Die Anzahl der Zeilen der Gewichtsmatrix entspricht nicht der erwarteten Größe.");
        assertEquals(2, matrix[0].length, "Die Anzahl der Spalten der Gewichtsmatrix entspricht nicht der erwarteten Größe.");
    }

    @Generated("Claude AI")
    @Test
    void heatmapDataMatrixDiagonalIsNaN() {
        Network network = TestUtil.simpleDummyNetwork();
        HeatmapData data = new HeatmapData(network);
        double[][] matrix = data.buildFullWeightMatrix();

        // Test, ob die Diagonale der Matrix NaN ist (Neuronen verbinden sich nicht mit sich selbst)
        assertTrue(Double.isNaN(matrix[0][0]), "Der diagonale Wert der Gewichtsmatrix sollte NaN sein.");
    }

    @Generated("Claude AI")
    @Test
    void heatmapDataConnectionWeightIsSet() {
        Network network = TestUtil.simpleDummyNetwork();
        HeatmapData data = new HeatmapData(network);
        double[][] matrix = data.buildFullWeightMatrix();

        // Test, ob die Verbindung zwischen Input → Output gesetzt ist (nicht NaN)
        assertFalse(Double.isNaN(matrix[0][1]), "Die Verbindung zwischen Input und Output sollte einen Wert haben und nicht NaN sein.");
    }

    @Generated("Claude AI")
    @Test
    void heatmapDataLabelsHaveCorrectCount() {
        Network network = TestUtil.simpleDummyNetwork();
        HeatmapData data = new HeatmapData(network);
        String[] labels = data.getNeuronLabels();

        // Test, ob die korrekte Anzahl der Labels generiert wird
        assertEquals(2, labels.length, "Die Anzahl der Neuronen-Labels entspricht nicht der erwarteten Anzahl.");
    }

    @Generated("Claude AI")
    @Test
    void heatmapDataLabelsHaveCorrectFormat() {
        Network network = TestUtil.simpleDummyNetwork();
        HeatmapData data = new HeatmapData(network);
        String[] labels = data.getNeuronLabels();

        // Test, ob die Labels das erwartete Format haben
        assertEquals("I-N1", labels[0], "Das Label für den ersten Input-Neuron entspricht nicht dem erwarteten Format.");
        assertEquals("O-N1", labels[1], "Das Label für den ersten Output-Neuron entspricht nicht dem erwarteten Format.");
    }

    @Generated("Claude AI")
    @Test
    void heatmapDataLargerNetworkLabels() {
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 2,
                DenseLayer.createLayers(null, ActivationFunction.LINEAR, 3, 1));
        HeatmapData data = new HeatmapData(network);
        String[] labels = data.getNeuronLabels();

        // Test, ob die Labels für ein größeres Netzwerk korrekt generiert werden (2 Input + 3 Hidden + 1 Output = 6)
        assertEquals(6, labels.length, "Die Anzahl der Neuronen-Labels für das größere Netzwerk entspricht nicht der erwarteten Anzahl.");
        assertEquals("I-N1", labels[0], "Das Label für den ersten Input-Neuron entspricht nicht dem erwarteten Format.");
        assertEquals("I-N2", labels[1], "Das Label für den zweiten Input-Neuron entspricht nicht dem erwarteten Format.");
        assertEquals("H1-N1", labels[2], "Das Label für das erste Hidden-Neuron der ersten Hidden-Layer entspricht nicht dem erwarteten Format.");
        assertEquals("O-N1", labels[5], "Das Label für den ersten Output-Neuron entspricht nicht dem erwarteten Format.");
    }
}
