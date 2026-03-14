package de.fhdw.knn.network.io;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.OutputDerived;
import de.fhdw.knn.network.neuron.SuperNeuron;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

public class ExporterImporterTest {

    @Test
    public void testExportImport() {
        Network inner = new Network(42, WeightInitializer.GLOROT_UNIFORM, 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.SNAKE));
        SuperNeuron sn = new SuperNeuron(inner, new double[]{3}, new double[][]{new double[]{1, 0, 0, 0, 0, 2, 0, 0, 0, 0}});

        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 10,
                DenseLayer.createLayers(ActivationFunction.SWISH, ActivationFunction.SIGMOID, 20, 5));
        sn.insert(network, 0, 12);
        network.denseLayers[0].neurons[10].incoming[3].guard = false;
        network.export("target/test-export.knn");

        Network imported = Importer.importNetwork("target/test-export.knn");
        double[][] inputs = new double[100][];
        for (int i = 0; i < inputs.length; i++) {
            double[] input = new double[network.inputLayer.neurons.length];
            inputs[i] = input;
            for (int j = 0; j < input.length; j++) {
                input[j] = Math.random();
            }
        }

        double[][] expected = network.predict(inputs);
        double[][] compare = imported.predict(inputs);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], compare[i]);
        }
    }

    private static class FakeNeuron extends AbstractDenseNeuron {

        @Override
        public OutputDerived compute(double[] input) {
            return null;
        }

    }

    @Test
    public void testExportFail() {
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 10,
                DenseLayer.createLayers(ActivationFunction.SWISH, ActivationFunction.SIGMOID, 20, 5));
        network.denseLayers[0].neurons[1] = new FakeNeuron();
        assertThrowsExactly(IllegalArgumentException.class, () -> network.export("target/test-export.knn"));
    }

    @Test
    public void testImportFail() {
        assertThrows(IOException.class, () -> Importer.importNetwork("src/main/java/invalid-file.knn"));
    }

    @Generated("GitHub Copilot")
    @Test
    public void testImportMalformedDataFails() {
        Importer importer = new Importer();
        byte[] malformed = new byte[]{0, 0, 0};
        assertThrows(RuntimeException.class, () -> importer.load(malformed));
    }

}
