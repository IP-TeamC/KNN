package de.fhdw.knn.trainer;

import de.fhdw.knn.TestUtil;
import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.SuperNeuron;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class TrainerTest {

    @Test
    public void testWithExportStopNoShuffleBatched() throws IOException {
        OptimizationFunction optimizationFunction = new GradientDescent(LossFunction.MEAN_ABSOLUTE_ERROR, new ConstantLearningRate(0.01));
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 7,
                DenseLayer.createLayers(null, ActivationFunction.SIGMOID, 1));
        Trainer trainer = new Trainer(network, 500, false, 50, LossFunction.MEAN_ABSOLUTE_ERROR,
                new EarlyStopping(100, 5), optimizationFunction);

        DataSet data = CsvReader.readFile("data/banana_quality.csv", 0, 7, 7, 1);
        double[] firstInput = data.inputs[0];
        trainer.train(data, "target/test-train-export-%d.knn", 5);

        assertEquals(firstInput, data.inputs[0]);
        Path pathExists = Path.of("target/test-train-export-5.knn");
        assertTrue(Files.isRegularFile(pathExists));
        assertFalse(Files.isRegularFile(Path.of("target/test-train-export-10.knn")));
        Files.delete(pathExists);

        Trainer trainer2 = new Trainer(network, 1, false, 50, LossFunction.MEAN_ABSOLUTE_ERROR,
                new EarlyStopping(100, 5), optimizationFunction);
        trainer2.train(data, "target/test-train-export-%d.knn", 5);
        Path pathExists2 = Path.of("target/test-train-export-1.knn");
        assertTrue(Files.isRegularFile(pathExists2));
        Files.delete(pathExists2);

        Trainer trainer3 = new Trainer(network, 2, false, 50, null,
                EarlyStopping.NEVER, optimizationFunction);
        trainer3.train(data, "target/test-train-export-%d.knn", 0);
        Path pathExists3 = Path.of("target/test-train-export-2.knn");
        assertFalse(Files.isRegularFile(pathExists3));
    }

    @Test
    public void testWithSuperNeuron() throws IOException {
        Network outer = TestUtil.simpleDummyNetwork();
        Network inner = TestUtil.simpleDummyNetwork();
        SuperNeuron sn = new SuperNeuron(inner);
        sn.insert(outer, 0, 0);

        DataSet data = CsvReader.readFile("data/banana_quality.csv", 0, 1, 7, 1);
        OptimizationFunction optimizationFunction = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));
        Trainer trainer = new Trainer(outer, 2, true, 1, null,
                EarlyStopping.NEVER, optimizationFunction);
        double initialWeight = outer.denseLayers[0].neurons[0].incoming[0].weight;
        trainer.train(data);
        assertNotEquals(initialWeight, outer.denseLayers[0].neurons[0].incoming[0].weight);
    }

}
