package de.fhdw.knn.trainer;

import de.fhdw.knn.util.TestUtil;
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
import de.fhdw.knn.visualization.ColorScheme;
import de.fhdw.knn.visualization.ViewManager;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

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
        ViewManager viewManager = new ViewManager(10, 10, false, false, ColorScheme.RED_GREEN, outer);
        AtomicBoolean earlyStoppingOnTime = new AtomicBoolean(false);
        Trainer trainer = new Trainer(outer, 2, true, 1, null,
                (loss) -> earlyStoppingOnTime.getAndSet(true), optimizationFunction, viewManager);
        double initialWeight = outer.denseLayers[0].neurons[0].incoming[0].weight;
        trainer.train(data);
        assertNotEquals(initialWeight, outer.denseLayers[0].neurons[0].incoming[0].weight);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testTrainingReducesLoss() {
        // Einfaches Netz (1→1, Linear) soll die Funktion y=2*x annäherungsweise erlernen.
        // Nach ausreichend vielen Epochen muss der Loss kleiner sein als vor dem Training.
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 1,
                DenseLayer.createLayers(null, ActivationFunction.LINEAR, 1));

        DataSet data = new DataSet(
                new double[][]{{1.0}, {2.0}, {3.0}, {4.0}},
                new double[][]{{2.0}, {4.0}, {6.0}, {8.0}}
        );

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        double initialLoss = lossFunction.totalLoss(data.outputs, network.predict(data.inputs));

        OptimizationFunction opt = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));
        Trainer trainer = new Trainer(network, 200, false, 1, LossFunction.MEAN_SQUARED_ERROR,
                EarlyStopping.NEVER, opt);
        trainer.train(data);

        double finalLoss = lossFunction.totalLoss(data.outputs, network.predict(data.inputs));
        assertTrue(finalLoss < initialLoss,
                "Training sollte den Loss verringern. Vorher: " + initialLoss + ", Nachher: " + finalLoss);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testShuffleChangesDataOrder() {
        // DataSet.shuffle() muss die Reihenfolge der Zeilen verändern
        DataSet data = new DataSet(
                new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}},
                new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}}
        );
        double[] originalOrder = new double[5];
        for (int i = 0; i < 5; i++) {
            originalOrder[i] = data.inputs[i][0];
        }

        data.shuffle(42);

        boolean changed = false;
        for (int i = 0; i < 5; i++) {
            if (data.inputs[i][0] != originalOrder[i]) {
                changed = true;
                break;
            }
        }
        assertTrue(changed, "shuffle() sollte die Reihenfolge der Einträge ändern");
    }

    @Generated("GitHub Copilot")
    @Test
    public void testBatchedTrainingReducesLoss() {
        // Auch Mini-Batching (batchSize > 1) soll den Loss über mehrere Epochen verringern
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 1,
                DenseLayer.createLayers(null, ActivationFunction.LINEAR, 1));

        DataSet data = new DataSet(
                new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}},
                new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}}
        );

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        double initialLoss = lossFunction.totalLoss(data.outputs, network.predict(data.inputs));

        OptimizationFunction opt = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));
        Trainer trainer = new Trainer(network, 200, false, 4, LossFunction.MEAN_SQUARED_ERROR,
                EarlyStopping.NEVER, opt);
        trainer.train(data);

        double finalLoss = lossFunction.totalLoss(data.outputs, network.predict(data.inputs));
        assertTrue(finalLoss < initialLoss,
                "Batched Training sollte den Loss verringern. Vorher: " + initialLoss + ", Nachher: " + finalLoss);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testBatchSizeLargerThanDataSizeDoesNotFail() {
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 1,
                DenseLayer.createLayers(null, ActivationFunction.LINEAR, 1));
        DataSet data = new DataSet(
                new double[][]{{1.0}, {2.0}},
                new double[][]{{1.0}, {2.0}}
        );

        OptimizationFunction opt = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));
        Trainer trainer = new Trainer(network, 3, false, 10, LossFunction.MEAN_SQUARED_ERROR,
                EarlyStopping.NEVER, opt);

        assertDoesNotThrow(() -> trainer.train(data));
    }

    @Generated("GitHub Copilot")
    @Test
    public void testBatchedTrainingWithRemainderDoesNotReuseOldAdjustments() {
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 1,
                DenseLayer.createLayers(null, ActivationFunction.LINEAR, 1));
        DataSet data = new DataSet(
                new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}},
                new double[][]{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}}
        );
        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        double initialLoss = lossFunction.totalLoss(data.outputs, network.predict(data.inputs));

        OptimizationFunction opt = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));
        Trainer trainer = new Trainer(network, 30, false, 2, lossFunction,
                EarlyStopping.NEVER, opt);

        assertDoesNotThrow(() -> trainer.train(data));
        double finalLoss = lossFunction.totalLoss(data.outputs, network.predict(data.inputs));
        assertTrue(finalLoss < initialLoss,
                "Training mit Rest-Batch sollte stabil laufen und den Loss senken. Vorher: "
                        + initialLoss + ", Nachher: " + finalLoss);
    }
}
