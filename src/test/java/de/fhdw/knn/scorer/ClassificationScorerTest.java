package de.fhdw.knn.scorer;

import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.Pair;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import org.junit.jupiter.api.Test;
import org.opentest4j.TestAbortedException;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

public class ClassificationScorerTest {

    public Pair<DataSet, Network> setUp() {
        try {
            DataSet dataSet = CsvReader.readFile("data/scorer_training_dataset.csv", 0, 1, 1, 1, 1);

            DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SIGMOID, ActivationFunction.SIGMOID, 10, 1);
            Network network = new Network(42, WeightInitializer.HE_UNIFORM, 1, denseLayers);

            LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
            StopFunction stopFunction = EarlyStopping.NEVER;
            OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(0.01));

            Trainer trainer = new Trainer(network, 50, true, 1, lossFunction, stopFunction, optimizationFunction);
            trainer.train(dataSet);

            return new Pair<>(dataSet, network);
        } catch (IOException e){
            throw new TestAbortedException("Alarm im ClassificationScorerTest Setup");
        }
    }

    @Test
    public void testConfusionMatrix(){
        Pair<DataSet, Network> testDataAndNetwork = this.setUp();
        ClassificationScorer classificationScorer = new ClassificationScorer(testDataAndNetwork.y);

        ClassificationScorer.Score result = classificationScorer.score(testDataAndNetwork.x);

        assertEquals(234, result.truePositives);
        assertEquals(182, result.trueNegatives);
        assertEquals(54, result.falsePositives);
        assertEquals(32, result.falseNegatives);
    }

    @Test
    public void testAccuracyRecallErrorPrecisionF1(){
        ClassificationScorer.Score score = new ClassificationScorer.Score(234, 182, 54, 32);
        assertEquals(0.8287, score.accuracy, 0.0001);
        assertEquals(0.1713, score.error, 0.0001);
        assertEquals(0.8125, score.precision);
        assertEquals(0.8797, score.recall, 0.0001);
        assertEquals(0.8448, score.f1, 0.0001);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testPrintFunction(){
        ClassificationScorer.Score score = new ClassificationScorer.Score(234, 182, 54, 32);
        
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(outputStream));
        
        try {
            score.print();
            System.setOut(originalOut);
            String output = outputStream.toString();
            
            assertTrue(output.contains("TP:"), "Ausgabe sollte 'TP:' enthalten");
            assertTrue(output.contains("TN:"), "Ausgabe sollte 'TN:' enthalten");
            assertTrue(output.contains("FP:"), "Ausgabe sollte 'FP:' enthalten");
            assertTrue(output.contains("FN:"), "Ausgabe sollte 'FN:' enthalten");
            assertTrue(output.contains("Accuracy:"), "Ausgabe sollte 'Accuracy:' enthalten");
            assertTrue(output.contains("Error:"), "Ausgabe sollte 'Error:' enthalten");
            assertTrue(output.contains("Precision:"), "Ausgabe sollte 'Precision:' enthalten");
            assertTrue(output.contains("Recall:"), "Ausgabe sollte 'Recall:' enthalten");
            assertTrue(output.contains("F1:"), "Ausgabe sollte 'F1:' enthalten");
        } finally {
            System.setOut(originalOut);
        }
    }
}

