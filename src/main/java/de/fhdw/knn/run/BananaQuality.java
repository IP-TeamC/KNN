package de.fhdw.knn.run;

import de.fhdw.knn.classification.ClassificationScorer;
import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.TrainTestSplit;
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
import de.fhdw.knn.visualization.HeatmapData;
import de.fhdw.knn.visualization.HeatmapWindow;

import java.io.IOException;

public class BananaQuality {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("data/banana_quality.csv", 0, 7, 7, 1);
        TrainTestSplit trainTest = data.shuffleAndSplit(42, 0.2);
        DataSet train = trainTest.train;
        DataSet test = trainTest.test;

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SWISH, ActivationFunction.SIGMOID, 100, 100, 1);
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 7, denseLayers);
        //network = Importer.importNetwork("models/bq.knn");

        LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
        StopFunction stopFunction = EarlyStopping.NEVER;
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(0.06));

        Trainer trainer = new Trainer(network, 20, true, 1, null, stopFunction, optimizationFunction);
        trainer.train(train);
        long trainStop = System.currentTimeMillis();

        long testStart = System.currentTimeMillis();
        ClassificationScorer scorer = new ClassificationScorer(network);
        ClassificationScorer.Score score = scorer.score(test);
        long testStop = System.currentTimeMillis();

        score.print();

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);

        HeatmapData heatmapData = new HeatmapData(network);
        HeatmapWindow window = new HeatmapWindow();
        window.showSingleMatrix("Manuelle Gewichtsmatrix", heatmapData);
    }

}
