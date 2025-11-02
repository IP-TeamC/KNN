package de.fhdw.knn.run;

import de.fhdw.knn.classification.ClassificationScorer;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.TrainTestSplit;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.util.FutureUtil;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("banana_quality.csv", 0, 7, 7, 1);
        TrainTestSplit trainTest = data.shuffleAndSplit(42, 0.2);
        DataSet train = trainTest.train;
        DataSet test = trainTest.test;

        long trainStart = System.currentTimeMillis();
        Network network = new Network(42, 7, 2, 100, 1);

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        StopFunction stopFunction = new EarlyStopping(0.001, 5);
        OptimizationFunction optimizationFunction = new GradientDescent(network, LossFunction.MEAN_SQUARED_ERROR, 0.05);

        Trainer trainer = new Trainer(network, 100, lossFunction, stopFunction, optimizationFunction);
        trainer.train(train);
        long trainStop = System.currentTimeMillis();

        long testStart = System.currentTimeMillis();
        ClassificationScorer scorer = new ClassificationScorer(network);
        ClassificationScorer.Score score = scorer.score(test);
        long testStop = System.currentTimeMillis();

        System.out.println();
        System.out.println("TP: " + score.truePositives);
        System.out.println("TN: " + score.trueNegatives);
        System.out.println("FP: " + score.falsePositives);
        System.out.println("FN: " + score.falseNegatives);
        System.out.println();
        System.out.println("Accuracy: " + score.accuracy);
        System.out.println("Error: " + score.error);
        System.out.println("Precision: " + score.precision);
        System.out.println("Recall: " + score.recall);
        System.out.println("F1: " + score.f1);
        System.out.println();

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);
        FutureUtil.EXECUTOR.close();
    }

}
