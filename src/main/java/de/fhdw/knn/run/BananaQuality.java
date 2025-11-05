package de.fhdw.knn.run;

import de.fhdw.knn.classification.ClassificationScorer;
import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.TrainTestSplit;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.util.FutureUtil;

import java.io.IOException;

public class BananaQuality {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("banana_quality.csv", 0, 7, 7, 1);
        TrainTestSplit trainTest = data.shuffleAndSplit(42, 0.2);
        DataSet train = trainTest.train;
        DataSet test = trainTest.test;

        long trainStart = System.currentTimeMillis();
        Network network = new Network(42, 7, 100, 100, 1);
        //network = Importer.importNetwork("bq.knn");

        LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
        StopFunction stopFunction = new EarlyStopping(0.0001, 8);
        OptimizationFunction optimizationFunction = new GradientDescent(network, lossFunction, 0.05);

        Trainer trainer = new Trainer(network, 20, true, 1, lossFunction, stopFunction, optimizationFunction);
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
        FutureUtil.EXECUTOR.close();
    }

}
