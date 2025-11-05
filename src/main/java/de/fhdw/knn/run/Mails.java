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

public class Mails {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("emails.csv", 1, 3000, 3001, 1);
        TrainTestSplit trainTest = data.shuffleAndSplit(42, 0.2);
        DataSet train = trainTest.train;
        DataSet test = trainTest.test;

        long trainStart = System.currentTimeMillis();
        Network network = new Network(42, 3000, 300, 300, 1);
        //network = Importer.importNetwork("mails_best.knn");

        LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
        StopFunction stopFunction = new EarlyStopping(0.0001, 8);
        OptimizationFunction optimizationFunction = new GradientDescent(network, lossFunction, 0.01);

        Trainer trainer = new Trainer(network, 100, true, 1, lossFunction, stopFunction, optimizationFunction);
        trainer.train(train, "target/models/mails_%d.knn", 1);
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
