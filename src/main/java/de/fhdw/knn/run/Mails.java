package de.fhdw.knn.run;

import de.fhdw.knn.scorer.ClassificationScorer;
import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.TrainTestSplit;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.io.Importer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.StopFunction;

import java.io.IOException;

class Mails {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("data/emails.csv", 1, 3000, 3001, 1);
        TrainTestSplit trainTest = data.shuffleAndSplit(42, 0.2);
        DataSet train = trainTest.train;
        DataSet test = trainTest.test;

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SWISH, ActivationFunction.SIGMOID, 300, 300, 1);
        Network network = new Network(42, WeightInitializer.HE, 3000, denseLayers);
        network = Importer.importNetwork("models/mails_swish_97.knn");
        //network = Importer.importNetwork("models/mails_91.knn");
        //network = Importer.importNetwork("models/mails_98.knn");

        LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
        StopFunction stopFunction = StopFunction.NEVER;
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(0.00001));

        Trainer trainer = new Trainer(network, 100, true, 1, null, stopFunction, optimizationFunction);
        //trainer.train(train);
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
    }

}
