package de.fhdw.knn.run;

import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.SoftstartLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;

import java.io.IOException;

public class F2 {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("f2_full.csv", 0, 1, 2, 1);
        data.shuffle(42);

        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.RELU, ActivationFunction.LINEAR, 200, 200, 200, 1);
        Network network = new Network(42, WeightInitializer.HE, 1, denseLayers);
        //network = Importer.importNetwork("f2.knn");

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        StopFunction stopFunction = new EarlyStopping(0.0001, 100);
        OptimizationFunction optimizationFunction = new GradientDescent(network, lossFunction, new SoftstartLearningRate(0.00005, 0.3, 5));

        Trainer trainer = new Trainer(network, 50, true, 1, lossFunction, stopFunction, optimizationFunction);
        trainer.train(data);

        double[][] x = new double[21][];
        for (int i = -10; i <= 10; i++) {
            x[i + 10] = new double[]{i};
        }
        double[][] predictions = network.predict(x);

        for (int i = 0; i < x.length; i++) {
            System.out.printf("Expected %.2f, Predicted %.2f%n", x[i][0] * x[i][0], predictions[i][0]);
        }
        System.out.println();
    }

}
