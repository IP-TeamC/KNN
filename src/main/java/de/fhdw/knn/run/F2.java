package de.fhdw.knn.run;

import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.MinMaxNormalizer;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.io.Importer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.util.FutureUtil;

import java.io.IOException;

public class F2 {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("f2_full.csv", 0, 1, 2, 3);
        MinMaxNormalizer normalizer = new MinMaxNormalizer(0, 1);
        data.normalizeOutputs(normalizer);

        Network network = new Network(42, 1, 200, 300, 200, 3);
        network = Importer.importNetwork("f2_200x300x200_50.knn");

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        StopFunction stopFunction = new EarlyStopping(0.0001, 100);
        OptimizationFunction optimizationFunction = new GradientDescent(network, lossFunction, 0.1);

        Trainer trainer = new Trainer(network, 50, true, 1, lossFunction, stopFunction, optimizationFunction);
        //trainer.train(data);

        double[][] x = new double[21][];
        for (int i = -10; i <= 10; i++) {
            x[i + 10] = new double[]{i};
        }
        double[][] predictions = network.predict(x);
        normalizer.denormalize(predictions);

        for (int i = 0; i < x.length; i++) {
            System.out.printf("Expected %.2f, Predicted %.2f%n", x[i][0] * x[i][0], predictions[i][0]);
        }
        System.out.println();

        FutureUtil.EXECUTOR.close();
    }

}
