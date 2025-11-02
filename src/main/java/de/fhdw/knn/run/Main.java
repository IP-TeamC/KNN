package de.fhdw.knn.run;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.normalizer.MinMaxNormalizer;
import de.fhdw.knn.reader.CsvReader;
import de.fhdw.knn.trainer.Trainer;

import java.io.IOException;
import java.util.Map;

public class Main {

    public static void main(String[] args) throws IOException {
        Network network = new Network(new InputLayer(7), new DenseLayer(50), new DenseLayer(50), new DenseLayer(1));
        Trainer trainer = new Trainer(network, 30, 0.1);
        Map.Entry<double[][], double[][]> data = CsvReader.readFile("banana_quality.csv", 0, 7, 7, 1);
        for (int i = 0; i < data.getKey().length; i++) {
            double[] input = data.getKey()[i];
            double[] output = data.getValue()[i];
            int pos = (int) (Math.random() * data.getKey().length);
            data.getKey()[i] = data.getKey()[pos];
            data.getValue()[i] = data.getValue()[pos];
            data.getKey()[pos] = input;
            data.getValue()[pos] = output;
        }
        //MinMaxNormalizer normalizer = new MinMaxNormalizer(0, 1);
        //normalizer.normalize(data.getValue());
        trainer.train(
                data.getKey(),
                data.getValue()
        );

        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < data.getKey().length; i++) {
            double[] rawPrediction = network.feedForward(data.getKey()[i]);
            //normalizer.denormalize(new double[][]{rawPrediction});
            double prediction = rawPrediction[0];
            double expected = data.getValue()[i][0];
            if (expected > 0.9 && prediction >= 0.5) {
                tp += 1;
            } else if (expected > 0.9 && prediction < 0.5) {
                fn += 1;
            } else if (expected < 0.1 && prediction >= 0.5) {
                fp += 1;
            } else if (expected < 0.1 && prediction < 0.5) {
                tn += 1;
            }
            //System.out.println("Expected: " + expected + ", Predicted: " + prediction);
        }
        System.out.println("TP: " + tp);
        System.out.println("TN: " + tn);
        System.out.println("FP: " + fp);
        System.out.println("FN: " + fn);

//        for (int i = -10; i <= 10; i++) {
//            double[] rawPrediction = network.feedForward(new double[]{i});
//            normalizer.denormalize(new double[][]{rawPrediction});
//            double prediction = rawPrediction[0];
//            System.out.println("Expected: " + (i * i) + ", Predicted: " + prediction);
//        }

//        double[] predictions1 = new double[]{network.feedForward(data[0])[0], network.feedForward(data[1])[0], network.feedForward(data[2])[0], network.feedForward(data[3])[0]};
//        double loss = MeanSquaredError.DEFAULT.loss(new double[]{0.98, 0.95, 0.01, 0.2}, predictions1);
//        System.out.println("Expected: 0.98, Predicted: " + network.feedForward(data[0])[0]);
//        System.out.println("Expected: 0.95, Predicted: " + network.feedForward(data[1])[0]);
//        System.out.println("Expected: 0.01, Predicted: " + network.feedForward(data[2])[0]);
//        System.out.println("Expected: 0.2, Predicted: " + network.feedForward(data[3])[0]);
//        System.out.println("Loss: " + loss);
    }

}
