package de.fhdw.knn.run;

import de.fhdw.knn.scorer.ClassificationScorer;
import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.TrainTestSplit;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.scorer.Score;
import de.fhdw.knn.scorer.Scorer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.learningrate.LearningRateFunction;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.visualization.HeatmapData;
import de.fhdw.knn.visualization.HeatmapView;
import de.fhdw.knn.visualization.SankeyView;
import javafx.application.Platform;

import java.io.IOException;

class BananaQuality {

    public static void main(String[] args) throws IOException {
        /// Datenaufbereitung
        DataSet data = CsvReader.readFile("data/banana_quality.csv", 0, 7, 7, 1);
        /// optional Inputs (oder Outputs) normalisieren
        // MinMaxNormalizer data = new MinMaxNormalizer(-10, 10);
        // train.normalizeInputs(data);

        /// Train-/Test-Split erzeugen
        TrainTestSplit trainTest = data.shuffleAndSplit(42, 0.2);
        DataSet train = trainTest.train;
        DataSet test = trainTest.test;


        /// Netzwerk erzeugen
        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(
                ActivationFunction.SWISH, ActivationFunction.SIGMOID,
                100, 100, 1);
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM,
                7, denseLayers);
        /// alternativ vorhandenes Netzwerk importieren
        // network = Importer.importNetwork("models/bq.knn");

        /// Loss-, Stop- und Optimization-Funktion für das Training festlegen
        LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
        StopFunction stopFunction = new EarlyStopping(0.002, 5);
        LearningRateFunction learningRateFunction = new ConstantLearningRate(0.03);
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, learningRateFunction);

        /// Trainer mit Mini-Batching definieren und starten
        Trainer trainer = new Trainer(network, 30, true, 10,
                lossFunction, stopFunction, optimizationFunction);
        trainer.train(train);
        long trainStop = System.currentTimeMillis();

        /// Evaluierung des trainierten Netzwerks
        long testStart = System.currentTimeMillis();
        Scorer scorer = new ClassificationScorer(network);
        Score score = scorer.score(test);
        long testStop = System.currentTimeMillis();

        /// Ausgabe der Ergebnisse und optionaler Export des Netzwerks
        score.print();
        // network.export("models/bq_demo.knn");

        /// Ausgabe der Trainings-/Test-Dauer
        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);

        /// Visualisierung des Netzwerks als Heatmap
        HeatmapData heatmapData = new HeatmapData(network);
        HeatmapView window = new HeatmapView();
        window.addHeatmap(heatmapData, "Manuelle Gewichtsmatrix");

        /// Visualisierung des Netzwerks als Sankey-Plot
        try {
            Platform.startup(() -> {
            }); // JavaFx initialisieren
        } catch (IllegalStateException ignored) {
        } // Ignorieren, falls es schon läuft
        SankeyView sankeyView = new SankeyView();
        sankeyView.show(network);
    }

}
