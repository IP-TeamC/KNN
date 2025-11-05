package de.fhdw.knn.util;

import de.fhdw.knn.classification.ClassificationScorer;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.io.Importer;

import java.util.Map;

public class BestModel {

    public static Map.Entry<String, Double> find(DataSet test, String... fileNames) {
        String best = null;
        double f1 = 0;
        for (String fileName : fileNames) {
            Network network;
            if (fileName.endsWith(".knn")) {
                network = new Importer().loadText(fileName);
            } else if (fileName.endsWith(".knn2")) {
                network = Importer.importNetwork(fileName);
            } else {
                throw new IllegalArgumentException("unknown file extension");
            }

            ClassificationScorer scorer = new ClassificationScorer(network);
            ClassificationScorer.Score score = scorer.score(test);
            score.print();
            if (score.f1 > f1) {
                best = fileName;
                f1 = score.f1;
            }
        }
        FutureUtil.EXECUTOR.close();
        return Map.entry(best, f1);
    }

}
