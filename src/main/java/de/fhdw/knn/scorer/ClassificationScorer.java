package de.fhdw.knn.scorer;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;

/**
 * Evaluiert das Netzwerk für Klassifikationsprobleme anhand folgender Metriken:
 * True Positives - True Negatives - False Positives - False Negatives
 * - Accuracy - Error - Precision - Recall - F1-Score
 * @param network
 */
public record ClassificationScorer(Network network) implements Scorer {

    /**
     * Evaluierung anhand des übergebenen Datensatzes
     * @param data Test-Datensatz
     */
    public Score score(DataSet data) {
        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        for (int i = 0; i < data.size; i++) {
            double[][] outputs = network.feedForward(data.inputs[i]).output();
            double prediction = outputs[outputs.length - 1][0];
            double expected = data.outputs[i][0];
            if (prediction >= 0.5) {
                if (expected == 1) {
                    truePositives += 1;
                } else {
                    falsePositives += 1;
                }
            } else if (expected == 0) {
                trueNegatives += 1;
            } else {
                falseNegatives += 1;
            }
        }
        return new Score(truePositives, trueNegatives, falsePositives, falseNegatives);
    }

    /**
     * Enthält die Metriken des {@link ClassificationScorer}
     * @see ClassificationScorer
     */
    public static class Score implements de.fhdw.knn.scorer.Score {
        public final int truePositives;
        public final int trueNegatives;
        public final int falsePositives;
        public final int falseNegatives;

        public final double accuracy;
        public final double error;
        public final double precision;
        public final double recall;
        public final double f1;

        /**
         * Berechnet die zusätzlichen Metriken auf Basis der Parameter
         * @see ClassificationScorer
         */
        public Score(int truePositives, int trueNegatives, int falsePositives, int falseNegatives) {
            this.truePositives = truePositives;
            this.trueNegatives = trueNegatives;
            this.falsePositives = falsePositives;
            this.falseNegatives = falseNegatives;
            int sum = truePositives + trueNegatives + falsePositives + falseNegatives;

            this.accuracy = ((double) (truePositives + trueNegatives)) / sum;
            this.error = ((double) (falsePositives + falseNegatives)) / sum;
            this.precision = ((double) truePositives) / (truePositives + falsePositives);
            this.recall = ((double) truePositives) / (truePositives + falseNegatives);
            this.f1 = 2 * precision * recall / (precision + recall);
        }

        /**
         * Gibt alle Metriken im Terminal aus
         * @see ClassificationScorer
         */
        public void print() {
            System.out.println();
            System.out.println("TP: " + truePositives);
            System.out.println("TN: " + trueNegatives);
            System.out.println("FP: " + falsePositives);
            System.out.println("FN: " + falseNegatives);
            System.out.println();
            System.out.println("Accuracy: " + accuracy);
            System.out.println("Error: " + error);
            System.out.println("Precision: " + precision);
            System.out.println("Recall: " + recall);
            System.out.println("F1: " + f1);
            System.out.println();
        }

    }

}
