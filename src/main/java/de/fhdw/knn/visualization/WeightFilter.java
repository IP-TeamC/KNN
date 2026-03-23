package de.fhdw.knn.visualization;

/**
 * Definiert, welche Gewichte bei der Visualisierung berücksichtigt werden sollen.
 */
public enum WeightFilter {
    /**
     * Nur positive Gewichte anzeigen.
     */
    POSITIVE,
    /**
     * Nur negative Gewichte anzeigen.
     */
    NEGATIVE,
    /**
     * Positive und negative Gewichte anzeigen.
     */
    BOTH
}