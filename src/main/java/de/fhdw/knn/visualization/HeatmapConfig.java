package de.fhdw.knn.visualization;

/**
 * Unveränderliche Konfiguration für die Heatmap-Visualisierung.
 *
 * @param interval        Aktualisierungsintervall in Epochen. {@code 0} = deaktiviert, {@code -1} = nur am Trainingsende.
 * @param threshold       Verbindungen mit {@code |weight| ≤ threshold} werden nicht dargestellt.
 * @param weightFilter    Gibt an, ob positive, negative oder alle Gewichte dargestellt werden.
 * @param showWeights     {@code true}, um Gewichtswerte als Text anzuzeigen.
 * @param normalizeColors {@code true} normalisiert die Farbskala auf den tatsächlichen Wertebereich.
 * @param colorScheme     Das Farbschema für die Heatmap.
 */
public record HeatmapConfig(
        int interval,
        double threshold,
        WeightFilter weightFilter,
        boolean showWeights,
        boolean normalizeColors,
        ColorScheme colorScheme
) {

    /**
     * Erstellt eine {@code HeatmapConfig} mit dem angegebenen Intervall und Standardwerten
     * für alle übrigen Parameter.
     *
     * <p>Standardwerte: {@code threshold = 0.1}, {@code weightFilter = BOTH},
     * {@code showWeights = true}, {@code normalizeColors = true},
     * {@code colorScheme = GREEN_RED}.
     *
     * @param interval Aktualisierungsintervall in Epochen.
     *                 {@code 0} = deaktiviert, {@code -1} = nur am Trainingsende.
     * @return eine neue {@code HeatmapConfig}-Instanz mit Standardwerten.
     */
    public static HeatmapConfig withDefaults(int interval) {
        return new HeatmapConfig(interval, 0.1, WeightFilter.BOTH, true, true, ColorScheme.GREEN_RED);
    }
}