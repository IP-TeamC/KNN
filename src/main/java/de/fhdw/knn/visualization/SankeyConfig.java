package de.fhdw.knn.visualization;

/**
 * Unveränderliche Konfiguration für die Sankey-Visualisierung.
 *
 * @param interval     Aktualisierungsintervall in Epochen. {@code 0} = deaktiviert, {@code -1} = nur am Trainingsende.
 * @param threshold    Verbindungen mit {@code |weight| ≤ threshold} werden nicht dargestellt.
 * @param weightFilter Gibt an, ob positive, negative oder alle Gewichte dargestellt werden.
 */
public record SankeyConfig(
        int interval,
        double threshold,
        WeightFilter weightFilter
) {

    /**
     * Erstellt eine {@code SankeyConfig} mit dem angegebenen Intervall und Standardwerten
     * für alle übrigen Parameter.
     *
     * <p>Standardwerte: {@code threshold = 0.1}, {@code weightFilter = BOTH}.
     *
     * @param interval Aktualisierungsintervall in Epochen.
     *                 {@code 0} = deaktiviert, {@code -1} = nur am Trainingsende.
     * @return eine neue {@code SankeyConfig}-Instanz mit Standardwerten.
     */
    public static SankeyConfig withDefaults(int interval) {
        return new SankeyConfig(interval, 0.1, WeightFilter.BOTH);
    }
}