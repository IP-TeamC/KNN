package de.fhdw.knn.visualization;

import java.awt.*;

/**
 * Definiert Farbschemata für die Heatmap-Visualisierung von Gewichtsmatrizen.
 *
 * <p>Jedes Schema besteht aus zwei Farben, die jeweils negative und positive Gewichtswerte
 * repräsentieren. Die tatsächliche Farbe eines Feldes wird durch lineare Interpolation
 * zwischen Weiß (Wert = 0) und der jeweiligen Zielfarbe (maximaler Absolutwert) berechnet.
 *
 * @see ColorScheme#GREEN_RED
 * @see ColorScheme#BLUE_RED
 * @see ColorScheme#MONOCHROME
 */
public enum ColorScheme {

    /**
     * Negative Gewichte werden in Grün, positive in Rot dargestellt.
     */
    GREEN_RED(Color.GREEN, Color.RED),

    /**
     * Negative Gewichte werden in Blau, positive in Rot dargestellt.
     */
    BLUE_RED(Color.BLUE, Color.RED),

    /**
     * Sowohl negative als auch positive Gewichte werden in Schwarz dargestellt.
     */
    MONOCHROME(Color.BLACK, Color.BLACK),

    /**
     * Negative Gewichte werden "nicht" dargestellt, positive in Rot dargestellt.
     */
    WHITE_RED(Color.WHITE, Color.RED),

    /**
     * Negative Gewichte werden in Blau dargestellt, positive "nicht" dargestellt.
     */
    BLUE_WHITE(Color.BLUE, Color.WHITE);

    /**
     * Die Farbe, die für negative Gewichtswerte verwendet wird.
     * Bei einem Wert von 0 wird Weiß angezeigt; mit zunehmendem Absolutwert wird linear zu dieser Farbe interpoliert.
     */
    public final Color negative;

    /**
     * Die Farbe, die für positive Gewichtswerte verwendet wird.
     * Bei einem Wert von 0 wird Weiß angezeigt; mit zunehmendem Absolutwert wird linear zu dieser Farbe interpoliert.
     */
    public final Color positive;

    /**
     * Erstellt ein neues {@code ColorScheme} mit den angegebenen Farben für negative und positive Gewichtswerte.
     *
     * @param negative die Farbe für negative Gewichtswerte.
     * @param positive die Farbe für positive Gewichtswerte.
     */
    ColorScheme(Color negative, Color positive) {
        this.negative = negative;
        this.positive = positive;
    }
}
