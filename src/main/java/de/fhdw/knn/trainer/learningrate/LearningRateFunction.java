package de.fhdw.knn.trainer.learningrate;

/**
 * Die Learning Rate gibt an, wie stark eine Epoche (bzw. konkret jede Daten-Zeile innerhalb einer Epoche)
 * die Gewichte und den Bias der Neuronen beeinflusst. Dazu wird der Gradient beim Optimierungsalgorithmus Gradient Descent
 * mit diesem Faktor multipliziert, um den Einfluss in der Regel zu verringern (0 < Learning Rate < 1).<br>
 * Die LearningRateFunction gibt für jede Epoche die zu verwendende Learning Rate an.
 */
public interface LearningRateFunction {

    /**
     * Anhand der aktuellen Epoche und des Total-Loss wird die
     *
     * @param epoch        Aktuelle Epoche (beginnt mit 1)
     * @param previousLoss Ergebnis der Verlustfunktion über den gesamten Trainings-Datensatz in der vorherigen Epoche (NaN in 1. Epoche)
     * @see LearningRateFunction
     */
    double calc(int epoch, double previousLoss);

}
